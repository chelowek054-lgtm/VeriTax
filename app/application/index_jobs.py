from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.application.index_exports import IndexProgressSnapshot, IndexRunSummary, run_index_all_exports
from app.core.config import Settings


@dataclass
class IndexJobState:
    job_id: str
    status: str  # pending | running | completed | failed
    exports_total: int = 0
    exports_done: int = 0
    current_source_namespace: str | None = None
    phase: str = "idle"
    threads_indexed: int = 0
    current_export_relative_path: str | None = None
    current_export_threads_total: int = 0
    current_export_threads_done: int = 0
    errors: list[str] = field(default_factory=list)
    fatal_error: str | None = None
    summary: IndexRunSummary | None = None


class IndexJobRegistry:
    """In-memory index jobs for single-process MVP."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._jobs: dict[str, IndexJobState] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def create_and_start(self, settings: Settings) -> str:
        job_id = str(uuid.uuid4())
        state = IndexJobState(job_id=job_id, status="pending")
        async with self._lock:
            self._jobs[job_id] = state
        task = asyncio.create_task(self._run_job(job_id, settings))
        self._tasks[job_id] = task

        def _cleanup(t: asyncio.Task[None]) -> None:
            self._tasks.pop(job_id, None)

        task.add_done_callback(_cleanup)
        return job_id

    async def get(self, job_id: str) -> IndexJobState | None:
        async with self._lock:
            st = self._jobs.get(job_id)
            if st is None:
                return None
            return self._copy_state(st)

    def _copy_state(self, st: IndexJobState) -> IndexJobState:
        summary = st.summary
        if summary is not None:
            summary = IndexRunSummary(
                exports_processed=summary.exports_processed,
                threads_indexed=summary.threads_indexed,
                errors=list(summary.errors),
                exports=list(summary.exports),
            )
        return IndexJobState(
            job_id=st.job_id,
            status=st.status,
            exports_total=st.exports_total,
            exports_done=st.exports_done,
            current_source_namespace=st.current_source_namespace,
            phase=st.phase,
            threads_indexed=st.threads_indexed,
            current_export_relative_path=st.current_export_relative_path,
            current_export_threads_total=st.current_export_threads_total,
            current_export_threads_done=st.current_export_threads_done,
            errors=list(st.errors),
            fatal_error=st.fatal_error,
            summary=summary,
        )

    async def _update(self, job_id: str, **kwargs: Any) -> None:
        async with self._lock:
            st = self._jobs.get(job_id)
            if st is None:
                return
            for k, v in kwargs.items():
                setattr(st, k, v)

    async def _apply_snapshot(self, job_id: str, snap: IndexProgressSnapshot) -> None:
        await self._update(
            job_id,
            exports_total=snap.exports_total,
            exports_done=snap.exports_done,
            current_source_namespace=snap.current_source_namespace,
            phase=snap.phase,
            threads_indexed=snap.threads_indexed_cumulative,
            current_export_relative_path=snap.current_export_relative_path,
            current_export_threads_total=snap.current_export_threads_total,
            current_export_threads_done=snap.current_export_threads_done,
            errors=list(snap.errors),
        )

    async def _run_job(self, job_id: str, settings: Settings) -> None:
        await self._update(job_id, status="running", phase="starting")

        async def on_progress(snap: IndexProgressSnapshot) -> None:
            await self._apply_snapshot(job_id, snap)

        try:
            summary = await run_index_all_exports(settings, on_progress=on_progress)
            async with self._lock:
                st = self._jobs.get(job_id)
                if st is not None:
                    st.status = "completed"
                    st.phase = "done"
                    st.summary = summary
                    st.exports_done = summary.exports_processed
                    st.exports_total = max(st.exports_total, summary.exports_processed)
                    st.threads_indexed = summary.threads_indexed
                    st.errors = list(summary.errors)
        except Exception as exc:  # noqa: BLE001
            await self._update(
                job_id,
                status="failed",
                phase="error",
                fatal_error=str(exc),
            )
