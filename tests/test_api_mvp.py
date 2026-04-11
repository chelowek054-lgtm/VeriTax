from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.application.ask_chat import AskResult, AskSource
from app.application.index_exports import ExportIndexStats, IndexRunSummary
from app.main import app


def test_ask_endpoint_mocked() -> None:
    src = AskSource(
        thread_id=1,
        root_message_id=1,
        message_ids=[10],
        source_namespace="ns",
        source_chat="Chat",
        date_from="2026-01-01",
        date_to="2026-01-02",
        authors=["u"],
        text="ctx",
        score=0.88,
    )
    with TestClient(app) as client:
        with patch(
            "app.api.v1.endpoints.ask.ask_chat_default_clients",
            new_callable=AsyncMock,
            return_value=AskResult(answer="Ответ теста", sources=[src]),
        ):
            r = client.post("/api/v1/ask", json={"question": "Вопрос?"})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "Ответ теста"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["text"] == "ctx"


async def _fake_run_index(settings, *, on_progress=None):  # noqa: ARG001
    from app.application.index_exports import IndexProgressSnapshot

    if on_progress:
        await on_progress(
            IndexProgressSnapshot(
                exports_total=2,
                exports_done=0,
                current_source_namespace=None,
                phase="running",
                threads_indexed_cumulative=0,
                current_export_relative_path=None,
                errors=(),
            )
        )
        await on_progress(
            IndexProgressSnapshot(
                exports_total=2,
                exports_done=2,
                current_source_namespace="done",
                phase="processing",
                threads_indexed_cumulative=3,
                current_export_relative_path="x/result.json",
                errors=(),
            )
        )
    return IndexRunSummary(
        exports_processed=2,
        threads_indexed=3,
        errors=[],
        exports=[
            ExportIndexStats(
                path="x/result.json",
                source_namespace="done",
                threads_total=1,
                threads_indexed=1,
                messages_loaded=5,
            )
        ],
    )


def test_index_job_lifecycle_mocked() -> None:
    import time

    with TestClient(app) as client:
        with patch(
            "app.application.index_jobs.run_index_all_exports",
            new_callable=AsyncMock,
            side_effect=_fake_run_index,
        ):
            start = client.post("/api/v1/index/jobs")
            assert start.status_code == 200
            job_id = start.json()["job_id"]
            last = None
            for _ in range(80):
                last = client.get(f"/api/v1/index/jobs/{job_id}")
                assert last.status_code == 200
                st = last.json().get("status")
                if st in ("completed", "failed"):
                    break
                time.sleep(0.02)
            body = last.json() if last is not None else {}
    assert body.get("job_id") == job_id
    assert body.get("status") == "completed"
    assert body.get("exports_total") == 2
    assert body.get("result") is not None
    assert body["result"]["threads_indexed"] == 3


def test_index_job_not_found() -> None:
    with TestClient(app) as client:
        r = client.get("/api/v1/index/jobs/00000000-0000-0000-0000-000000000099")
    assert r.status_code == 404
