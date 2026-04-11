from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.v1.schemas import (
    ExportIndexStatsDTO,
    IndexJobCreateResponse,
    IndexJobStatusResponse,
    IndexRunResponse,
)
from app.application.index_jobs import IndexJobRegistry, IndexJobState

router = APIRouter(tags=["indexing"])


def _registry(request: Request) -> IndexJobRegistry:
    reg = getattr(request.app.state, "index_jobs", None)
    if reg is None:
        raise HTTPException(status_code=500, detail="Index job registry is not initialized")
    return reg


def _state_to_response(state: IndexJobState) -> IndexJobStatusResponse:
    summary = state.summary
    result: IndexRunResponse | None = None
    if summary is not None:
        result = IndexRunResponse(
            exports_processed=summary.exports_processed,
            threads_indexed=summary.threads_indexed,
            errors=summary.errors,
            exports=[
                ExportIndexStatsDTO(
                    path=e.path,
                    source_namespace=e.source_namespace,
                    threads_total=e.threads_total,
                    threads_indexed=e.threads_indexed,
                    messages_loaded=e.messages_loaded,
                )
                for e in summary.exports
            ],
        )
    return IndexJobStatusResponse(
        job_id=state.job_id,
        status=state.status,
        phase=state.phase,
        exports_total=state.exports_total,
        exports_done=state.exports_done,
        current_source_namespace=state.current_source_namespace,
        threads_indexed=state.threads_indexed,
        current_export_relative_path=state.current_export_relative_path,
        current_export_threads_total=state.current_export_threads_total,
        current_export_threads_done=state.current_export_threads_done,
        errors=state.errors,
        fatal_error=state.fatal_error,
        result=result,
    )


@router.post("/index/jobs", response_model=IndexJobCreateResponse)
async def start_index_job(request: Request) -> IndexJobCreateResponse:
    from app.core.config import get_settings

    settings = get_settings()
    job_id = await _registry(request).create_and_start(settings)
    return IndexJobCreateResponse(job_id=job_id)


@router.get("/index/jobs/{job_id}", response_model=IndexJobStatusResponse)
async def get_index_job(request: Request, job_id: str) -> IndexJobStatusResponse:
    state = await _registry(request).get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _state_to_response(state)
