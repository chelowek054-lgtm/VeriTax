from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.schemas import ExportIndexStatsDTO, IndexRunResponse
from app.application.index_exports import run_index_all_exports
from app.core.config import get_settings

router = APIRouter(tags=["indexing"])


@router.post("/index", response_model=IndexRunResponse)
async def trigger_index() -> IndexRunResponse:
    settings = get_settings()
    summary = await run_index_all_exports(settings)
    return IndexRunResponse(
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
