from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import ask, index_jobs, indexing

api_router = APIRouter()
api_router.include_router(indexing.router)
api_router.include_router(ask.router)
api_router.include_router(index_jobs.router)
