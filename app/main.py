from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.api.v1.router import api_router
from app.application.index_jobs import IndexJobRegistry
from app.core.config import get_settings
from app.web.routes import router as web_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.index_jobs = IndexJobRegistry()
    yield


app = FastAPI(title="VeriTax API", version="0.1.0", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=settings.session_secret)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(web_router)
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
