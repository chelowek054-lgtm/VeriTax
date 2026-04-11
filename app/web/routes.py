from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.application.index_exports import list_chat_exports
from app.core.config import get_settings

APP_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = APP_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(tags=["web"])


def _pop_flash(request: Request) -> tuple[str | None, str | None]:
    session = request.session
    message = session.pop("flash_message", None)
    kind = session.pop("flash_kind", None)
    if message is not None and not isinstance(message, str):
        message = str(message)
    if kind is not None and not isinstance(kind, str):
        kind = str(kind)
    return message, kind


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    settings = get_settings()
    flash_message, flash_kind = _pop_flash(request)
    chats = list_chat_exports(settings.data_dir)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "chats": chats,
            "data_dir": str(settings.data_dir),
            "flash_message": flash_message,
            "flash_kind": flash_kind,
        },
    )


