from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException

from app.api.v1.schemas import AskRequest, AskResponse, AskSourceDTO
from app.application.ask_chat import ask_chat_default_clients
from app.core.config import get_settings

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
async def ask_question(body: AskRequest) -> AskResponse:
    settings = get_settings()
    ns = body.source_namespace
    if ns is not None and ns.strip() == "":
        ns = None
    try:
        result = await ask_chat_default_clients(
            settings,
            question=body.question,
            source_namespace=ns,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Сервис Ollama или сеть недоступны: {exc}",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AskResponse(
        answer=result.answer,
        sources=[
            AskSourceDTO(
                thread_id=s.thread_id,
                root_message_id=s.root_message_id,
                message_ids=s.message_ids,
                source_namespace=s.source_namespace,
                source_chat=s.source_chat,
                date_from=s.date_from,
                date_to=s.date_to,
                authors=s.authors,
                text=s.text,
                score=s.score,
            )
            for s in result.sources
        ],
    )
