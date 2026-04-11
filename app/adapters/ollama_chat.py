from __future__ import annotations

from typing import Any

import httpx

from app.core.config import Settings


class OllamaChatService:
    """Non-streaming chat completion via Ollama `/api/chat`."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._settings.ollama_base_url,
                timeout=httpx.Timeout(180.0, connect=15.0),
            )
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat(self, system_prompt: str, user_message: str) -> str:
        client = await self._get_client()
        payload: dict[str, Any] = {
            "model": self._settings.ollama_chat_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        resp = await client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()
        raise RuntimeError(f"Unexpected Ollama chat response: {data!r}")
