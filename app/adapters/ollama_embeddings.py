from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)


class OllamaEmbeddingService:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._settings.ollama_base_url,
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = await self._get_client()
        url = "/api/embed"
        payload: dict[str, Any] = {
            "model": self._settings.ollama_embed_model,
            "input": texts,
        }
        logger.info(
            "Requesting embeddings from Ollama: model=%s batch_size=%s",
            self._settings.ollama_embed_model,
            len(texts),
        )
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError(
                f"Unexpected Ollama embed response: expected {len(texts)} embeddings, got {embeddings!r}"
            )
        out: list[list[float]] = []
        for row in embeddings:
            if not isinstance(row, list):
                raise RuntimeError(f"Invalid embedding row: {row!r}")
            out.append([float(x) for x in row])
        logger.info(
            "Received embeddings from Ollama: model=%s batch_size=%s vector_dim=%s",
            self._settings.ollama_embed_model,
            len(texts),
            len(out[0]) if out else 0,
        )
        return out


@dataclass(frozen=True, slots=True)
class EmbedFailure:
    index: int
    text_length: int
    error: str


EmbedProgressCallback = Callable[[int, int], Awaitable[None]]


async def embed_texts_batched(
    service: OllamaEmbeddingService,
    texts: list[str],
    *,
    batch_size: int,
    max_concurrent: int,
) -> list[list[float]]:
    """Embed in batches with limited concurrency."""
    if not texts:
        return []
    sem = asyncio.Semaphore(max_concurrent)
    batches: list[tuple[int, list[str]]] = []
    for i in range(0, len(texts), batch_size):
        batches.append((i, texts[i : i + batch_size]))

    async def run_batch(start: int, chunk: list[str]) -> tuple[int, list[list[float]]]:
        async with sem:
            vecs = await service.embed_texts(chunk)
            return start, vecs

    tasks = [run_batch(s, c) for s, c in batches]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x[0])
    merged: list[list[float]] = []
    for _, vecs in results:
        merged.extend(vecs)
    if len(merged) != len(texts):
        raise RuntimeError("Embedding batch merge size mismatch")
    return merged


async def embed_texts_batched_resilient(
    service: OllamaEmbeddingService,
    texts: list[str],
    *,
    batch_size: int,
    max_concurrent: int,
    on_progress: EmbedProgressCallback | None = None,
) -> tuple[list[list[float] | None], list[EmbedFailure]]:
    """
    Embed texts with graceful degradation.

    If Ollama fails on a batch, split the batch recursively down to single texts.
    Single texts that still fail are reported and skipped instead of aborting the
    whole export indexing run.
    """
    _ = max_concurrent  # currently processed sequentially for predictable fallback
    vectors: list[list[float] | None] = [None] * len(texts)
    failures: list[EmbedFailure] = []
    completed = 0

    async def report_progress(delta: int) -> None:
        nonlocal completed
        completed += delta
        if on_progress is not None:
            await on_progress(completed, len(texts))

    async def embed_chunk(start: int, chunk: list[str]) -> None:
        if not chunk:
            return
        try:
            rows = await service.embed_texts(chunk)
            for offset, row in enumerate(rows):
                vectors[start + offset] = row
            await report_progress(len(chunk))
            return
        except Exception as exc:  # noqa: BLE001 - preserve progress across bad inputs
            if len(chunk) == 1:
                failures.append(
                    EmbedFailure(
                        index=start,
                        text_length=len(chunk[0]),
                        error=str(exc).strip() or type(exc).__name__,
                    )
                )
                await report_progress(1)
                return

        mid = len(chunk) // 2
        await embed_chunk(start, chunk[:mid])
        await embed_chunk(start + mid, chunk[mid:])

    for i in range(0, len(texts), batch_size):
        await embed_chunk(i, texts[i : i + batch_size])

    return vectors, failures
