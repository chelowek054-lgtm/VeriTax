from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingService(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text (same order)."""
        ...


@runtime_checkable
class VectorIndex(Protocol):
    async def ensure_collection(self, *, vector_size: int) -> None: ...

    async def upsert_points(
        self,
        *,
        ids: list[Any],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None: ...

    async def search_similar(
        self,
        *,
        query_vector: list[float],
        top_k: int,
        source_namespace: str | None = None,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Return list of (score, payload) sorted by relevance (higher is better for cosine)."""
        ...
