from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app.core.config import Settings

logger = logging.getLogger(__name__)


class QdrantVectorIndex:
    def __init__(self, settings: Settings, client: AsyncQdrantClient | None = None) -> None:
        self._settings = settings
        self._client = client or AsyncQdrantClient(url=settings.qdrant_url)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()

    async def ensure_collection(self, *, vector_size: int) -> None:
        name = self._settings.qdrant_collection
        try:
            await self._client.get_collection(collection_name=name)
            logger.info("Qdrant collection exists: name=%s", name)
            return
        except Exception:
            pass
        await self._client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        )
        logger.info("Created Qdrant collection: name=%s vector_size=%s", name, vector_size)

    async def upsert_points(
        self,
        *,
        ids: list[Any],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        if len(ids) != len(vectors) or len(ids) != len(payloads):
            raise ValueError("ids, vectors, payloads length mismatch")
        points = [
            qm.PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        logger.info(
            "Upserting vectors to Qdrant: collection=%s points=%s",
            self._settings.qdrant_collection,
            len(points),
        )
        await self._client.upsert(
            collection_name=self._settings.qdrant_collection,
            points=points,
            wait=True,
        )
        logger.info(
            "Qdrant upsert completed: collection=%s points=%s",
            self._settings.qdrant_collection,
            len(points),
        )

    async def search_similar(
        self,
        *,
        query_vector: list[float],
        top_k: int,
        source_namespace: str | None = None,
    ) -> list[tuple[float, dict[str, Any]]]:
        name = self._settings.qdrant_collection
        query_filter: qm.Filter | None = None
        if source_namespace is not None and source_namespace.strip() != "":
            query_filter = qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="source_namespace",
                        match=qm.MatchValue(value=source_namespace),
                    )
                ]
            )
        response = await self._client.query_points(
            collection_name=name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        out: list[tuple[float, dict[str, Any]]] = []
        for hit in response.points:
            payload = hit.payload or {}
            if not isinstance(payload, dict):
                payload = dict(payload)  # type: ignore[arg-type]
            out.append((float(hit.score), payload))
        return out


def qdrant_point_uuid(source_namespace: str, root_message_id: int) -> str:
    """Deterministic UUID string for Qdrant point id (namespace-safe)."""
    ns = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # DNS namespace
    key = f"{source_namespace}\n{root_message_id}"
    return str(uuid.uuid5(ns, key))
