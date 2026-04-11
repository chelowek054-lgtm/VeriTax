from __future__ import annotations

import asyncio

from app.adapters.ollama_embeddings import embed_texts_batched_resilient


class FakeEmbeddingService:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if any("boom" in text for text in texts):
            raise RuntimeError("synthetic ollama 500")
        return [[float(len(text))] for text in texts]


def test_embed_texts_batched_resilient_skips_only_failed_texts() -> None:
    service = FakeEmbeddingService()

    vectors, failures = asyncio.run(
        embed_texts_batched_resilient(
            service,
            ["ok-1", "boom-text", "ok-2"],
            batch_size=3,
            max_concurrent=1,
        )
    )

    assert vectors[0] == [4.0]
    assert vectors[1] is None
    assert vectors[2] == [4.0]
    assert len(failures) == 1
    assert failures[0].index == 1
    assert "500" in failures[0].error
