from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _env_str(key: str, default: str) -> str:
    val = os.getenv(key)
    return val.strip() if val and val.strip() else default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    """Environment-driven settings (no pydantic-settings dependency)."""

    app_env: str
    session_secret: str
    data_dir: Path
    ollama_base_url: str
    ollama_embed_model: str
    ollama_chat_model: str
    qdrant_url: str
    qdrant_collection: str
    embedding_dimension: int
    embed_batch_size: int
    embed_max_concurrent: int
    thread_text_max_chars: int
    rag_top_k: int
    ask_context_max_chars: int


@lru_cache
def get_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = Path(_env_str("DATA_DIR", str(repo_root / ".data"))).resolve()
    return Settings(
        app_env=_env_str("APP_ENV", "development"),
        session_secret=_env_str("SESSION_SECRET", "dev-change-me-in-production"),
        data_dir=data_dir,
        ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://host.docker.internal:11434").rstrip("/"),
        ollama_embed_model=_env_str("OLLAMA_EMBED_MODEL", "bge-m3"),
        ollama_chat_model=_env_str("OLLAMA_CHAT_MODEL", "qwen2.5:3b"),
        qdrant_url=_env_str("QDRANT_URL", "http://localhost:6333").rstrip("/"),
        qdrant_collection=_env_str("QDRANT_COLLECTION", "veritax_threads"),
        embedding_dimension=_env_int("EMBEDDING_DIMENSION", 1024),
        embed_batch_size=max(1, _env_int("EMBED_BATCH_SIZE", 16)),
        embed_max_concurrent=max(1, _env_int("EMBED_MAX_CONCURRENT", 4)),
        thread_text_max_chars=max(1000, _env_int("THREAD_TEXT_MAX_CHARS", 12000)),
        rag_top_k=max(1, min(50, _env_int("RAG_TOP_K", 5))),
        ask_context_max_chars=max(2000, _env_int("ASK_CONTEXT_MAX_CHARS", 24000)),
    )
