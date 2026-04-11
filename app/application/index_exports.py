from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.adapters.ollama_embeddings import (
    EmbedFailure,
    OllamaEmbeddingService,
    embed_texts_batched_resilient,
)
from app.adapters.qdrant_index import QdrantVectorIndex, qdrant_point_uuid
from app.adapters.telegram_export_loader import TelegramResultJsonLoader
from app.core.config import Settings
from app.domain.thread_builder import build_threads

logger = logging.getLogger(__name__)


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + "\n...[truncated]"


def _format_embed_failures(rel: str, failures: list[EmbedFailure]) -> str:
    sample = failures[0]
    return (
        f"{rel}: embedding failed for {len(failures)} thread(s); "
        f"first failed thread had {sample.text_length} chars; "
        f"reason: {sample.error}"
    )


def discover_exports(data_dir: Path) -> list[tuple[Path, str]]:
    """Return (result.json path, source_namespace) for each export under data_dir."""
    if not data_dir.is_dir():
        return []
    out: list[tuple[Path, str]] = []
    for p in sorted(data_dir.rglob("result.json")):
        if not p.is_file():
            continue
        ns = p.parent.relative_to(data_dir).as_posix()
        out.append((p, ns))
    return out


@dataclass(frozen=True, slots=True)
class ChatExportInfo:
    """One Telegram export folder under DATA_DIR (for UI listing)."""

    source_namespace: str
    relative_result_path: str
    indexed_status: str = "not_indexed"
    indexed_status_label: str = "не индексировался"
    last_indexed_at: str | None = None
    threads_indexed: int | None = None
    threads_total: int | None = None
    last_error: str | None = None


def _index_state_path(data_dir: Path) -> Path:
    return data_dir / ".veritax-index-state.json"


def _status_label(status: str) -> str:
    return {
        "indexed": "проиндексирован",
        "indexed_with_errors": "с ошибками",
        "failed": "сбой индексации",
        "stale": "изменён после индексации",
        "not_indexed": "не индексировался",
    }.get(status, status)


def _load_index_state(data_dir: Path) -> dict[str, dict[str, Any]]:
    path = _index_state_path(data_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, dict):
            out[key] = value
    return out


def _save_index_state(data_dir: Path, state: dict[str, dict[str, Any]]) -> None:
    path = _index_state_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def list_chat_exports(data_dir: Path) -> list[ChatExportInfo]:
    """List discovered `result.json` exports with stable namespace per folder."""
    out: list[ChatExportInfo] = []
    state = _load_index_state(data_dir)
    for path, ns in discover_exports(data_dir):
        rel = path.relative_to(data_dir).as_posix()
        record = state.get(rel, {})
        status = "not_indexed"
        last_indexed_at: str | None = None
        threads_indexed: int | None = None
        threads_total: int | None = None
        last_error: str | None = None
        if record:
            size, mtime_ns = _file_signature(path)
            file_size = record.get("file_size")
            file_mtime_ns = record.get("file_mtime_ns")
            last_indexed_at = record.get("last_indexed_at") if isinstance(record.get("last_indexed_at"), str) else None
            threads_indexed = int(record["threads_indexed"]) if isinstance(record.get("threads_indexed"), int) else None
            threads_total = int(record["threads_total"]) if isinstance(record.get("threads_total"), int) else None
            last_error = record.get("last_error") if isinstance(record.get("last_error"), str) else None
            if file_size != size or file_mtime_ns != mtime_ns:
                status = "stale"
            else:
                raw_status = record.get("status")
                status = raw_status if isinstance(raw_status, str) else "not_indexed"
        out.append(
            ChatExportInfo(
                source_namespace=ns,
                relative_result_path=rel,
                indexed_status=status,
                indexed_status_label=_status_label(status),
                last_indexed_at=last_indexed_at,
                threads_indexed=threads_indexed,
                threads_total=threads_total,
                last_error=last_error,
            )
        )
    return out


@dataclass
class ExportIndexStats:
    path: str
    source_namespace: str
    threads_total: int
    threads_indexed: int
    messages_loaded: int


@dataclass
class IndexRunSummary:
    exports_processed: int
    threads_indexed: int
    errors: list[str] = field(default_factory=list)
    exports: list[ExportIndexStats] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class IndexProgressSnapshot:
    exports_total: int
    exports_done: int
    current_source_namespace: str | None
    phase: str
    threads_indexed_cumulative: int
    current_export_relative_path: str | None
    errors: tuple[str, ...]
    current_export_threads_total: int = 0
    current_export_threads_done: int = 0


ProgressCallback = Callable[[IndexProgressSnapshot], Awaitable[None]]


async def run_index_all_exports(
    settings: Settings,
    *,
    on_progress: ProgressCallback | None = None,
) -> IndexRunSummary:
    """
    Scan `settings.data_dir` for `**/result.json`, build threads, embed via Ollama, upsert to Qdrant.
    """
    loader = TelegramResultJsonLoader()
    embedder = OllamaEmbeddingService(settings)
    index = QdrantVectorIndex(settings)
    summary = IndexRunSummary(exports_processed=0, threads_indexed=0)
    index_state = _load_index_state(settings.data_dir)

    exports = discover_exports(settings.data_dir)
    if not exports:
        summary.errors.append(f"No result.json found under {settings.data_dir}")
        if on_progress:
            await on_progress(
                IndexProgressSnapshot(
                    exports_total=0,
                    exports_done=0,
                    current_source_namespace=None,
                    phase="empty",
                    threads_indexed_cumulative=0,
                    current_export_relative_path=None,
                    errors=tuple(summary.errors),
                )
            )
        await embedder.aclose()
        await index.aclose()
        return summary

    if on_progress:
        await on_progress(
            IndexProgressSnapshot(
                exports_total=len(exports),
                exports_done=0,
                current_source_namespace=None,
                phase="running",
                threads_indexed_cumulative=0,
                current_export_relative_path=None,
                errors=tuple(summary.errors),
            )
        )

    collection_ready = False
    vector_size = settings.embedding_dimension

    async def emit_progress(
        *,
        exports_done: int,
        source_namespace: str | None,
        phase: str,
        current_export_relative_path: str | None,
        current_export_threads_total: int = 0,
        current_export_threads_done: int = 0,
    ) -> None:
        if on_progress:
            await on_progress(
                IndexProgressSnapshot(
                    exports_total=len(exports),
                    exports_done=exports_done,
                    current_source_namespace=source_namespace,
                    phase=phase,
                    threads_indexed_cumulative=summary.threads_indexed,
                    current_export_relative_path=current_export_relative_path,
                    errors=tuple(summary.errors),
                    current_export_threads_total=current_export_threads_total,
                    current_export_threads_done=current_export_threads_done,
                )
            )

    try:
        for path, source_namespace in exports:
            rel = path.relative_to(settings.data_dir).as_posix()
            file_size, file_mtime_ns = _file_signature(path)
            logger.info("Indexing export started: path=%s namespace=%s", rel, source_namespace)
            await emit_progress(
                exports_done=summary.exports_processed,
                source_namespace=source_namespace,
                phase="loading_export",
                current_export_relative_path=rel,
            )
            try:
                meta, messages = loader.load_file(path)
                logger.info("Loaded export: path=%s messages=%s", rel, len(messages))
                await emit_progress(
                    exports_done=summary.exports_processed,
                    source_namespace=source_namespace,
                    phase="building_threads",
                    current_export_relative_path=rel,
                )
                threads = build_threads(messages)
                to_index = [t for t in threads if t.text.strip()]
                logger.info(
                    "Built threads: path=%s total_threads=%s non_empty_threads=%s",
                    rel,
                    len(threads),
                    len(to_index),
                )
                stats = ExportIndexStats(
                    path=rel,
                    source_namespace=source_namespace,
                    threads_total=len(threads),
                    threads_indexed=0,
                    messages_loaded=len(messages),
                )
                summary.exports_processed += 1

                if not to_index:
                    logger.info("No non-empty threads to index: path=%s", rel)
                    summary.exports.append(stats)
                    index_state[rel] = {
                        "status": "indexed",
                        "source_namespace": source_namespace,
                        "relative_result_path": rel,
                        "file_size": file_size,
                        "file_mtime_ns": file_mtime_ns,
                        "last_indexed_at": datetime.now(UTC).isoformat(),
                        "threads_total": stats.threads_total,
                        "threads_indexed": stats.threads_indexed,
                        "messages_loaded": stats.messages_loaded,
                        "last_error": None,
                    }
                    _save_index_state(settings.data_dir, index_state)
                else:
                    texts = [_truncate(t.text, settings.thread_text_max_chars) for t in to_index]
                    logger.info(
                        "Embedding export threads: path=%s threads=%s batch_size=%s max_concurrent=%s",
                        rel,
                        len(texts),
                        settings.embed_batch_size,
                        settings.embed_max_concurrent,
                    )
                    await emit_progress(
                        exports_done=summary.exports_processed - 1,
                        source_namespace=source_namespace,
                        phase="embedding",
                        current_export_relative_path=rel,
                        current_export_threads_total=len(to_index),
                        current_export_threads_done=0,
                    )

                    async def on_embed_progress(done: int, total: int) -> None:
                        await emit_progress(
                            exports_done=summary.exports_processed - 1,
                            source_namespace=source_namespace,
                            phase="embedding",
                            current_export_relative_path=rel,
                            current_export_threads_total=total,
                            current_export_threads_done=done,
                        )

                    vector_rows, embed_failures = await embed_texts_batched_resilient(
                        embedder,
                        texts,
                        batch_size=settings.embed_batch_size,
                        max_concurrent=settings.embed_max_concurrent,
                        on_progress=on_embed_progress,
                    )
                    successful_items: list[tuple[object, list[float], dict]] = []
                    for i, t in enumerate(to_index):
                        vec = vector_rows[i]
                        if vec is None:
                            continue
                        pid = qdrant_point_uuid(source_namespace, t.root_message_id)
                        successful_items.append(
                            (
                                pid,
                                vec,
                                {
                                    "thread_id": t.thread_id,
                                    "root_message_id": t.root_message_id,
                                    "message_ids": list(t.message_ids),
                                    "text": _truncate(t.text, settings.thread_text_max_chars),
                                    "date_from": t.date_from.isoformat(),
                                    "date_to": t.date_to.isoformat(),
                                    "authors": list(t.authors),
                                    "source_chat": meta.chat_name or "",
                                    "source_chat_id": meta.chat_id,
                                    "source_namespace": source_namespace,
                                    "export_relative_path": (
                                        f"{source_namespace}/result.json"
                                        if source_namespace not in ("", ".")
                                        else "result.json"
                                    ),
                                    "message_count": t.message_count,
                                },
                            )
                        )

                    if embed_failures:
                        summary.errors.append(_format_embed_failures(rel, embed_failures))
                        logger.warning(
                            "Embedding finished with failures: path=%s success=%s failed=%s",
                            rel,
                            len(successful_items),
                            len(embed_failures),
                        )
                    else:
                        logger.info(
                            "Embedding finished successfully: path=%s embedded_threads=%s",
                            rel,
                            len(successful_items),
                        )

                    if successful_items:
                        vector_size = len(successful_items[0][1])
                        await emit_progress(
                            exports_done=summary.exports_processed - 1,
                            source_namespace=source_namespace,
                            phase="upserting",
                            current_export_relative_path=rel,
                            current_export_threads_total=len(to_index),
                            current_export_threads_done=len(successful_items),
                        )

                        if not collection_ready:
                            await index.ensure_collection(vector_size=vector_size)
                            collection_ready = True

                        ids = [item[0] for item in successful_items]
                        vectors = [item[1] for item in successful_items]
                        payloads = [item[2] for item in successful_items]
                        await index.upsert_points(ids=ids, vectors=vectors, payloads=payloads)
                    else:
                        logger.warning("No vectors produced for export: path=%s", rel)

                    stats.threads_indexed = len(successful_items)
                    summary.threads_indexed += len(successful_items)
                    summary.exports.append(stats)
                    logger.info(
                        "Indexing export completed: path=%s indexed_threads=%s total_threads=%s",
                        rel,
                        stats.threads_indexed,
                        stats.threads_total,
                    )
                    index_state[rel] = {
                        "status": "indexed_with_errors" if embed_failures else "indexed",
                        "source_namespace": source_namespace,
                        "relative_result_path": rel,
                        "file_size": file_size,
                        "file_mtime_ns": file_mtime_ns,
                        "last_indexed_at": datetime.now(UTC).isoformat(),
                        "threads_total": stats.threads_total,
                        "threads_indexed": stats.threads_indexed,
                        "messages_loaded": stats.messages_loaded,
                        "last_error": _format_embed_failures(rel, embed_failures) if embed_failures else None,
                    }
                    _save_index_state(settings.data_dir, index_state)
            except Exception as exc:  # noqa: BLE001 - aggregate errors per export
                summary.errors.append(f"{rel}: {exc}")
                logger.exception("Indexing export failed: path=%s", rel)
                index_state[rel] = {
                    "status": "failed",
                    "source_namespace": source_namespace,
                    "relative_result_path": rel,
                    "file_size": file_size,
                    "file_mtime_ns": file_mtime_ns,
                    "last_indexed_at": datetime.now(UTC).isoformat(),
                    "threads_total": 0,
                    "threads_indexed": 0,
                    "messages_loaded": 0,
                    "last_error": str(exc),
                }
                _save_index_state(settings.data_dir, index_state)
            await emit_progress(
                exports_done=summary.exports_processed,
                source_namespace=source_namespace,
                phase="processing",
                current_export_relative_path=rel,
            )
    finally:
        await embedder.aclose()
        await index.aclose()

    logger.info(
        "Index run completed: exports_processed=%s threads_indexed=%s errors=%s",
        summary.exports_processed,
        summary.threads_indexed,
        len(summary.errors),
    )
    return summary
