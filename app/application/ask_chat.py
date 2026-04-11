from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.adapters.ollama_chat import OllamaChatService
from app.adapters.ollama_embeddings import OllamaEmbeddingService
from app.adapters.qdrant_index import QdrantVectorIndex
from app.core.config import Settings

RAG_SYSTEM_PROMPT_RU = (
    "Ты помощник по архиву чатов Telegram. Отвечай только на основе приведённых ниже фрагментов обсуждений. "
    "Если данных недостаточно, прямо напиши об этом по-русски. Не выдумывай факты и не цитируй то, чего нет во фрагментах."
)


class _Embedder(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def aclose(self) -> None: ...


class _IndexSearch(Protocol):
    async def search_similar(
        self,
        *,
        query_vector: list[float],
        top_k: int,
        source_namespace: str | None = None,
    ) -> list[tuple[float, dict[str, Any]]]: ...

    async def aclose(self) -> None: ...


class _Chat(Protocol):
    async def chat(self, system_prompt: str, user_message: str) -> str: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AskSource:
    thread_id: int
    root_message_id: int
    message_ids: list[int]
    source_namespace: str
    source_chat: str
    date_from: str
    date_to: str
    authors: list[str]
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class AskResult:
    answer: str
    sources: list[AskSource]


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str = "") -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return default
    return str(value)


def _as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for x in value:
        out.append(_as_int(x))
    return out


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_as_str(x) for x in value]


def payload_to_ask_source(score: float, payload: dict[str, Any]) -> AskSource:
    return AskSource(
        thread_id=_as_int(payload.get("thread_id")),
        root_message_id=_as_int(payload.get("root_message_id")),
        message_ids=_as_int_list(payload.get("message_ids")),
        source_namespace=_as_str(payload.get("source_namespace")),
        source_chat=_as_str(payload.get("source_chat")),
        date_from=_as_str(payload.get("date_from")),
        date_to=_as_str(payload.get("date_to")),
        authors=_as_str_list(payload.get("authors")),
        text=_as_str(payload.get("text")),
        score=float(score),
    )


def build_context_blocks(sources: list[AskSource], max_chars: int) -> tuple[list[AskSource], list[str]]:
    """Pick sources in order until assembled context fits max_chars."""
    used: list[AskSource] = []
    blocks: list[str] = []
    budget = max_chars
    for src in sources:
        header = (
            f"Чат: {src.source_chat or '(без названия)'} | namespace: {src.source_namespace or '—'} | "
            f"период: {src.date_from} — {src.date_to} | авторы: {', '.join(src.authors) or '—'}"
        )
        body = src.text.strip()
        block = f"{header}\n{body}"
        overhead = 80
        if len(block) + overhead > budget and blocks:
            break
        if len(block) > budget and not blocks:
            block = block[: max(500, budget - 40)] + "\n...[обрезано]"
        used.append(src)
        blocks.append(block)
        budget -= len(block) + overhead
        if budget <= 0:
            break
    return used, blocks


def build_user_rag_message(question: str, context_blocks: list[str]) -> str:
    parts: list[str] = ["Ниже фрагменты обсуждений из архива:\n"]
    for i, block in enumerate(context_blocks, 1):
        parts.append(f"--- Фрагмент [{i}] ---\n{block}\n")
    parts.append(f"\nВопрос пользователя:\n{question.strip()}\n")
    parts.append("\nСформулируй ответ по-русски, опираясь только на фрагменты выше.")
    return "\n".join(parts)


async def ask_chat(
    settings: Settings,
    *,
    question: str,
    source_namespace: str | None,
    embedder: _Embedder,
    index: _IndexSearch,
    chat: _Chat,
) -> AskResult:
    q = question.strip()
    if not q:
        return AskResult(answer="Введите непустой вопрос.", sources=[])

    vectors = await embedder.embed_texts([q])
    if not vectors:
        return AskResult(
            answer="Не удалось получить векторное представление вопроса (пустой ответ эмбеддинга).",
            sources=[],
        )
    query_vector = vectors[0]

    ns = source_namespace.strip() if source_namespace and source_namespace.strip() else None
    hits = await index.search_similar(
        query_vector=query_vector,
        top_k=settings.rag_top_k,
        source_namespace=ns,
    )
    if not hits:
        return AskResult(
            answer="В индексе не найдено релевантных фрагментов для этого вопроса. "
            "Проверьте, что индексация выполнена и при необходимости уточните фильтр по папке чата.",
            sources=[],
        )

    raw_sources = [payload_to_ask_source(score, pl) for score, pl in hits]
    used_sources, blocks = build_context_blocks(raw_sources, settings.ask_context_max_chars)
    if not used_sources:
        return AskResult(
            answer="Найденные фрагменты слишком объёмные для текущего лимита контекста. "
            "Уменьшите размер веток или увеличьте ASK_CONTEXT_MAX_CHARS.",
            sources=raw_sources[: settings.rag_top_k],
        )

    user_message = build_user_rag_message(q, blocks)
    answer = await chat.chat(RAG_SYSTEM_PROMPT_RU, user_message)
    return AskResult(answer=answer.strip() or "(пустой ответ модели)", sources=used_sources)


async def ask_chat_default_clients(
    settings: Settings,
    *,
    question: str,
    source_namespace: str | None = None,
) -> AskResult:
    """Convenience: own Ollama embed + Qdrant + Ollama chat clients (closed after call)."""
    embedder = OllamaEmbeddingService(settings)
    index = QdrantVectorIndex(settings)
    chat = OllamaChatService(settings)
    try:
        return await ask_chat(
            settings,
            question=question,
            source_namespace=source_namespace,
            embedder=embedder,
            index=index,
            chat=chat,
        )
    finally:
        await embedder.aclose()
        await index.aclose()
        await chat.aclose()
