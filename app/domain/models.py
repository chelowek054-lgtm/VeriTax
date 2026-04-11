from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class TelegramExportMeta:
    """Top-level fields from Telegram `result.json`."""

    chat_name: str | None
    chat_id: int | None
    export_type: str | None


@dataclass(frozen=True, slots=True)
class NormalizedMessage:
    """User message normalized for threading."""

    id: int
    date: datetime
    from_display: str | None
    from_id: str | None
    text: str
    reply_to_message_id: int | None
    raw: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class DiscussionThread:
    """One reply-tree (ветка) to index as a single vector."""

    root_message_id: int
    message_ids: tuple[int, ...]
    text: str
    date_from: datetime
    date_to: datetime
    authors: tuple[str, ...]
    message_count: int

    @property
    def thread_id(self) -> int:
        return self.root_message_id
