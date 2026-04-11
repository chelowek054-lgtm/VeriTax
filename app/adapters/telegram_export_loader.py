from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.domain.models import NormalizedMessage, TelegramExportMeta
from app.domain.telegram_text import normalize_telegram_text


def _parse_date(raw: str | None) -> datetime:
    if not raw:
        return datetime.min.replace(tzinfo=None)
    # Telegram export uses "2026-01-21T23:31:06" without timezone
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return datetime.min.replace(tzinfo=None)


class TelegramResultJsonLoader:
    """Load Telegram Desktop export `result.json` into domain messages."""

    def load_file(self, path: Path) -> tuple[TelegramExportMeta, list[NormalizedMessage]]:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"Expected object at root, got {type(data)}")

        meta = TelegramExportMeta(
            chat_name=data.get("name") if isinstance(data.get("name"), str) else None,
            chat_id=data.get("id") if isinstance(data.get("id"), int) else None,
            export_type=data.get("type") if isinstance(data.get("type"), str) else None,
        )
        raw_messages = data.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError("Missing or invalid `messages` array")

        out: list[NormalizedMessage] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            msg_id = item.get("id")
            if not isinstance(msg_id, int):
                continue
            date_s = item.get("date")
            date_s = date_s if isinstance(date_s, str) else None
            reply_to = item.get("reply_to_message_id")
            reply_to_i = int(reply_to) if isinstance(reply_to, int) else None
            from_name = item.get("from")
            from_name = from_name if isinstance(from_name, str) else None
            from_id = item.get("from_id")
            from_id_s = from_id if isinstance(from_id, str) else None
            body = normalize_telegram_text(item.get("text"))
            out.append(
                NormalizedMessage(
                    id=msg_id,
                    date=_parse_date(date_s),
                    from_display=from_name,
                    from_id=from_id_s,
                    text=body,
                    reply_to_message_id=reply_to_i,
                    raw=item,
                )
            )
        return meta, out
