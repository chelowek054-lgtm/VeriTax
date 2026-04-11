from __future__ import annotations

from typing import Any


def normalize_telegram_text(value: Any) -> str:
    """
    Flatten Telegram export `text` field to a single string.

    `text` may be a str, empty, or a list of str / dict fragments.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts).strip()
    return str(value).strip()
