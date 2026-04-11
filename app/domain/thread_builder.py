from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterable

from app.domain.models import DiscussionThread, NormalizedMessage


def _root_for_message(
    msg_id: int,
    by_id: dict[int, NormalizedMessage],
    cache: dict[int, int],
) -> int:
    if msg_id in cache:
        return cache[msg_id]
    visited: set[int] = set()
    current = msg_id
    while True:
        if current in visited:
            cache[msg_id] = current
            return current
        visited.add(current)
        m = by_id.get(current)
        if m is None or m.reply_to_message_id is None:
            cache[msg_id] = current
            return current
        parent = m.reply_to_message_id
        if parent not in by_id:
            cache[msg_id] = current
            return current
        current = parent


def build_threads(messages: Iterable[NormalizedMessage]) -> list[DiscussionThread]:
    """
    Group messages into reply-trees: walk reply_to until parent is missing or not a user message.

    Messages not in `by_id` (e.g. service events) terminate the walk; the last user message
    in the chain becomes the root for that subtree.
    """
    by_id: dict[int, NormalizedMessage] = {m.id: m for m in messages}
    cache: dict[int, int] = {}
    roots: dict[int, list[NormalizedMessage]] = defaultdict(list)

    for m in messages:
        r = _root_for_message(m.id, by_id, cache)
        roots[r].append(m)

    threads: list[DiscussionThread] = []
    for root_id, msgs in roots.items():
        ordered = sorted(msgs, key=lambda x: (x.date, x.id))
        message_ids = tuple(x.id for x in ordered)
        authors_set: set[str] = set()
        for x in ordered:
            if x.from_display and x.from_display.strip():
                authors_set.add(x.from_display.strip())
        authors = tuple(sorted(authors_set))
        date_from = min(x.date for x in ordered)
        date_to = max(x.date for x in ordered)
        lines: list[str] = []
        for x in ordered:
            who = x.from_display or "unknown"
            lines.append(f"[{x.id}] {who}: {x.text}")
        merged = "\n".join(lines).strip()
        threads.append(
            DiscussionThread(
                root_message_id=root_id,
                message_ids=message_ids,
                text=merged,
                date_from=date_from,
                date_to=date_to,
                authors=authors,
                message_count=len(ordered),
            )
        )
    return threads
