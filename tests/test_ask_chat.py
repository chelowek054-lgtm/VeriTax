from __future__ import annotations

from app.application.ask_chat import (
    AskSource,
    build_context_blocks,
    build_user_rag_message,
    payload_to_ask_source,
)


def test_payload_to_ask_source_coerces_types() -> None:
    src = payload_to_ask_source(
        0.91,
        {
            "thread_id": 12,
            "root_message_id": 12,
            "message_ids": [1, 2],
            "source_namespace": "Chat",
            "source_chat": "My chat",
            "date_from": "2026-01-01",
            "date_to": "2026-01-02",
            "authors": ["a", "b"],
            "text": "hello",
        },
    )
    assert src.score == 0.91
    assert src.thread_id == 12
    assert src.message_ids == [1, 2]


def test_build_context_blocks_respects_budget() -> None:
    sources = [
        AskSource(
            thread_id=1,
            root_message_id=1,
            message_ids=[1],
            source_namespace="ns",
            source_chat="c",
            date_from="a",
            date_to="b",
            authors=[],
            text="x" * 500,
            score=1.0,
        ),
        AskSource(
            thread_id=2,
            root_message_id=2,
            message_ids=[2],
            source_namespace="ns",
            source_chat="c",
            date_from="a",
            date_to="b",
            authors=[],
            text="y" * 5000,
            score=0.5,
        ),
    ]
    used, blocks = build_context_blocks(sources, max_chars=800)
    assert len(used) >= 1
    assert len(blocks) == len(used)
    assert sum(len(b) for b in blocks) <= 900


def test_build_user_rag_message_contains_question() -> None:
    msg = build_user_rag_message("Кто это?", ["frag1", "frag2"])
    assert "Кто это?" in msg
    assert "[1]" in msg or "Фрагмент [1]" in msg
