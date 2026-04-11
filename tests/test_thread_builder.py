from datetime import datetime

from app.domain.models import NormalizedMessage
from app.domain.thread_builder import build_threads


def _msg(
    mid: int,
    reply_to: int | None,
    text: str,
    who: str = "A",
) -> NormalizedMessage:
    return NormalizedMessage(
        id=mid,
        date=datetime(2026, 1, 1, 12, 0, 0),
        from_display=who,
        from_id="user1",
        text=text,
        reply_to_message_id=reply_to,
        raw={},
    )


def test_thread_replies_group_under_root() -> None:
    messages = [
        _msg(120, 22, "Question?", "U1"),
        _msg(129, 120, "Answer one", "U2"),
        _msg(130, 120, "Answer two", "U3"),
    ]
    threads = build_threads(messages)
    assert len(threads) == 1
    t = threads[0]
    assert t.root_message_id == 120
    assert t.message_ids == (120, 129, 130)
    assert "120" in t.text and "129" in t.text


def test_separate_roots_when_no_reply_chain_between() -> None:
    messages = [
        _msg(10, None, "root a"),
        _msg(11, None, "root b"),
    ]
    threads = build_threads(messages)
    assert len(threads) == 2
    roots = {t.root_message_id for t in threads}
    assert roots == {10, 11}
