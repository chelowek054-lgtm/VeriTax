from app.domain.telegram_text import normalize_telegram_text


def test_normalize_plain_string() -> None:
    assert normalize_telegram_text("  hello  ") == "hello"


def test_normalize_list_fragments() -> None:
    raw = [
        "See ",
        {"type": "link", "text": "https://example.com"},
        {"type": "plain", "text": " end"},
    ]
    assert normalize_telegram_text(raw) == "See https://example.com end"


def test_normalize_none() -> None:
    assert normalize_telegram_text(None) == ""
