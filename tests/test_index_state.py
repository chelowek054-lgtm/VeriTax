from __future__ import annotations

import json

from app.application.index_exports import list_chat_exports


def test_list_chat_exports_reads_saved_index_state(tmp_path) -> None:
    data_dir = tmp_path / ".data"
    export_dir = data_dir / "chat-a"
    export_dir.mkdir(parents=True)
    result_path = export_dir / "result.json"
    result_path.write_text('{"messages":[]}', encoding="utf-8")

    stat = result_path.stat()
    state_path = data_dir / ".veritax-index-state.json"
    state_path.write_text(
        json.dumps(
            {
                "chat-a/result.json": {
                    "status": "indexed",
                    "file_size": stat.st_size,
                    "file_mtime_ns": stat.st_mtime_ns,
                    "last_indexed_at": "2026-04-11T08:00:00+00:00",
                    "threads_total": 10,
                    "threads_indexed": 9,
                    "last_error": None,
                }
            }
        ),
        encoding="utf-8",
    )

    chats = list_chat_exports(data_dir)

    assert len(chats) == 1
    assert chats[0].indexed_status == "indexed"
    assert chats[0].indexed_status_label == "проиндексирован"
    assert chats[0].threads_total == 10
    assert chats[0].threads_indexed == 9


def test_list_chat_exports_marks_changed_file_as_stale(tmp_path) -> None:
    data_dir = tmp_path / ".data"
    export_dir = data_dir / "chat-b"
    export_dir.mkdir(parents=True)
    result_path = export_dir / "result.json"
    result_path.write_text('{"messages":[]}', encoding="utf-8")

    state_path = data_dir / ".veritax-index-state.json"
    state_path.write_text(
        json.dumps(
            {
                "chat-b/result.json": {
                    "status": "indexed",
                    "file_size": 1,
                    "file_mtime_ns": 1,
                    "last_indexed_at": "2026-04-11T08:00:00+00:00",
                    "threads_total": 1,
                    "threads_indexed": 1,
                    "last_error": None,
                }
            }
        ),
        encoding="utf-8",
    )

    chats = list_chat_exports(data_dir)

    assert len(chats) == 1
    assert chats[0].indexed_status == "stale"
    assert chats[0].indexed_status_label == "изменён после индексации"
