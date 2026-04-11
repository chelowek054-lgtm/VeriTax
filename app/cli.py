from __future__ import annotations

import argparse
import asyncio
import json
import sys

from app.application.index_exports import run_index_all_exports
from app.core.config import Settings, get_settings


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="veritax", description="VeriTax CLI")
    sub = p.add_subparsers(dest="command", required=True)

    idx = sub.add_parser("index", help="Index all Telegram exports under DATA_DIR into Qdrant")
    idx.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override DATA_DIR (default from env or repo .data)",
    )
    idx.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary to stdout",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command != "index":
        return 2

    settings: Settings
    if args.data_dir:
        from dataclasses import replace
        from pathlib import Path

        base = get_settings()
        settings = replace(base, data_dir=Path(args.data_dir).resolve())
    else:
        settings = get_settings()

    summary = asyncio.run(run_index_all_exports(settings))
    if args.json:
        payload = {
            "exports_processed": summary.exports_processed,
            "threads_indexed": summary.threads_indexed,
            "errors": summary.errors,
            "exports": [
                {
                    "path": e.path,
                    "source_namespace": e.source_namespace,
                    "threads_total": e.threads_total,
                    "threads_indexed": e.threads_indexed,
                    "messages_loaded": e.messages_loaded,
                }
                for e in summary.exports
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Exports processed: {summary.exports_processed}")
        print(f"Threads indexed: {summary.threads_indexed}")
        if summary.errors:
            print("Errors:")
            for err in summary.errors:
                print(f"  - {err}")
        for e in summary.exports:
            print(
                f"  {e.source_namespace}: threads {e.threads_indexed}/{e.threads_total}, "
                f"messages {e.messages_loaded}"
            )

    return 0 if not summary.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
