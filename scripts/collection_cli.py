"""Inspect and run T04 collection import, confirmation, and native indexing locally."""

import argparse
import asyncio
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import sys

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.collections.indexing import build_collection_index
from backend.app.collections.service import (
    SelectedSource,
    confirm_collection_revision,
    create_collection,
    get_collection,
    get_collection_revision,
    set_collection_selection,
)
from backend.app.db import open_database
from backend.app.ingestion.batch import get_import_report, import_collection
from check_models import load_config
from model_config import build_settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path,
        default=PROJECT_ROOT / "data" / "runtime" / "papertrail.sqlite",
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("create", help="Create an empty selection draft.")
    choose = commands.add_parser("select", help="Replace the current draft selection.")
    choose.add_argument("--collection-id", required=True)
    choose.add_argument("--expected-revision", required=True, type=int)
    choose.add_argument("--item", action="append", nargs=2, metavar=("SEARCH_RUN", "VERSIONED_ID"), required=True)

    importing = commands.add_parser("import", help="Import each selected fixed PDF.")
    importing.add_argument("--collection-id", required=True)
    importing.add_argument("--selection-revision", required=True, type=int)
    importing.add_argument("--live", action="store_true", help="Allow PDF requests and database writes.")

    confirming = commands.add_parser("confirm", help="Freeze explicitly chosen successful documents.")
    confirming.add_argument("--collection-id", required=True)
    confirming.add_argument("--selection-revision", required=True, type=int)
    confirming.add_argument("--import-run-id", required=True)
    confirming.add_argument("--document-id", action="append", required=True)

    indexing = commands.add_parser("index", help="Build the native PaperQA index for a frozen revision.")
    indexing.add_argument("--revision-id", required=True)
    indexing.add_argument("--live", action="store_true", help="Allow embedding API calls and database writes.")

    showing = commands.add_parser("show", help="Read a saved collection, import report, or revision.")
    selected = showing.add_mutually_exclusive_group(required=True)
    selected.add_argument("--collection-id")
    selected.add_argument("--import-run-id")
    selected.add_argument("--revision-id")
    return parser.parse_args(argv)


def _print_record(record) -> None:
    if record is None:
        raise LookupError("Requested record does not exist.")
    print(json.dumps(asdict(record), ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None, *, client: httpx.Client | None = None) -> int:
    args = parse_args(argv)
    secrets = []
    try:
        preview = args.command in ("import", "index") and not args.live
        read_only = args.command == "show" or preview
        if read_only and not args.database.is_file():
            raise FileNotFoundError("Database does not exist.")
        engine = open_database(args.database, create_schema=not read_only)
        try:
            if args.command == "create":
                _print_record(create_collection(engine))
            elif args.command == "select":
                _print_record(set_collection_selection(
                    engine,
                    args.collection_id,
                    expected_revision=args.expected_revision,
                    data_root=args.data_root,
                    selections=tuple(SelectedSource(Path(run), version) for run, version in args.item),
                ))
            elif args.command == "import":
                collection = get_collection(engine, args.collection_id)
                if collection is None or collection.selection_revision != args.selection_revision:
                    raise ValueError("Collection selection revision changed.")
                if not args.live:
                    print(f"[PREVIEW] {len(collection.selections)} fixed versions; no PDF request or import record created.")
                    _print_record(collection)
                else:
                    manager = nullcontext(client) if client is not None else httpx.Client()
                    with manager as resolved_client:
                        _print_record(asyncio.run(import_collection(
                            engine, args.collection_id,
                            expected_selection_revision=args.selection_revision,
                            data_root=args.data_root,
                            client=resolved_client,
                        )))
            elif args.command == "confirm":
                _print_record(confirm_collection_revision(
                    engine, args.collection_id,
                    expected_selection_revision=args.selection_revision,
                    import_run_id=args.import_run_id,
                    document_ids=tuple(args.document_id),
                    data_root=args.data_root,
                ))
            elif args.command == "index":
                revision = get_collection_revision(engine, args.revision_id)
                if revision is None:
                    raise LookupError("Collection revision does not exist.")
                if not args.live:
                    print(f"[PREVIEW] {len(revision.members)} frozen documents; no embedding call or index created.")
                    _print_record(revision)
                else:
                    config = load_config("all")
                    secrets = [config.get("DEEPSEEK_API_KEY"), config.get("EMBEDDING_API_KEY")]
                    settings = build_settings(config)
                    _print_record(asyncio.run(build_collection_index(
                        engine, revision.id,
                        data_root=args.data_root,
                        settings=settings,
                        embedding_model=settings.get_embedding_model(),
                    )))
            elif args.command == "show":
                if args.collection_id:
                    _print_record(get_collection(engine, args.collection_id))
                elif args.import_run_id:
                    _print_record(get_import_report(engine, args.import_run_id))
                else:
                    _print_record(get_collection_revision(engine, args.revision_id))
        finally:
            engine.dispose()
        return 0
    except Exception as error:
        message = str(error)
        for secret in secrets:
            if secret:
                message = message.replace(secret, "[REDACTED]")
        print(f"[FAIL] {type(error).__name__}: {message[:800]}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
