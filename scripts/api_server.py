"""Run the local PaperTrail HTTP API without starting a worker."""

import argparse
import os
from pathlib import Path
import sys

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.api import create_app
from backend.app.db import open_database
from backend.app.public_access import PublicAccess


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PaperTrail's local HTTP API.")
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "runtime" / "papertrail.sqlite",
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--share", action="store_true", help="Require PAPERTRAIL_ACCESS_TOKEN and bound public operations.")
    parser.add_argument(
        "--init-db", action="store_true", help="Explicitly create the database schema."
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def main(argv: list[str] | None = None, *, runner=uvicorn.run) -> int:
    args = parse_args(argv)
    if args.init_db:
        engine = open_database(args.database)
        engine.dispose()
    elif not args.database.is_file():
        print("[MISSING] Database does not exist; add --init-db to create it.")
        return 2
    gate = PublicAccess(os.environ.get("PAPERTRAIL_ACCESS_TOKEN", "")) if args.share else None
    options = {"access_log": False} if args.share else {}
    runner(create_app(args.database, args.data_root, public_access=gate), host="127.0.0.1", port=args.port, **options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
