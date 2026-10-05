"""Run queued arXiv searches in a process separate from the submitter."""

import argparse
import os
from pathlib import Path
import sys
import time
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.db import open_database
from backend.app.jobs.process_lock import worker_singleton
from backend.app.jobs.worker import run_one_search
from backend.app.search.client import ArxivClient


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PaperTrail's queued searches.")
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "runtime" / "papertrail.sqlite",
    )
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "data"
    )
    parser.add_argument(
        "--worker-id", default=f"worker-{os.getpid()}-{uuid4().hex[:8]}"
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="Process at most one job.")
    parser.add_argument(
        "--live", action="store_true", help="Allow cache reads or arXiv requests."
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None, *, client: ArxivClient | None = None
) -> int:
    args = parse_args(argv)
    if not args.live:
        print("[PREVIEW] No job was claimed. Add --live to start the worker.")
        return 0
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive.")

    with worker_singleton(args.database) as acquired:
        if not acquired:
            print("[BUSY] Another search worker already owns this database.")
            return 2

        engine = open_database(args.database)
        resolved_client = client or ArxivClient()
        try:
            while True:
                try:
                    result = run_one_search(
                        engine,
                        worker_id=args.worker_id,
                        data_root=args.data_root,
                        client=resolved_client,
                    )
                except Exception as error:
                    print(f"[ERROR] {type(error).__name__}: {error}")
                    return 1
                if result is not None:
                    print(
                        f"[{result.status.upper()}] job={result.job_id} "
                        f"attempt={result.attempt_id} artifact={result.artifact_ref}"
                    )
                    if args.once:
                        return 0 if result.status == "succeeded" else 1
                elif args.once:
                    print("[IDLE] No queued search job.")
                    return 0
                else:
                    time.sleep(args.poll_seconds)
        finally:
            engine.dispose()
            if client is None:
                resolved_client.transport.close()


if __name__ == "__main__":
    raise SystemExit(main())
