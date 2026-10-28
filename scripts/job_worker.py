"""Run the Job kinds currently registered in PaperTrail's unified worker."""

import argparse
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

import httpx

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.db import open_database
from backend.app.jobs.dispatcher import run_one_job
from backend.app.jobs.process_lock import worker_singleton
from backend.app.research.workflow import recover_orphan_research_attempts
from backend.app.search.client import ArxivClient, HttpxArxivTransport
from backend.app.search.fallback import ARXIV_TRAFFIC_GATE, OpenAlexArxivFallback


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run registered PaperTrail Jobs.")
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "runtime" / "papertrail.sqlite",
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--worker-id", default=f"worker-{os.getpid()}-{uuid4().hex[:8]}")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="Process at most one Job.")
    parser.add_argument(
        "--live", action="store_true", help="Allow registered Jobs to run."
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    search_client: ArxivClient | None = None,
    import_client: httpx.Client | None = None,
) -> int:
    args = parse_args(argv)
    if not args.live:
        print("[PREVIEW] No Job was claimed. Add --live to start the worker.")
        return 0
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive.")

    with worker_singleton(args.database) as acquired:
        if not acquired:
            print("[BUSY] Another worker already owns this database.")
            return 2

        engine = open_database(args.database)
        resolved_client = search_client or ArxivClient(
            transport=HttpxArxivTransport(traffic_gate=ARXIV_TRAFFIC_GATE))
        # Injected clients stay fully offline; the ordinary live worker enables fallback.
        search_fallback = OpenAlexArxivFallback() if search_client is None else None
        resolved_import_client = import_client or httpx.Client()
        try:
            while True:
                try:
                    recover_orphan_research_attempts(engine, worker_id=args.worker_id, singleton_acquired=acquired)
                    result = run_one_job(
                        engine,
                        worker_id=args.worker_id,
                        data_root=args.data_root,
                        search_client=resolved_client,
                        search_fallback=search_fallback,
                        import_client=resolved_import_client,
                    )
                except Exception as error:
                    print(f"[ERROR] {type(error).__name__}: {error}")
                    return 1
                if result is not None:
                    print(
                        f"[{result.status.upper()}] kind={result.kind} "
                        f"job={result.job_id} attempt={result.attempt_id} "
                        f"result={result.result_resource_id} "
                        f"artifact={result.artifact_ref}"
                    )
                    if args.once:
                        return 0 if result.status == "succeeded" else 1
                elif args.once:
                    print("[IDLE] No queued registered Job.")
                    return 0
                else:
                    time.sleep(args.poll_seconds)
        finally:
            engine.dispose()
            if search_client is None:
                resolved_client.transport.close()
                search_fallback.close()
            if import_client is None:
                resolved_import_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
