"""Inspect, interrupt, or explicitly retry a registered PaperTrail Job."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

from sqlalchemy import inspect


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.app.collections.revision_submission import (
    IndexRetryNotAllowed, InvalidIndexRequest, retry_index_job,
)
from backend.app.db import open_database
from backend.app.jobs.import_submission import (
    ImportRetryNotAllowed, InvalidJobRequest, retry_import_job,
)
from backend.app.jobs.service import (
    RetryNotAllowed, get_attempt, get_job, list_attempts, retry_search,
)
from backend.app.jobs.worker import interrupt_expired_attempt
from backend.app.qa.submission import (
    AnswerRetryNotAllowed, InvalidAnswerRequest, retry_answer,
)


RECOVERABLE_KINDS = ("search", "import", "index", "answer")
RETRY_ERRORS = (
    RetryNotAllowed, ImportRetryNotAllowed, IndexRetryNotAllowed,
    AnswerRetryNotAllowed, InvalidJobRequest, InvalidIndexRequest,
    InvalidAnswerRequest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a Job, interrupt an expired worker, or retry explicitly."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "runtime" / "papertrail.sqlite",
    )
    parser.add_argument("--job-id", required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--confirm-worker-stopped",
        action="store_true",
        help="Assert the old worker stopped, then interrupt if its lease expired.",
    )
    action.add_argument(
        "--retry",
        action="store_true",
        help="Explicitly requeue a failed or interrupted Job; external work may repeat.",
    )
    return parser.parse_args(argv)


def _retry(engine, kind: str, job_id: str, attempt_id: str) -> tuple[str, str]:
    if kind == "search":
        return retry_search(engine, job_id, expected_attempt_id=attempt_id).attempt_id, ""
    if kind == "import":
        return retry_import_job(engine, job_id, expected_attempt_id=attempt_id).attempt_id, ""
    if kind == "index":
        return retry_index_job(engine, job_id, expected_attempt_id=attempt_id), ""
    if kind == "answer":
        retried = retry_answer(engine, job_id, expected_attempt_id=attempt_id)
        identities = (
            f" answer_run_id={retried.answer_run_id}"
            f" previous_answer_run_id={retried.previous_answer_run_id}"
        )
        return retried.attempt_id, identities
    raise ValueError(f"Unsupported Job kind {kind!r}.")


def main(
    argv: list[str] | None = None, *,
    allowed_kinds: tuple[str, ...] = RECOVERABLE_KINDS,
) -> int:
    args = parse_args(argv)
    if not args.database.is_file():
        print(f"[NOT FOUND] database={args.database}")
        return 1

    engine = open_database(args.database, create_schema=False)
    try:
        schema = inspect(engine)
        if not schema.has_table("jobs") or not schema.has_table("job_attempts"):
            print(f"[INVALID DATABASE] No PaperTrail job tables in {args.database}")
            return 1
        job = get_job(engine, args.job_id)
        if job is None or job.kind not in allowed_kinds:
            label = "search job" if allowed_kinds == ("search",) else "job"
            print(f"[NOT FOUND] {label}={args.job_id}")
            return 1
        print(f"job={job.id} kind={job.kind} resource={job.resource_id} status={job.status}")
        attempt = (
            get_attempt(engine, job.active_attempt_id)
            if job.active_attempt_id is not None
            else None
        )
        if attempt is not None:
            expired = (
                attempt.lease_until is not None
                and datetime.fromisoformat(attempt.lease_until.replace("Z", "+00:00"))
                <= datetime.now(timezone.utc)
            )
            print(
                f"attempt={attempt.id} status={attempt.status} worker={attempt.worker_id} "
                f"heartbeat={attempt.heartbeat_at} lease_until={attempt.lease_until} "
                f"lease_expired={expired}"
            )
        elif job.status == "queued":
            print(f"reserved_attempt={job.active_attempt_id} status=queued")
        else:
            print(f"[NOT FOUND] attempt={job.active_attempt_id}")
            return 1

        for previous in list_attempts(engine, job.id):
            print(
                f"history attempt={previous.id} no={previous.attempt_no} "
                f"status={previous.status} worker={previous.worker_id} "
                f"started={previous.started_at} finished={previous.finished_at} "
                f"artifact={previous.artifact_ref}"
            )

        if args.retry:
            print("[NOTICE] A retry may repeat external requests or charges.")
            if attempt is None:
                print("[UNCHANGED] No completed attempt can be retried.")
                return 1
            try:
                next_attempt_id, identities = _retry(engine, job.kind, job.id, attempt.id)
            except RETRY_ERRORS as error:
                print(f"[UNCHANGED] {error}")
                return 1
            print(f"[QUEUED] job={job.id} reserved_attempt={next_attempt_id}{identities}")
            return 0
        if not args.confirm_worker_stopped:
            print("[PREVIEW] No state changed. Verify the old worker before confirming.")
            return 0
        if attempt is None:
            print("[UNCHANGED] No active running attempt can be interrupted.")
            return 1
        if interrupt_expired_attempt(
            engine, attempt.id, worker_stopped_confirmed=True
        ):
            print(f"[INTERRUPTED] job={job.id} attempt={attempt.id}")
            return 0
        print("[UNCHANGED] The attempt is not active with an expired lease.")
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
