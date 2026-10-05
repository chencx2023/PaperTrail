import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from backend.app.db import open_database
from backend.app.jobs.claim import claim_next_job
from backend.app.jobs.dispatcher import run_one_job
from backend.app.jobs.ownership import finish_active_attempt
from backend.app.jobs.process_lock import worker_singleton
from backend.app.jobs.service import get_attempt, get_job, list_attempts, retry_search, submit_search
from backend.app.models import Job, JobAttemptResult, SearchRun
from backend.app.search.client import ArxivClient, TransportResponse
from scripts.job_worker import main as job_worker_main


class FixedTransport:
    def __init__(self):
        self.calls = 0

    def send(self, _request):
        self.calls += 1
        return TransportResponse(
            status_code=200,
            body=(
                Path(__file__).parent
                / "fixtures"
                / "arxiv"
                / "normal_with_versions.xml"
            ).read_bytes(),
            headers=(("Content-Type", "application/atom+xml"),),
        )


class JobDispatchTest(unittest.TestCase):
    def test_claim_skips_unregistered_kind_and_reuses_retry_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = open_database(Path(directory) / "papertrail.sqlite")
            try:
                future_job_id = uuid4().hex
                with Session(engine) as session:
                    session.add(
                        Job(
                            id=future_job_id,
                            kind="import",
                            resource_id=uuid4().hex,
                            status="queued",
                            idempotency_key="future-import",
                            request_hash="0" * 64,
                            created_at="2026-01-01T00:00:00Z",
                            updated_at="2026-01-01T00:00:00Z",
                        )
                    )
                    session.commit()
                submitted = submit_search(
                    engine,
                    question="How do agents remember?",
                    query_text="LLM agent memory",
                    idempotency_key="unified-claim",
                )

                first = claim_next_job(
                    engine, worker_id="worker-a", supported_kinds=("search",)
                )
                self.assertEqual(first.job_id, submitted.job_id)
                self.assertEqual(first.kind, "search")
                self.assertEqual(first.resource_id, submitted.search_id)
                self.assertIsNone(
                    claim_next_job(
                        engine, worker_id="worker-b", supported_kinds=("search",)
                    )
                )
                self.assertEqual(get_job(engine, future_job_id).status, "queued")

                finish_active_attempt(
                    engine,
                    job_id=first.job_id,
                    attempt_id=first.attempt_id,
                    worker_id="worker-a",
                    status="failed",
                    finished_at="2026-09-24T00:00:00Z",
                )
                reserved = retry_search(
                    engine, first.job_id, expected_attempt_id=first.attempt_id
                )
                second = claim_next_job(
                    engine, worker_id="worker-b", supported_kinds=("search",)
                )
                self.assertEqual(second.attempt_id, reserved.attempt_id)
                self.assertEqual(get_attempt(engine, second.attempt_id).attempt_no, 2)
                self.assertEqual(get_job(engine, future_job_id).status, "queued")
            finally:
                engine.dispose()

    def test_unified_cli_executes_one_search_and_reopens_its_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "papertrail.sqlite"
            engine = open_database(database)
            submitted = submit_search(
                engine,
                question="How do agents remember?",
                query_text="LLM agent memory",
                idempotency_key="unified-run",
            )
            engine.dispose()
            transport = FixedTransport()
            client = ArxivClient(
                transport=transport,
                clock=lambda: datetime(2026, 9, 24, tzinfo=timezone.utc),
                monotonic=lambda: 0.0,
                sleeper=lambda _seconds: None,
            )

            output = StringIO()
            with redirect_stdout(output):
                exit_code = job_worker_main(
                    [
                        "--database", str(database),
                        "--data-root", str(root / "data"),
                        "--worker-id", "worker-unified",
                        "--live", "--once",
                    ],
                    search_client=client,
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(transport.calls, 1)
            self.assertIn("[SUCCEEDED]", output.getvalue())
            reopened = open_database(database)
            try:
                job = get_job(reopened, submitted.job_id)
                self.assertEqual(job.status, "succeeded")
                attempt = get_attempt(reopened, job.active_attempt_id)
                self.assertEqual(attempt.worker_id, "worker-unified")
                self.assertEqual(len(list_attempts(reopened, job.id)), 1)
                with Session(reopened) as session:
                    binding = session.get(JobAttemptResult, attempt.id)
                    self.assertEqual(binding.resource_id, submitted.search_id)
                    self.assertEqual(binding.artifact_ref, attempt.artifact_ref)
            finally:
                reopened.dispose()

    def test_preview_and_existing_worker_lock_do_not_create_or_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "papertrail.sqlite"
            with redirect_stdout(StringIO()):
                self.assertEqual(
                    job_worker_main(["--database", str(database), "--once"]), 0
                )
            self.assertFalse(database.exists())
            with worker_singleton(database) as acquired:
                self.assertTrue(acquired)
                output = StringIO()
                with redirect_stdout(output):
                    result = job_worker_main(
                        ["--database", str(database), "--live", "--once"]
                    )
                self.assertEqual(result, 2)
                self.assertIn("[BUSY]", output.getvalue())
            self.assertFalse(database.exists())

    def test_missing_search_resource_fails_attempt_before_upstream_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = open_database(root / "papertrail.sqlite")
            try:
                submitted = submit_search(
                    engine,
                    question="How do agents remember?",
                    query_text="LLM agent memory",
                    idempotency_key="missing-resource",
                )
                with Session(engine) as session:
                    session.delete(session.get(SearchRun, submitted.search_id))
                    session.commit()

                class NoCallClient:
                    def fetch(self, _plan):
                        raise AssertionError("The upstream must not be called.")

                with self.assertRaisesRegex(RuntimeError, "no search resource"):
                    run_one_job(
                        engine,
                        worker_id="worker-unified",
                        data_root=root / "data",
                        search_client=NoCallClient(),
                    )
                job = get_job(engine, submitted.job_id)
                self.assertEqual(job.status, "failed")
                attempt = get_attempt(engine, job.active_attempt_id)
                self.assertEqual(attempt.error_kind, "missing_search_resource")
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
