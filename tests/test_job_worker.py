import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path


class SearchWorkerTest(unittest.TestCase):
    def test_claim_creates_one_attempt_and_leaves_no_second_claim(self):
        from backend.app.db import open_database
        from backend.app.jobs.service import get_attempt, get_job, submit_search
        from backend.app.jobs.worker import claim_next_search

        with tempfile.TemporaryDirectory() as directory:
            engine = open_database(Path(directory) / "papertrail.sqlite")
            try:
                submitted = submit_search(
                    engine,
                    question="Agent 记忆如何评估？",
                    query_text="LLM agent memory",
                    idempotency_key="first-search",
                )

                claimed = claim_next_search(engine, worker_id="worker-a")

                self.assertEqual(claimed.job_id, submitted.job_id)
                self.assertEqual(claimed.search_id, submitted.search_id)
                self.assertEqual(claimed.query_text, "LLM agent memory")
                self.assertIsNone(claim_next_search(engine, worker_id="worker-b"))
                job = get_job(engine, submitted.job_id)
                attempt = get_attempt(engine, claimed.attempt_id)
                self.assertEqual(job.status, "running")
                self.assertEqual(job.active_attempt_id, claimed.attempt_id)
                self.assertEqual(attempt.job_id, submitted.job_id)
                self.assertEqual(attempt.attempt_no, 1)
                self.assertEqual(attempt.status, "running")
                self.assertEqual(attempt.worker_id, "worker-a")
            finally:
                engine.dispose()

    def test_one_search_run_saves_replayable_snapshot_and_terminal_state(self):
        from backend.app.db import open_database
        from backend.app.jobs.service import (
            get_attempt,
            get_job,
            get_search,
            submit_search,
        )
        from backend.app.jobs.worker import run_one_search
        from backend.app.models import JobAttemptResult
        from backend.app.search.client import ArxivClient, TransportResponse
        from backend.app.search.service import replay_search_run, run_search
        from sqlalchemy.orm import Session

        class FixedTransport:
            calls = 0

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

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "papertrail.sqlite"
            data_root = root / "data"
            transport = FixedTransport()
            client = ArxivClient(
                transport=transport,
                clock=lambda: datetime(2026, 9, 23, 8, 0, tzinfo=timezone.utc),
                monotonic=lambda: 0.0,
                sleeper=lambda _seconds: None,
            )
            engine = open_database(database_path)
            reopened = None
            try:
                submitted = submit_search(
                    engine,
                    question="Agent 记忆如何评估？",
                    query_text="LLM agent memory",
                    idempotency_key="one-run",
                )

                executed = run_one_search(
                    engine,
                    worker_id="worker-a",
                    data_root=data_root,
                    client=client,
                )

                self.assertEqual(transport.calls, 1)
                self.assertEqual(executed.job_id, submitted.job_id)
                self.assertEqual(executed.status, "succeeded")
                committed_run_dir = data_root / "runs" / f"search-{executed.attempt_id}"
                self.assertTrue((committed_run_dir / "cache-committed.json").is_file())
                cached = run_search(
                    "LLM agent memory",
                    client=client,
                    output_root=data_root / "runs",
                    run_id="search-after-committed-job",
                    clock=lambda: datetime(2026, 9, 23, 8, 0, tzinfo=timezone.utc)
                    + timedelta(minutes=1),
                )
                self.assertTrue(cached.cache_hit)
                self.assertEqual(transport.calls, 1)
                engine.dispose()
                reopened = open_database(database_path)
                job = get_job(reopened, submitted.job_id)
                search = get_search(reopened, submitted.search_id)
                attempt = get_attempt(reopened, executed.attempt_id)
                self.assertEqual(job.status, "succeeded")
                self.assertEqual(attempt.status, "succeeded")
                self.assertEqual(attempt.artifact_ref, search.snapshot_ref)
                with Session(reopened) as session:
                    binding = session.get(JobAttemptResult, executed.attempt_id)
                    self.assertEqual(binding.resource_id, submitted.search_id)
                    self.assertEqual(binding.artifact_ref, search.snapshot_ref)
                self.assertEqual(
                    search.snapshot_ref,
                    f"runs/search-{executed.attempt_id}/run.json",
                )
                run_dir = (data_root / search.snapshot_ref).parent
                self.assertTrue((run_dir / "response.atom").is_file())
                replay = replay_search_run(run_dir)
                self.assertTrue(replay.candidates_match)
                self.assertGreater(len(replay.feed.candidates), 0)
            finally:
                engine.dispose()
                if reopened is not None:
                    reopened.dispose()

    def test_source_failure_keeps_error_artifact_and_marks_job_failed(self):
        from backend.app.db import open_database
        from backend.app.jobs.service import (
            get_attempt,
            get_job,
            get_search,
            submit_search,
        )
        from backend.app.jobs.worker import run_one_search
        from backend.app.search.client import ArxivClient, TransportResponse

        class HttpErrorTransport:
            def send(self, _request):
                return TransportResponse(
                    status_code=400,
                    body=b"bad query",
                    headers=(("Content-Type", "text/plain"),),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            engine = open_database(root / "papertrail.sqlite")
            try:
                submitted = submit_search(
                    engine,
                    question="Agent 记忆如何评估？",
                    query_text="LLM agent memory",
                    idempotency_key="failed-search",
                )
                client = ArxivClient(
                    transport=HttpErrorTransport(),
                    clock=lambda: datetime(2026, 9, 23, 8, 0, tzinfo=timezone.utc),
                )

                executed = run_one_search(
                    engine,
                    worker_id="worker-a",
                    data_root=data_root,
                    client=client,
                )

                job = get_job(engine, submitted.job_id)
                search = get_search(engine, submitted.search_id)
                attempt = get_attempt(engine, executed.attempt_id)
                self.assertEqual(executed.status, "failed")
                self.assertEqual(job.status, "failed")
                self.assertEqual(attempt.status, "failed")
                self.assertEqual(attempt.error_kind, "http_error")
                self.assertEqual(search.snapshot_ref, attempt.artifact_ref)
                saved = json.loads((data_root / attempt.artifact_ref).read_text())
                self.assertEqual(saved["status"], "failed")
                self.assertEqual(saved["http_status"], 400)
            finally:
                engine.dispose()

    def test_unexpected_search_exception_marks_attempt_failed_without_false_snapshot(self):
        from backend.app.db import open_database
        from backend.app.jobs.service import (
            get_attempt,
            get_job,
            get_search,
            submit_search,
        )
        from backend.app.jobs.worker import run_one_search

        class ExplodingClient:
            def fetch(self, _plan):
                raise RuntimeError("unexpected transport failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = open_database(root / "papertrail.sqlite")
            try:
                submitted = submit_search(
                    engine,
                    question="Agent 记忆如何评估？",
                    query_text="LLM agent memory",
                    idempotency_key="unexpected-failure",
                )

                with self.assertRaisesRegex(RuntimeError, "unexpected transport failure"):
                    run_one_search(
                        engine,
                        worker_id="worker-a",
                        data_root=root / "data",
                        client=ExplodingClient(),
                    )

                job = get_job(engine, submitted.job_id)
                search = get_search(engine, submitted.search_id)
                attempt = get_attempt(engine, job.active_attempt_id)
                self.assertEqual(job.status, "failed")
                self.assertEqual(attempt.status, "failed")
                self.assertEqual(attempt.error_kind, "RuntimeError")
                self.assertIsNone(attempt.artifact_ref)
                self.assertIsNone(search.snapshot_ref)
            finally:
                engine.dispose()

    def test_worker_entrypoint_processes_queued_job_after_submitter_closes(self):
        from backend.app.db import open_database
        from backend.app.jobs.service import get_job, submit_search
        from backend.app.search.client import ArxivClient, TransportResponse
        from scripts.search_worker import main

        class FixedTransport:
            def send(self, _request):
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

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "papertrail.sqlite"
            engine = open_database(database_path)
            submitted = submit_search(
                engine,
                question="Agent 记忆如何评估？",
                query_text="LLM agent memory",
                idempotency_key="from-other-process",
            )
            engine.dispose()
            client = ArxivClient(
                transport=FixedTransport(),
                clock=lambda: datetime(2026, 9, 23, 8, 0, tzinfo=timezone.utc),
            )
            output = StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--database",
                        str(database_path),
                        "--data-root",
                        str(root / "data"),
                        "--once",
                        "--live",
                    ],
                    client=client,
                )

            reopened = open_database(database_path)
            try:
                self.assertEqual(exit_code, 0)
                self.assertIn("[SUCCEEDED]", output.getvalue())
                self.assertEqual(get_job(reopened, submitted.job_id).status, "succeeded")
            finally:
                reopened.dispose()

    def test_changed_query_contract_fails_before_contacting_source(self):
        from sqlalchemy.orm import Session

        from backend.app.db import open_database
        from backend.app.jobs.service import get_attempt, get_job, submit_search
        from backend.app.jobs.worker import run_one_search
        from backend.app.models import SearchRun

        class CountingClient:
            calls = 0

            def fetch(self, _plan):
                self.calls += 1
                raise AssertionError("source must not be contacted")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = open_database(root / "papertrail.sqlite")
            try:
                submitted = submit_search(
                    engine,
                    question="Agent 记忆如何评估？",
                    query_text="LLM agent memory",
                    idempotency_key="old-query-contract",
                )
                with Session(engine) as session:
                    search = session.get(SearchRun, submitted.search_id)
                    search.query_contract_version += 1
                    session.commit()
                client = CountingClient()

                with self.assertRaisesRegex(RuntimeError, "query contract"):
                    run_one_search(
                        engine,
                        worker_id="worker-a",
                        data_root=root / "data",
                        client=client,
                    )

                job = get_job(engine, submitted.job_id)
                attempt = get_attempt(engine, job.active_attempt_id)
                self.assertEqual(client.calls, 0)
                self.assertEqual(job.status, "failed")
                self.assertEqual(attempt.error_kind, "QueryContractMismatch")
                self.assertIsNone(attempt.artifact_ref)
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
