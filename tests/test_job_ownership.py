import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, inspect, update
from sqlalchemy.orm import Session

from backend.app.db import open_database
from backend.app.jobs.service import get_attempt, get_job, get_search, retry_search, submit_search
from backend.app.jobs.worker import claim_next_search, interrupt_expired_attempt
from backend.app.jobs.ownership import LostClaim, finish_active_attempt
from backend.app.models import Base, JobAttemptResult, SearchRun


class JobOwnershipTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "papertrail.sqlite"
        self.engine = open_database(self.database_path)
        self.addCleanup(self.engine.dispose)
        self.submitted = submit_search(
            self.engine,
            question="How do agents remember?",
            query_text="LLM agent memory",
            idempotency_key="ownership-test",
        )
        self.claimed = claim_next_search(self.engine, worker_id="worker-a")
        self.artifact_ref = f"runs/search-{self.claimed.attempt_id}/run.json"
        self.finished_at = "2026-09-24T00:00:00Z"

    def _finish(self, **overrides):
        arguments = {
            "job_id": self.claimed.job_id,
            "attempt_id": self.claimed.attempt_id,
            "worker_id": "worker-a",
            "status": "succeeded",
            "finished_at": self.finished_at,
            "artifact_ref": self.artifact_ref,
            "result_resource_id": self.claimed.search_id,
        }
        arguments.update(overrides)
        return finish_active_attempt(self.engine, **arguments)

    def test_success_commits_job_attempt_domain_row_and_result_binding(self):
        def publish(session):
            session.execute(
                update(SearchRun)
                .where(SearchRun.id == self.claimed.search_id)
                .values(snapshot_ref=self.artifact_ref)
            )

        self._finish(publish=publish)

        self.assertEqual(get_job(self.engine, self.claimed.job_id).status, "succeeded")
        self.assertEqual(get_attempt(self.engine, self.claimed.attempt_id).status, "succeeded")
        self.assertEqual(get_search(self.engine, self.claimed.search_id).snapshot_ref, self.artifact_ref)
        with Session(self.engine) as session:
            bound = session.get(JobAttemptResult, self.claimed.attempt_id)
            self.assertEqual(bound.resource_id, self.claimed.search_id)
            self.assertEqual(bound.artifact_ref, self.artifact_ref)

    def test_interrupted_old_attempt_cannot_publish_after_retry(self):
        lease_until = datetime.fromisoformat(
            get_attempt(self.engine, self.claimed.attempt_id).lease_until.replace("Z", "+00:00")
        )
        self.assertTrue(
            interrupt_expired_attempt(
                self.engine,
                self.claimed.attempt_id,
                worker_stopped_confirmed=True,
                now=lease_until + timedelta(seconds=1),
            )
        )
        replacement = retry_search(
            self.engine,
            self.claimed.job_id,
            expected_attempt_id=self.claimed.attempt_id,
        )
        published = []

        with self.assertRaises(LostClaim):
            self._finish(publish=lambda _session: published.append(True))

        self.assertEqual(published, [])
        self.assertEqual(get_job(self.engine, self.claimed.job_id).status, "queued")
        self.assertEqual(get_job(self.engine, self.claimed.job_id).active_attempt_id, replacement.attempt_id)
        self.assertIsNone(get_search(self.engine, self.claimed.search_id).snapshot_ref)
        with Session(self.engine) as session:
            self.assertIsNone(session.get(JobAttemptResult, self.claimed.attempt_id))

    def test_wrong_worker_cannot_publish(self):
        published = []
        with self.assertRaises(LostClaim):
            self._finish(
                worker_id="worker-b", publish=lambda _session: published.append(True)
            )

        self.assertEqual(published, [])
        self.assertEqual(get_job(self.engine, self.claimed.job_id).status, "running")
        self.assertEqual(get_attempt(self.engine, self.claimed.attempt_id).status, "running")
        with Session(self.engine) as session:
            self.assertIsNone(session.get(JobAttemptResult, self.claimed.attempt_id))

    def test_publisher_failure_rolls_back_all_state(self):
        def failing_publish(session):
            session.execute(
                update(SearchRun)
                .where(SearchRun.id == self.claimed.search_id)
                .values(snapshot_ref=self.artifact_ref)
            )
            raise RuntimeError("publication failed")

        with self.assertRaisesRegex(RuntimeError, "publication failed"):
            self._finish(publish=failing_publish)

        self.assertEqual(get_job(self.engine, self.claimed.job_id).status, "running")
        self.assertEqual(get_attempt(self.engine, self.claimed.attempt_id).status, "running")
        self.assertIsNone(get_search(self.engine, self.claimed.search_id).snapshot_ref)
        with Session(self.engine) as session:
            self.assertIsNone(session.get(JobAttemptResult, self.claimed.attempt_id))

    def test_failed_attempt_keeps_diagnostic_artifact_without_success_binding(self):
        self._finish(status="failed", result_resource_id=None, error_kind="http_error")

        self.assertEqual(get_job(self.engine, self.claimed.job_id).status, "failed")
        attempt = get_attempt(self.engine, self.claimed.attempt_id)
        self.assertEqual(attempt.status, "failed")
        self.assertEqual(attempt.artifact_ref, self.artifact_ref)
        with Session(self.engine) as session:
            self.assertIsNone(session.get(JobAttemptResult, self.claimed.attempt_id))

    def test_escape_artifact_reference_is_rejected_without_finishing(self):
        with self.assertRaises(ValueError):
            self._finish(artifact_ref="../outside/run.json")
        self.assertEqual(get_job(self.engine, self.claimed.job_id).status, "running")


class AdditiveSchemaTest(unittest.TestCase):
    def test_old_database_gains_result_table_without_changing_existing_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.sqlite"
            old_engine = create_engine(f"sqlite:///{path.as_posix()}")
            try:
                Base.metadata.create_all(
                    old_engine,
                    tables=[
                        table for table in Base.metadata.sorted_tables
                        if table.name != "job_attempt_results"
                    ],
                )
                previous = submit_search(
                    old_engine,
                    question="What was already saved?",
                    query_text="saved search",
                    idempotency_key="legacy-job",
                )
            finally:
                old_engine.dispose()

            upgraded = open_database(path)
            try:
                self.assertTrue(inspect(upgraded).has_table("jobs"))
                self.assertTrue(inspect(upgraded).has_table("job_attempts"))
                self.assertTrue(inspect(upgraded).has_table("job_attempt_results"))
                self.assertEqual(get_job(upgraded, previous.job_id).status, "queued")
                self.assertEqual(get_search(upgraded, previous.search_id).query_text, "saved search")
            finally:
                upgraded.dispose()


if __name__ == "__main__":
    unittest.main()
