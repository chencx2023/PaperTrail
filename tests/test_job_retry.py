import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

from backend.app.db import open_database
from backend.app.jobs.service import (
    RetryNotAllowed,
    get_attempt,
    get_job,
    get_search,
    list_attempts,
    retry_search,
    submit_search,
)
from backend.app.jobs.worker import (
    LostClaim,
    claim_next_search,
    interrupt_expired_attempt,
    run_one_search,
)
from backend.app.search.client import ArxivClient, TransportResponse


class ExplodingClient:
    def fetch(self, _plan):
        raise RuntimeError("offline source failure")


class FixedTransport:
    def __init__(self, status_code):
        self.status_code = status_code
        self.calls = 0

    def send(self, _request):
        self.calls += 1
        if self.status_code == 400:
            return TransportResponse(
                status_code=400,
                body=b"bad query",
                headers=(("Content-Type", "text/plain"),),
            )
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


class BlockingTransport(FixedTransport):
    def __init__(self):
        super().__init__(200)
        self.entered = threading.Event()
        self.release = threading.Event()

    def send(self, request):
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test transport was never released")
        return super().send(request)


class SearchRetryTest(unittest.TestCase):
    def _submit(self, engine, key="retry-search"):
        return submit_search(
            engine,
            question="Agent 记忆如何评估？",
            query_text="LLM agent memory",
            idempotency_key=key,
        )

    def _client(self, transport):
        return ArxivClient(
            transport=transport,
            clock=lambda: datetime(2026, 9, 23, 8, 0, tzinfo=timezone.utc),
        )

    def _retry_current(self, engine, job_id):
        current_id = get_job(engine, job_id).active_attempt_id
        return retry_search(engine, job_id, expected_attempt_id=current_id or "")

    def test_failed_job_retry_reserves_id_then_claim_creates_next_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = open_database(root / "papertrail.sqlite")
            try:
                submitted = self._submit(engine)
                with self.assertRaisesRegex(RuntimeError, "offline source failure"):
                    run_one_search(
                        engine,
                        worker_id="worker-a",
                        data_root=root / "data",
                        client=ExplodingClient(),
                    )
                first_id = get_job(engine, submitted.job_id).active_attempt_id
                first = get_attempt(engine, first_id)
                self.assertEqual(first.status, "failed")
                self.assertEqual(first.attempt_no, 1)

                reserved = self._retry_current(engine, submitted.job_id)

                job = get_job(engine, submitted.job_id)
                self.assertEqual(job.status, "queued")
                self.assertEqual(job.active_attempt_id, reserved.attempt_id)
                self.assertNotEqual(reserved.attempt_id, first_id)
                self.assertIsNone(get_attempt(engine, reserved.attempt_id))
                self.assertEqual([item.id for item in list_attempts(engine, submitted.job_id)], [first_id])
                self.assertEqual(get_attempt(engine, first_id).status, "failed")

                claimed = claim_next_search(engine, worker_id="worker-b")
                self.assertEqual(claimed.attempt_id, reserved.attempt_id)
                second = get_attempt(engine, reserved.attempt_id)
                self.assertEqual(second.attempt_no, 2)
                self.assertEqual(second.worker_id, "worker-b")
                self.assertEqual(second.status, "running")
                self.assertEqual(
                    [(item.id, item.attempt_no) for item in list_attempts(engine, submitted.job_id)],
                    [(first_id, 1), (reserved.attempt_id, 2)],
                )
            finally:
                engine.dispose()

    def test_retry_preserves_failed_artifact_and_replaces_current_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = open_database(root / "papertrail.sqlite")
            try:
                submitted = self._submit(engine)
                failed = run_one_search(
                    engine,
                    worker_id="worker-a",
                    data_root=root / "data",
                    client=self._client(FixedTransport(400)),
                )
                self.assertEqual(failed.status, "failed")
                previous_ref = get_attempt(engine, failed.attempt_id).artifact_ref
                self.assertEqual(get_search(engine, submitted.search_id).snapshot_ref, previous_ref)

                retry = self._retry_current(engine, submitted.job_id)
                self.assertIsNone(get_search(engine, submitted.search_id).snapshot_ref)
                self.assertEqual(get_attempt(engine, failed.attempt_id).artifact_ref, previous_ref)
                self.assertTrue((root / "data" / previous_ref).is_file())

                succeeded = run_one_search(
                    engine,
                    worker_id="worker-b",
                    data_root=root / "data",
                    client=self._client(FixedTransport(200)),
                )
                self.assertEqual(succeeded.attempt_id, retry.attempt_id)
                self.assertEqual(succeeded.status, "succeeded")
                self.assertEqual(get_job(engine, submitted.job_id).status, "succeeded")
                self.assertEqual(get_search(engine, submitted.search_id).snapshot_ref, succeeded.artifact_ref)
                self.assertEqual(get_attempt(engine, failed.attempt_id).artifact_ref, previous_ref)
                self.assertEqual([item.status for item in list_attempts(engine, submitted.job_id)], ["failed", "succeeded"])
            finally:
                engine.dispose()

    def test_interrupted_job_can_retry_but_queued_running_and_succeeded_cannot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = open_database(root / "papertrail.sqlite")
            try:
                submitted = self._submit(engine)
                with self.assertRaises(RetryNotAllowed):
                    self._retry_current(engine, submitted.job_id)
                claimed = claim_next_search(engine, worker_id="worker-a")
                with self.assertRaises(RetryNotAllowed):
                    self._retry_current(engine, submitted.job_id)
                lease_end = datetime.fromisoformat(
                    get_attempt(engine, claimed.attempt_id).lease_until.replace("Z", "+00:00")
                )
                self.assertTrue(
                    interrupt_expired_attempt(
                        engine,
                        claimed.attempt_id,
                        worker_stopped_confirmed=True,
                        now=lease_end + timedelta(seconds=1),
                    )
                )

                retry = self._retry_current(engine, submitted.job_id)
                with self.assertRaises(RetryNotAllowed):
                    self._retry_current(engine, submitted.job_id)
                self.assertEqual(get_job(engine, submitted.job_id).status, "queued")
                self.assertEqual(get_attempt(engine, claimed.attempt_id).status, "interrupted")
                self.assertEqual(get_job(engine, submitted.job_id).active_attempt_id, retry.attempt_id)
                completed = run_one_search(
                    engine,
                    worker_id="worker-b",
                    data_root=root / "data",
                    client=self._client(FixedTransport(200)),
                )
                self.assertEqual(completed.attempt_id, retry.attempt_id)
                self.assertEqual(completed.status, "succeeded")
                with self.assertRaises(RetryNotAllowed):
                    self._retry_current(engine, submitted.job_id)
            finally:
                engine.dispose()

    def test_retry_reservation_survives_restart_and_attempt_numbers_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "papertrail.sqlite"
            engine = open_database(database_path)
            try:
                submitted = self._submit(engine)
                with self.assertRaisesRegex(RuntimeError, "offline source failure"):
                    run_one_search(
                        engine,
                        worker_id="worker-a",
                        data_root=root / "data",
                        client=ExplodingClient(),
                    )
                second = self._retry_current(engine, submitted.job_id)
            finally:
                engine.dispose()

            reopened = open_database(database_path)
            try:
                with self.assertRaisesRegex(RuntimeError, "offline source failure"):
                    run_one_search(
                        reopened,
                        worker_id="worker-b",
                        data_root=root / "data",
                        client=ExplodingClient(),
                    )
                self.assertEqual(get_attempt(reopened, second.attempt_id).attempt_no, 2)
                third = self._retry_current(reopened, submitted.job_id)
                claimed = claim_next_search(reopened, worker_id="worker-c")
                self.assertEqual(claimed.attempt_id, third.attempt_id)
                self.assertEqual(get_attempt(reopened, third.attempt_id).attempt_no, 3)
                self.assertEqual(
                    [item.status for item in list_attempts(reopened, submitted.job_id)],
                    ["failed", "failed", "running"],
                )
            finally:
                reopened.dispose()

    def test_simultaneous_retry_requests_reserve_only_one_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = open_database(root / "papertrail.sqlite")
            try:
                submitted = self._submit(engine)
                with self.assertRaisesRegex(RuntimeError, "offline source failure"):
                    run_one_search(
                        engine,
                        worker_id="worker-a",
                        data_root=root / "data",
                        client=ExplodingClient(),
                    )
                first_id = get_job(engine, submitted.job_id).active_attempt_id
                barrier = Barrier(2)

                def try_retry():
                    barrier.wait(timeout=3)
                    try:
                        return (
                            "reserved",
                            retry_search(
                                engine,
                                submitted.job_id,
                                expected_attempt_id=first_id,
                            ).attempt_id,
                        )
                    except RetryNotAllowed:
                        return ("denied", None)

                with ThreadPoolExecutor(max_workers=2) as pool:
                    outcomes = list(pool.map(lambda _index: try_retry(), range(2)))
                self.assertEqual(sorted(item[0] for item in outcomes), ["denied", "reserved"])
                reserved_id = next(value for kind, value in outcomes if kind == "reserved")
                self.assertEqual(get_job(engine, submitted.job_id).active_attempt_id, reserved_id)
                self.assertEqual(claim_next_search(engine, worker_id="worker-b").attempt_id, reserved_id)
                self.assertEqual(
                    [item.attempt_no for item in list_attempts(engine, submitted.job_id)],
                    [1, 2],
                )
            finally:
                engine.dispose()

    def test_delayed_duplicate_cannot_retry_a_newer_failed_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = open_database(root / "papertrail.sqlite")
            try:
                submitted = self._submit(engine)
                with self.assertRaisesRegex(RuntimeError, "offline source failure"):
                    run_one_search(
                        engine,
                        worker_id="worker-a",
                        data_root=root / "data",
                        client=ExplodingClient(),
                    )
                first_id = get_job(engine, submitted.job_id).active_attempt_id
                second = retry_search(
                    engine, submitted.job_id, expected_attempt_id=first_id
                )
                with self.assertRaisesRegex(RuntimeError, "offline source failure"):
                    run_one_search(
                        engine,
                        worker_id="worker-b",
                        data_root=root / "data",
                        client=ExplodingClient(),
                    )
                self.assertEqual(get_job(engine, submitted.job_id).active_attempt_id, second.attempt_id)
                with self.assertRaises(RetryNotAllowed):
                    retry_search(
                        engine, submitted.job_id, expected_attempt_id=first_id
                    )
                self.assertEqual(get_job(engine, submitted.job_id).status, "failed")
                self.assertEqual(len(list_attempts(engine, submitted.job_id)), 2)
            finally:
                engine.dispose()

    def test_retry_update_time_does_not_move_backward_within_a_second(self):
        from sqlalchemy.orm import Session

        from backend.app.models import Job

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = open_database(root / "papertrail.sqlite")
            try:
                submitted = self._submit(engine)
                with self.assertRaisesRegex(RuntimeError, "offline source failure"):
                    run_one_search(
                        engine,
                        worker_id="worker-a",
                        data_root=root / "data",
                        client=ExplodingClient(),
                    )
                terminal_time = datetime(2026, 9, 23, 8, 0, 0, 500000, tzinfo=timezone.utc)
                with Session(engine) as session:
                    job = session.get(Job, submitted.job_id)
                    job.updated_at = terminal_time.isoformat().replace("+00:00", "Z")
                    session.commit()
                first_id = get_job(engine, submitted.job_id).active_attempt_id

                with patch("backend.app.jobs.service.datetime") as clock:
                    clock.now.return_value = terminal_time + timedelta(microseconds=1)
                    retry_search(
                        engine, submitted.job_id, expected_attempt_id=first_id
                    )

                with Session(engine) as session:
                    updated_at = session.get(Job, submitted.job_id).updated_at
                self.assertGreater(
                    datetime.fromisoformat(updated_at.replace("Z", "+00:00")),
                    terminal_time,
                )
            finally:
                engine.dispose()

    def test_recovery_cli_can_retry_and_preview_reserved_attempt(self):
        from scripts.recover_search_job import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "papertrail.sqlite"
            engine = open_database(database_path)
            try:
                submitted = self._submit(engine)
                with self.assertRaisesRegex(RuntimeError, "offline source failure"):
                    run_one_search(
                        engine,
                        worker_id="worker-a",
                        data_root=root / "data",
                        client=ExplodingClient(),
                    )
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(
                        main(["--database", str(database_path), "--job-id", submitted.job_id]),
                        0,
                    )
                self.assertIn("[PREVIEW]", output.getvalue())
                self.assertEqual(get_job(engine, submitted.job_id).status, "failed")

                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(
                        main(["--database", str(database_path), "--job-id", submitted.job_id, "--retry"]),
                        0,
                    )
                self.assertIn("[QUEUED]", output.getvalue())
                reserved_id = get_job(engine, submitted.job_id).active_attempt_id

                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(
                        main(["--database", str(database_path), "--job-id", submitted.job_id]),
                        0,
                    )
                self.assertIn(f"reserved_attempt={reserved_id}", output.getvalue())
                self.assertIn("[PREVIEW]", output.getvalue())
                self.assertIsNone(get_attempt(engine, reserved_id))
            finally:
                engine.dispose()

    def test_late_old_attempt_cannot_replace_successful_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = open_database(root / "papertrail.sqlite")
            old_transport = BlockingTransport()
            outcome = {}
            try:
                submitted = self._submit(engine)

                def run_old():
                    try:
                        run_one_search(
                            engine,
                            worker_id="worker-a",
                            data_root=root / "data",
                            client=self._client(old_transport),
                            heartbeat_interval_seconds=0.02,
                            lease_seconds=0.1,
                        )
                    except Exception as error:
                        outcome["error"] = error

                # A broken heartbeat claims success without renewing the DB
                # lease; a stale worker may still return after a new attempt.
                with patch("backend.app.jobs.worker.refresh_lease", return_value=True):
                    old_thread = threading.Thread(target=run_old)
                    old_thread.start()
                    self.assertTrue(old_transport.entered.wait(timeout=3))
                    old_id = get_job(engine, submitted.job_id).active_attempt_id
                    lease_end = datetime.fromisoformat(
                        get_attempt(engine, old_id).lease_until.replace("Z", "+00:00")
                    )
                    while datetime.now(timezone.utc) <= lease_end:
                        time.sleep(0.01)
                    self.assertTrue(
                        interrupt_expired_attempt(
                            engine,
                            old_id,
                            worker_stopped_confirmed=True,
                        )
                    )
                try:
                    reserved = self._retry_current(engine, submitted.job_id)
                    fresh = run_one_search(
                        engine,
                        worker_id="worker-b",
                        data_root=root / "data",
                        client=self._client(FixedTransport(200)),
                    )
                finally:
                    old_transport.release.set()
                    old_thread.join(timeout=5)
                self.assertFalse(old_thread.is_alive())
                self.assertIsInstance(outcome.get("error"), LostClaim)
                self.assertEqual(fresh.attempt_id, reserved.attempt_id)
                self.assertEqual(get_job(engine, submitted.job_id).status, "succeeded")
                self.assertEqual(get_search(engine, submitted.search_id).snapshot_ref, fresh.artifact_ref)
                self.assertEqual(get_attempt(engine, old_id).status, "interrupted")
                self.assertFalse(
                    (root / "data" / "runs" / f"search-{old_id}" / "cache-committed.json").exists()
                )
                self.assertEqual(
                    [item.status for item in list_attempts(engine, submitted.job_id)],
                    ["interrupted", "succeeded"],
                )
            finally:
                old_transport.release.set()
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
