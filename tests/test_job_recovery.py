import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from backend.app.db import open_database
from backend.app.jobs.service import get_attempt, get_job, get_search, submit_search
from backend.app.jobs.worker import (
    LostClaim,
    claim_next_search,
    interrupt_expired_attempt,
    refresh_lease,
    run_one_search,
)
from backend.app.search.client import ArxivClient, TransportResponse
from backend.app.search.service import run_search


class WaitingTransport:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def send(self, _request):
        self.calls += 1
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test transport was never released")
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


class JobRecoveryTest(unittest.TestCase):
    def _submit(self, engine):
        return submit_search(
            engine,
            question="Agent 记忆如何评估？",
            query_text="LLM agent memory",
            idempotency_key="recover-one-search",
        )

    def _client(self, transport):
        return ArxivClient(
            transport=transport,
            clock=lambda: datetime(2026, 9, 23, 8, 0, tzinfo=timezone.utc),
        )

    def test_claim_sets_lease_and_only_current_worker_can_refresh_it(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = open_database(Path(directory) / "papertrail.sqlite")
            try:
                submitted = self._submit(engine)
                claimed = claim_next_search(engine, worker_id="worker-a")
                first = get_attempt(engine, claimed.attempt_id)
                self.assertIsNotNone(first.lease_until)
                self.assertGreater(first.lease_until, first.heartbeat_at)
                self.assertFalse(refresh_lease(engine, claimed.attempt_id, "worker-b"))
                self.assertTrue(refresh_lease(engine, claimed.attempt_id, "worker-a"))
                renewed = get_attempt(engine, claimed.attempt_id)
                self.assertGreater(renewed.heartbeat_at, first.heartbeat_at)
                self.assertGreater(renewed.lease_until, first.lease_until)
                self.assertEqual(get_job(engine, submitted.job_id).status, "running")
            finally:
                engine.dispose()

    def test_heartbeat_advances_while_search_call_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = open_database(root / "papertrail.sqlite")
            transport = WaitingTransport()
            outcome = {}
            try:
                submitted = self._submit(engine)

                def execute():
                    try:
                        outcome["result"] = run_one_search(
                            engine,
                            worker_id="worker-a",
                            data_root=root / "data",
                            client=self._client(transport),
                            heartbeat_interval_seconds=0.05,
                            lease_seconds=0.5,
                        )
                    except Exception as error:
                        outcome["error"] = error

                worker = threading.Thread(target=execute)
                worker.start()
                try:
                    self.assertTrue(transport.entered.wait(timeout=3))
                    attempt_id = get_job(engine, submitted.job_id).active_attempt_id
                    first = get_attempt(engine, attempt_id)
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline:
                        renewed = get_attempt(engine, attempt_id)
                        if renewed.heartbeat_at > first.heartbeat_at:
                            break
                        time.sleep(0.02)
                    else:
                        self.fail("heartbeat did not advance during a blocked source call")
                    self.assertGreater(renewed.lease_until, first.lease_until)
                finally:
                    transport.release.set()
                    worker.join(timeout=5)
                self.assertFalse(worker.is_alive())
                self.assertNotIn("error", outcome)
                self.assertEqual(outcome["result"].status, "succeeded")
                self.assertFalse(refresh_lease(engine, attempt_id, "worker-a"))
            finally:
                transport.release.set()
                engine.dispose()

    def test_expired_lease_needs_explicit_stop_confirmation_to_interrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = open_database(Path(directory) / "papertrail.sqlite")
            try:
                submitted = self._submit(engine)
                claimed = claim_next_search(engine, worker_id="worker-a")
                lease_end = datetime.fromisoformat(
                    get_attempt(engine, claimed.attempt_id).lease_until.replace("Z", "+00:00")
                )
                self.assertFalse(
                    interrupt_expired_attempt(
                        engine,
                        claimed.attempt_id,
                        worker_stopped_confirmed=True,
                        now=lease_end - timedelta(microseconds=1),
                    )
                )
                with self.assertRaisesRegex(ValueError, "confirm"):
                    interrupt_expired_attempt(
                        engine,
                        claimed.attempt_id,
                        worker_stopped_confirmed=False,
                        now=lease_end + timedelta(seconds=1),
                    )
                self.assertEqual(get_job(engine, submitted.job_id).status, "running")
                self.assertTrue(
                    interrupt_expired_attempt(
                        engine,
                        claimed.attempt_id,
                        worker_stopped_confirmed=True,
                        now=lease_end + timedelta(seconds=1),
                    )
                )
                self.assertFalse(
                    interrupt_expired_attempt(
                        engine,
                        claimed.attempt_id,
                        worker_stopped_confirmed=True,
                        now=lease_end + timedelta(seconds=2),
                    )
                )
                self.assertEqual(get_job(engine, submitted.job_id).status, "interrupted")
                self.assertEqual(get_attempt(engine, claimed.attempt_id).status, "interrupted")
                self.assertIsNone(get_search(engine, submitted.search_id).snapshot_ref)
                self.assertFalse(refresh_lease(engine, claimed.attempt_id, "worker-a"))
            finally:
                engine.dispose()

    def test_late_result_cannot_overwrite_interrupted_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = open_database(root / "papertrail.sqlite")
            transport = WaitingTransport()
            outcome = {}
            try:
                submitted = self._submit(engine)

                def execute():
                    try:
                        outcome["result"] = run_one_search(
                            engine,
                            worker_id="worker-a",
                            data_root=root / "data",
                            client=ArxivClient(
                                transport=transport,
                                clock=lambda: datetime.now(timezone.utc),
                            ),
                            heartbeat_interval_seconds=0.02,
                            lease_seconds=0.1,
                        )
                    except Exception as error:
                        outcome["error"] = error

                # Simulate a mistaken stop confirmation while a blocked call can
                # still return. Even then, its result must be fenced at commit.
                with patch(
                    "backend.app.jobs.worker.refresh_lease", return_value=False
                ):
                    worker = threading.Thread(target=execute)
                    worker.start()
                    try:
                        self.assertTrue(transport.entered.wait(timeout=3))
                        attempt_id = get_job(engine, submitted.job_id).active_attempt_id
                        lease_end = datetime.fromisoformat(
                            get_attempt(engine, attempt_id).lease_until.replace(
                                "Z", "+00:00"
                            )
                        )
                        deadline = time.monotonic() + 3
                        while time.monotonic() < deadline:
                            if datetime.now(timezone.utc) > lease_end:
                                break
                            time.sleep(0.01)
                        self.assertTrue(
                            interrupt_expired_attempt(
                                engine,
                                attempt_id,
                                worker_stopped_confirmed=True,
                            )
                        )
                    finally:
                        transport.release.set()
                        worker.join(timeout=5)
                self.assertFalse(worker.is_alive())
                self.assertIsInstance(outcome.get("error"), LostClaim)
                self.assertNotIn("result", outcome)
                self.assertEqual(get_job(engine, submitted.job_id).status, "interrupted")
                attempt = get_attempt(engine, attempt_id)
                self.assertEqual(attempt.status, "interrupted")
                self.assertIsNone(attempt.artifact_ref)
                self.assertIsNone(get_search(engine, submitted.search_id).snapshot_ref)
                self.assertTrue(
                    (root / "data" / "runs" / f"search-{attempt_id}" / "run.json").is_file()
                )
                later_transport = WaitingTransport()
                later_transport.release.set()
                later = run_search(
                    "LLM agent memory",
                    client=ArxivClient(
                        transport=later_transport,
                        clock=lambda: datetime.now(timezone.utc),
                    ),
                    output_root=root / "data" / "runs",
                    run_id="search-later",
                )
                self.assertFalse(later.cache_hit)
                self.assertEqual(later_transport.calls, 1)
            finally:
                transport.release.set()
                engine.dispose()

    def test_recovery_cli_previews_before_explicit_interruption(self):
        from scripts.recover_search_job import main

        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "papertrail.sqlite"
            engine = open_database(database_path)
            try:
                submitted = self._submit(engine)
                claimed = claim_next_search(
                    engine, worker_id="worker-a", lease_seconds=0.05
                )
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(
                        main(["--database", str(database_path), "--job-id", submitted.job_id]),
                        0,
                    )
                self.assertIn("[PREVIEW]", output.getvalue())
                self.assertIn("worker-a", output.getvalue())
                self.assertEqual(get_job(engine, submitted.job_id).status, "running")

                lease_end = datetime.fromisoformat(
                    get_attempt(engine, claimed.attempt_id).lease_until.replace("Z", "+00:00")
                )
                while datetime.now(timezone.utc) <= lease_end:
                    time.sleep(0.01)
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(
                        main(
                            [
                                "--database",
                                str(database_path),
                                "--job-id",
                                submitted.job_id,
                                "--confirm-worker-stopped",
                            ]
                        ),
                        0,
                    )
                self.assertIn("[INTERRUPTED]", output.getvalue())
                self.assertEqual(get_job(engine, submitted.job_id).status, "interrupted")
            finally:
                engine.dispose()

    def test_crashed_worker_requires_explicit_interruption_before_retry(self):
        from scripts.recover_search_job import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "papertrail.sqlite"
            engine = open_database(database_path)
            submitted = self._submit(engine)
            engine.dispose()

            crash_after_claim = (
                "import os, sys\n"
                "from pathlib import Path\n"
                "from backend.app.db import open_database\n"
                "from backend.app.jobs.worker import claim_next_search\n"
                "engine = open_database(Path(sys.argv[1]), create_schema=False)\n"
                "claimed = claim_next_search(engine, worker_id='crashed-worker', lease_seconds=0.5)\n"
                "assert claimed is not None\n"
                "os._exit(17)\n"
            )
            crashed = subprocess.run(
                [sys.executable, "-c", crash_after_claim, str(database_path)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            self.assertEqual(crashed.returncode, 17, crashed.stderr)

            reopened = open_database(database_path, create_schema=False)
            try:
                job = get_job(reopened, submitted.job_id)
                old_attempt_id = job.active_attempt_id
                self.assertEqual(job.status, "running")
                self.assertEqual(get_attempt(reopened, old_attempt_id).worker_id, "crashed-worker")

                def recover(*flags):
                    output = StringIO()
                    with redirect_stdout(output):
                        exit_code = main([
                            "--database", str(database_path),
                            "--job-id", submitted.job_id,
                            *flags,
                        ])
                    return exit_code, output.getvalue()

                lease_end = datetime.fromisoformat(
                    get_attempt(reopened, old_attempt_id).lease_until.replace("Z", "+00:00")
                )
                deadline = time.monotonic() + 3
                while datetime.now(timezone.utc) <= lease_end:
                    if time.monotonic() >= deadline:
                        self.fail("The crashed worker's lease did not expire in time.")
                    time.sleep(0.01)

                preview_code, preview = recover()
                self.assertEqual(preview_code, 0)
                self.assertIn("lease_expired=True", preview)
                self.assertIn("[PREVIEW]", preview)
                self.assertEqual(get_job(reopened, submitted.job_id).status, "running")
                retry_code, retry_output = recover("--retry")
                self.assertEqual(retry_code, 1)
                self.assertIn("[UNCHANGED]", retry_output)
                self.assertEqual(get_job(reopened, submitted.job_id).status, "running")

                interrupt_code, interrupt_output = recover("--confirm-worker-stopped")
                self.assertEqual(interrupt_code, 0)
                self.assertIn("[INTERRUPTED]", interrupt_output)
                self.assertEqual(get_attempt(reopened, old_attempt_id).status, "interrupted")
                queue_code, queue_output = recover("--retry")
                self.assertEqual(queue_code, 0)
                self.assertIn("[QUEUED]", queue_output)
                queued = get_job(reopened, submitted.job_id)
                self.assertEqual(queued.status, "queued")
                self.assertNotEqual(queued.active_attempt_id, old_attempt_id)
                self.assertIsNone(get_attempt(reopened, queued.active_attempt_id))

                transport = WaitingTransport()
                transport.release.set()
                executed = run_one_search(
                    reopened, worker_id="replacement-worker", data_root=root / "data",
                    client=self._client(transport),
                )
                self.assertEqual(executed.attempt_id, queued.active_attempt_id)
                self.assertEqual(get_job(reopened, submitted.job_id).status, "succeeded")
                self.assertEqual(get_attempt(reopened, old_attempt_id).status, "interrupted")
                self.assertEqual(get_attempt(reopened, executed.attempt_id).status, "succeeded")
                self.assertEqual(get_search(reopened, submitted.search_id).snapshot_ref, executed.artifact_ref)
                self.assertEqual(transport.calls, 1)
            finally:
                reopened.dispose()

    def test_recovery_preview_does_not_bootstrap_an_empty_database(self):
        from scripts.recover_search_job import main

        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "empty.sqlite"
            sqlite3.connect(database_path).close()
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(["--database", str(database_path), "--job-id", "missing"]),
                    1,
                )
            connection = sqlite3.connect(database_path)
            try:
                tables = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            finally:
                connection.close()
            self.assertEqual(tables, [])
            self.assertIn("[INVALID DATABASE]", output.getvalue())


if __name__ == "__main__":
    unittest.main()
