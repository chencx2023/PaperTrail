import asyncio
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import httpx
from sqlalchemy import update
from sqlalchemy.orm import Session

from backend.app.collections.service import SelectedSource, create_collection, set_collection_selection
from backend.app.db import open_database
from backend.app.ingestion.batch import get_import_report, publish_import_run, stage_import_job
from backend.app.jobs.claim import claim_next_job
from backend.app.jobs.dispatcher import run_one_job
from backend.app.jobs.import_submission import (
    InvalidJobRequest,
    ImportRetryNotAllowed,
    get_import_request,
    retry_import_job,
    submit_import_job,
)
from backend.app.jobs.ownership import LostClaim, finish_active_attempt
from backend.app.jobs.service import get_attempt, get_job
from backend.app.jobs.worker import interrupt_expired_attempt
from backend.app.models import CollectionImportItem, JobAttemptResult, JobRequest
from scripts.job_worker import main as job_worker_main
from tests.test_document_preparation import text_pdf_bytes
from tests.test_ingestion_selection import saved_search


class ImportWorkerTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data_root = self.root / "data"
        self.run_dir = saved_search(self.data_root)
        self.database = self.root / "papertrail.sqlite"
        self.engine = open_database(self.database)
        self.addCleanup(self.engine.dispose)
        self.collection = create_collection(self.engine)

    def _submit(self, *versioned_ids):
        selected = set_collection_selection(
            self.engine,
            self.collection.id,
            expected_revision=0,
            data_root=self.data_root,
            selections=tuple(SelectedSource(self.run_dir, item) for item in versioned_ids),
        )
        return submit_import_job(
            self.engine,
            self.collection.id,
            expected_selection_revision=selected.selection_revision,
            idempotency_key="worker-import",
        )

    def test_worker_uses_submitted_list_and_publishes_partial_report(self):
        submitted = self._submit("2401.00001v1", "2401.00001v3")
        set_collection_selection(
            self.engine,
            self.collection.id,
            expected_revision=1,
            data_root=self.data_root,
            selections=(SelectedSource(self.run_dir, "cs/9901001v2"),),
        )
        requested = []

        def response(request):
            requested.append(request.url.path)
            if request.url.path.endswith("v1"):
                return httpx.Response(
                    200, headers={"Content-Type": "application/pdf"}, content=text_pdf_bytes()
                )
            return httpx.Response(404, content=b"missing")

        with httpx.Client(transport=httpx.MockTransport(response)) as client:
            executed = run_one_job(
                self.engine,
                worker_id="worker-import",
                data_root=self.data_root,
                import_client=client,
            )

        self.assertEqual(executed.kind, "import")
        self.assertEqual(executed.status, "succeeded")
        self.assertEqual(len(requested), 2)
        self.assertTrue(requested[0].endswith("v1"))
        self.assertTrue(requested[1].endswith("v3"))
        reopened = open_database(self.database)
        try:
            job = get_job(reopened, submitted.job_id)
            attempt = get_attempt(reopened, job.active_attempt_id)
            report = get_import_report(reopened, executed.result_resource_id)
            self.assertEqual(job.status, "succeeded")
            self.assertEqual(attempt.status, "succeeded")
            self.assertEqual(report.status, "partial_failed")
            self.assertEqual(report.selection_revision, 1)
            self.assertEqual(
                [item.versioned_id for item in report.items],
                ["2401.00001v1", "2401.00001v3"],
            )
            self.assertEqual(
                [item.status for item in report.items], ["succeeded", "download_failed"]
            )
            with Session(reopened) as session:
                binding = session.get(JobAttemptResult, attempt.id)
                self.assertEqual(binding.resource_id, report.id)
        finally:
            reopened.dispose()

    def test_cli_saves_all_failed_report_without_claiming_job_success_means_pdf_success(self):
        submitted = self._submit("2401.00001v1")
        output = StringIO()
        with httpx.Client(transport=httpx.MockTransport(
            lambda _request: httpx.Response(404, content=b"missing")
        )) as client, redirect_stdout(output):
            exit_code = job_worker_main(
                [
                    "--database", str(self.database),
                    "--data-root", str(self.data_root),
                    "--worker-id", "worker-cli",
                    "--live", "--once",
                ],
                import_client=client,
            )
        self.assertEqual(exit_code, 0)
        self.assertIn("[SUCCEEDED] kind=import", output.getvalue())
        job = get_job(self.engine, submitted.job_id)
        self.assertEqual(job.status, "succeeded")
        with Session(self.engine) as session:
            binding = session.get(JobAttemptResult, job.active_attempt_id)
            report = get_import_report(self.engine, binding.resource_id)
        self.assertEqual(report.status, "failed")
        self.assertEqual(report.items[0].status, "download_failed")

    def test_old_attempt_cannot_publish_after_interruption_and_retry(self):
        submitted = self._submit("2401.00001v1")
        old = claim_next_job(
            self.engine,
            worker_id="worker-old",
            supported_kinds=("import",),
            lease_seconds=0.01,
        )
        request = get_import_request(self.engine, submitted.job_id)
        with httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(
            200, headers={"Content-Type": "application/pdf"}, content=text_pdf_bytes()
        ))) as client:
            staged = asyncio.run(stage_import_job(
                self.engine, request, attempt_id=old.attempt_id,
                data_root=self.data_root, client=client,
            ))
        self.assertEqual(get_import_report(self.engine, staged.run_id).status, "running")
        lease_until = datetime.fromisoformat(
            get_attempt(self.engine, old.attempt_id).lease_until.replace("Z", "+00:00")
        )
        self.assertTrue(interrupt_expired_attempt(
            self.engine, old.attempt_id, worker_stopped_confirmed=True,
            now=lease_until + timedelta(seconds=1),
        ))
        reserved = retry_import_job(
            self.engine, submitted.job_id, expected_attempt_id=old.attempt_id
        )
        with self.assertRaises(LostClaim):
            finish_active_attempt(
                self.engine,
                job_id=old.job_id,
                attempt_id=old.attempt_id,
                worker_id=old.worker_id,
                status="succeeded",
                finished_at="2026-09-24T00:00:00Z",
                result_resource_id=staged.run_id,
                publish=lambda session: publish_import_run(
                    session, staged, finished_at="2026-09-24T00:00:00Z"
                ),
            )
        self.assertEqual(get_import_report(self.engine, staged.run_id).status, "running")
        with Session(self.engine) as session:
            self.assertIsNone(session.get(JobAttemptResult, old.attempt_id))

        def no_request(_request):
            self.fail("The retry should reuse the already prepared fixed PDF.")

        with httpx.Client(transport=httpx.MockTransport(no_request)) as client:
            executed = run_one_job(
                self.engine, worker_id="worker-new", data_root=self.data_root,
                import_client=client,
            )
        self.assertEqual(executed.attempt_id, reserved.attempt_id)
        self.assertNotEqual(executed.result_resource_id, staged.run_id)
        self.assertEqual(get_import_report(self.engine, executed.result_resource_id).status, "succeeded")
        self.assertEqual(get_job(self.engine, submitted.job_id).status, "succeeded")
        self.assertEqual(get_import_report(self.engine, staged.run_id).status, "running")
        with self.assertRaises(ImportRetryNotAllowed):
            retry_import_job(
                self.engine, submitted.job_id, expected_attempt_id=old.attempt_id
            )

    def test_tampered_request_fails_job_before_pdf_request(self):
        submitted = self._submit("2401.00001v1")
        with Session(self.engine) as session:
            row = session.get(JobRequest, submitted.job_id)
            row.request_json += " "
            session.commit()

        def no_request(_request):
            self.fail("A damaged request must not download a PDF.")

        with httpx.Client(transport=httpx.MockTransport(no_request)) as client:
            with self.assertRaises(InvalidJobRequest):
                run_one_job(
                    self.engine, worker_id="worker-import",
                    data_root=self.data_root, import_client=client,
                )
        job = get_job(self.engine, submitted.job_id)
        self.assertEqual(job.status, "failed")
        attempt = get_attempt(self.engine, job.active_attempt_id)
        self.assertEqual(attempt.error_kind, "InvalidJobRequest")
        with Session(self.engine) as session:
            self.assertIsNone(session.get(JobAttemptResult, attempt.id))

    def test_lease_renews_while_pdf_request_is_waiting(self):
        submitted = self._submit("2401.00001v1")
        entered = threading.Event()
        release = threading.Event()
        outcome = {}

        def delayed_pdf(_request):
            entered.set()
            if not release.wait(timeout=10):
                raise RuntimeError("Test PDF request was not released.")
            return httpx.Response(
                200, headers={"Content-Type": "application/pdf"}, content=text_pdf_bytes()
            )

        with httpx.Client(transport=httpx.MockTransport(delayed_pdf)) as client:
            def execute():
                try:
                    outcome["result"] = run_one_job(
                        self.engine,
                        worker_id="worker-import",
                        data_root=self.data_root,
                        import_client=client,
                        lease_seconds=0.5,
                        heartbeat_interval_seconds=0.05,
                    )
                except Exception as error:
                    outcome["error"] = error

            worker = threading.Thread(target=execute)
            worker.start()
            try:
                self.assertTrue(entered.wait(timeout=5))
                attempt_id = get_job(self.engine, submitted.job_id).active_attempt_id
                first = get_attempt(self.engine, attempt_id)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    renewed = get_attempt(self.engine, attempt_id)
                    if renewed.heartbeat_at > first.heartbeat_at:
                        break
                    time.sleep(0.02)
                else:
                    self.fail("Import attempt heartbeat did not advance.")
                self.assertGreater(renewed.lease_until, first.lease_until)
            finally:
                release.set()
                worker.join(timeout=15)
            self.assertFalse(worker.is_alive())
            self.assertNotIn("error", outcome)
            self.assertEqual(outcome["result"].status, "succeeded")

    def test_lease_renews_while_pdf_text_parsing_is_waiting(self):
        submitted = self._submit("2401.00001v1")
        entered = threading.Event()
        release = threading.Event()
        outcome = {}

        from backend.app.ingestion import preparation

        original_read_doc = preparation.read_doc

        async def delayed_read_doc(*args, **kwargs):
            entered.set()
            if not await asyncio.to_thread(release.wait, 10):
                raise RuntimeError("Test PDF parser was not released.")
            return await original_read_doc(*args, **kwargs)

        def pdf(_request):
            return httpx.Response(
                200, headers={"Content-Type": "application/pdf"}, content=text_pdf_bytes()
            )

        with httpx.Client(transport=httpx.MockTransport(pdf)) as client, patch(
            "backend.app.ingestion.preparation.read_doc", new=delayed_read_doc
        ):
            def execute():
                try:
                    outcome["result"] = run_one_job(
                        self.engine,
                        worker_id="worker-import",
                        data_root=self.data_root,
                        import_client=client,
                        lease_seconds=0.5,
                        heartbeat_interval_seconds=0.05,
                    )
                except Exception as error:
                    outcome["error"] = error

            worker = threading.Thread(target=execute)
            worker.start()
            try:
                self.assertTrue(entered.wait(timeout=5))
                attempt_id = get_job(self.engine, submitted.job_id).active_attempt_id
                first = get_attempt(self.engine, attempt_id)
                original_lease_end = datetime.fromisoformat(
                    first.lease_until.replace("Z", "+00:00")
                )
                deadline = time.monotonic() + 3
                while datetime.now(timezone.utc) <= original_lease_end:
                    if time.monotonic() >= deadline:
                        self.fail("The initial PDF parsing lease did not expire in time.")
                    time.sleep(0.02)
                renewed = get_attempt(self.engine, attempt_id)
                self.assertGreater(renewed.heartbeat_at, first.heartbeat_at)
                self.assertGreater(renewed.lease_until, first.lease_until)
                self.assertGreater(
                    datetime.fromisoformat(renewed.lease_until.replace("Z", "+00:00")),
                    datetime.now(timezone.utc),
                )
                self.assertEqual(get_job(self.engine, submitted.job_id).status, "running")
            finally:
                release.set()
                worker.join(timeout=15)
            self.assertFalse(worker.is_alive())
            self.assertNotIn("error", outcome)
            self.assertEqual(outcome["result"].status, "succeeded")
            self.assertEqual(get_job(self.engine, submitted.job_id).status, "succeeded")
            self.assertEqual(
                get_import_report(self.engine, outcome["result"].result_resource_id).status,
                "succeeded",
            )

    def test_changed_report_item_cannot_be_published_as_current_result(self):
        submitted = self._submit("2401.00001v1", "2401.00001v3")
        claimed = claim_next_job(
            self.engine, worker_id="worker-import", supported_kinds=("import",)
        )
        request = get_import_request(self.engine, submitted.job_id)
        with httpx.Client(transport=httpx.MockTransport(
            lambda _request: httpx.Response(404, content=b"missing")
        )) as client:
            staged = asyncio.run(stage_import_job(
                self.engine, request, attempt_id=claimed.attempt_id,
                data_root=self.data_root, client=client,
            ))
        with Session(self.engine) as session:
            session.execute(
                update(CollectionImportItem)
                .where(
                    CollectionImportItem.import_run_id == staged.run_id,
                    CollectionImportItem.position == 1,
                )
                .values(source_ref="runs/another-search")
            )
            session.commit()

        with self.assertRaisesRegex(RuntimeError, "does not match"):
            finish_active_attempt(
                self.engine,
                job_id=claimed.job_id,
                attempt_id=claimed.attempt_id,
                worker_id=claimed.worker_id,
                status="succeeded",
                finished_at="2026-09-24T00:00:00Z",
                result_resource_id=staged.run_id,
                publish=lambda session: publish_import_run(
                    session, staged, finished_at="2026-09-24T00:00:00Z"
                ),
            )
        self.assertEqual(get_job(self.engine, submitted.job_id).status, "running")
        self.assertEqual(get_import_report(self.engine, staged.run_id).status, "running")
        with Session(self.engine) as session:
            self.assertIsNone(session.get(JobAttemptResult, claimed.attempt_id))

    def test_report_cannot_claim_success_without_a_prepared_document(self):
        submitted = self._submit("2401.00001v1")
        claimed = claim_next_job(
            self.engine, worker_id="worker-import", supported_kinds=("import",)
        )
        request = get_import_request(self.engine, submitted.job_id)
        with httpx.Client(transport=httpx.MockTransport(
            lambda _request: httpx.Response(404, content=b"missing")
        )) as client:
            staged = asyncio.run(stage_import_job(
                self.engine, request, attempt_id=claimed.attempt_id,
                data_root=self.data_root, client=client,
            ))
        with Session(self.engine) as session:
            session.execute(
                update(CollectionImportItem)
                .where(CollectionImportItem.import_run_id == staged.run_id)
                .values(status="succeeded")
            )
            session.commit()

        with Session(self.engine) as session:
            with self.assertRaisesRegex(RuntimeError, "prepared document"):
                publish_import_run(session, staged, finished_at="2026-09-24T00:00:00Z")
        self.assertEqual(get_import_report(self.engine, staged.run_id).status, "running")


if __name__ == "__main__":
    unittest.main()
