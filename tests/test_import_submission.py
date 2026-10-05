import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.orm import Session

from backend.app.collections.service import (
    SelectedSource,
    StaleSelection,
    create_collection,
    get_collection,
    set_collection_selection,
)
from backend.app.db import open_database
from backend.app.jobs.import_submission import (
    InvalidJobRequest,
    get_import_request,
    submit_import_job,
)
from backend.app.jobs.service import IdempotencyConflict, get_job, list_attempts
from backend.app.models import Base, Job, JobRequest
from tests.test_ingestion_selection import saved_search


class ImportSubmissionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.database_path = self.root / "papertrail.sqlite"
        self.data_root = self.root / "data"
        self.run_dir = saved_search(self.data_root)
        self.engine = open_database(self.database_path)
        self.addCleanup(self.engine.dispose)
        self.collection = create_collection(self.engine)

    def _select(self, expected_revision, *versioned_ids):
        return set_collection_selection(
            self.engine,
            self.collection.id,
            expected_revision=expected_revision,
            data_root=self.data_root,
            selections=tuple(
                SelectedSource(self.run_dir, versioned_id)
                for versioned_id in versioned_ids
            ),
        )

    def _submit(self, revision, key="import-operation"):
        return submit_import_job(
            self.engine,
            self.collection.id,
            expected_selection_revision=revision,
            idempotency_key=key,
        )

    def test_submission_freezes_order_and_source_hash_after_draft_edit_and_reopen(self):
        original = self._select(0, "2401.00001v3", "2401.00001v1")
        submitted = self._submit(1)
        self._select(1, "2401.00001v1", "cs/9901001v2")

        reopened = open_database(self.database_path)
        try:
            request = get_import_request(reopened, submitted.job_id)
            self.assertEqual(request.collection_id, self.collection.id)
            self.assertEqual(request.selection_revision, 1)
            self.assertEqual(
                [item.versioned_id for item in request.selections],
                ["2401.00001v3", "2401.00001v1"],
            )
            self.assertEqual(request.selections, original.selections)
            self.assertTrue(all(len(item.snapshot_sha256) == 64 for item in request.selections))
            self.assertEqual(get_job(reopened, submitted.job_id).status, "queued")
            self.assertEqual(get_job(reopened, submitted.job_id).resource_id, self.collection.id)
            self.assertEqual(list_attempts(reopened, submitted.job_id), ())
        finally:
            reopened.dispose()

    def test_lost_response_replay_survives_draft_edit_and_different_body_conflicts(self):
        self._select(0, "2401.00001v1")
        first = self._submit(1)
        self._select(1, "2401.00001v3")

        reopened = open_database(self.database_path)
        try:
            repeated = submit_import_job(
                reopened,
                self.collection.id,
                expected_selection_revision=1,
                idempotency_key="import-operation",
            )
        finally:
            reopened.dispose()
        self.assertEqual(repeated, first)
        with self.assertRaises(IdempotencyConflict):
            self._submit(2)
        with Session(self.engine) as session:
            self.assertEqual(
                session.scalar(select(func.count()).select_from(Job).where(Job.kind == "import")),
                1,
            )
            self.assertEqual(session.scalar(select(func.count()).select_from(JobRequest)), 1)

    def test_empty_or_stale_selection_never_queues_import(self):
        with self.assertRaises(ValueError):
            self._submit(0)
        self._select(0, "2401.00001v1")
        with self.assertRaises(StaleSelection):
            self._submit(0)
        with Session(self.engine) as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(Job)), 0)
            self.assertEqual(session.scalar(select(func.count()).select_from(JobRequest)), 0)

    def test_tampered_saved_request_is_rejected(self):
        self._select(0, "2401.00001v1")
        submitted = self._submit(1)
        with Session(self.engine) as session:
            request = session.get(JobRequest, submitted.job_id)
            request.request_json += " "
            session.commit()
        with self.assertRaises(InvalidJobRequest):
            get_import_request(self.engine, submitted.job_id)

    def test_concurrent_same_key_creates_one_job_and_one_request(self):
        self._select(0, "2401.00001v1")
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _index: self._submit(1), range(2)))
        self.assertEqual(results[0], results[1])
        with Session(self.engine) as session:
            self.assertEqual(
                session.scalar(select(func.count()).select_from(Job).where(Job.kind == "import")),
                1,
            )
            self.assertEqual(session.scalar(select(func.count()).select_from(JobRequest)), 1)


class ImportSchemaUpgradeTest(unittest.TestCase):
    def test_existing_collection_survives_adding_job_request_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.sqlite"
            old_engine = create_engine(f"sqlite:///{path.as_posix()}")
            try:
                Base.metadata.create_all(
                    old_engine,
                    tables=[
                        table for table in Base.metadata.sorted_tables
                        if table.name != "job_requests"
                    ],
                )
                collection = create_collection(old_engine)
            finally:
                old_engine.dispose()

            upgraded = open_database(path)
            try:
                self.assertTrue(inspect(upgraded).has_table("job_requests"))
                self.assertEqual(get_collection(upgraded, collection.id).id, collection.id)
                data_root = Path(directory) / "data"
                run_dir = saved_search(data_root)
                set_collection_selection(
                    upgraded,
                    collection.id,
                    expected_revision=0,
                    data_root=data_root,
                    selections=(SelectedSource(run_dir, "2401.00001v1"),),
                )
                submitted = submit_import_job(
                    upgraded,
                    collection.id,
                    expected_selection_revision=1,
                    idempotency_key="after-upgrade",
                )
                self.assertEqual(get_import_request(upgraded, submitted.job_id).selection_revision, 1)
            finally:
                upgraded.dispose()


if __name__ == "__main__":
    unittest.main()
