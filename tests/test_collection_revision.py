"""T04: confirmation freezes only explicitly chosen successful fixed versions."""

from pathlib import Path
import tempfile
import unittest

import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.db import open_database
from tests.test_document_preparation import text_pdf_bytes
from tests.test_ingestion_selection import saved_search


class CollectionRevisionTest(unittest.IsolatedAsyncioTestCase):
    async def _import_two_with_one_failure(self, engine, data_root, run_dir):
        from backend.app.collections.service import SelectedSource, create_collection, set_collection_selection
        from backend.app.ingestion.batch import import_collection

        collection = create_collection(engine)
        set_collection_selection(
            engine, collection.id, expected_revision=0, data_root=data_root,
            selections=(SelectedSource(run_dir, "2401.00001v1"), SelectedSource(run_dir, "2401.00001v3")),
        )
        def response(request):
            if request.url.path.endswith("v1"):
                return httpx.Response(200, headers={"Content-Type": "application/pdf"}, content=text_pdf_bytes())
            return httpx.Response(404, headers={"Content-Type": "text/html"}, content=b"missing")
        with httpx.Client(transport=httpx.MockTransport(response)) as client:
            report = await import_collection(
                engine, collection.id, expected_selection_revision=1, data_root=data_root, client=client
            )
        return collection, report

    async def test_partial_failure_requires_explicit_success_subset_and_freezes_it(self):
        from backend.app.collections.service import (
            SelectedSource,
            StaleSelection,
            confirm_collection_revision,
            get_collection_revision,
            set_collection_selection,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            engine = open_database(root / "papertrail.sqlite")
            try:
                collection, report = await self._import_two_with_one_failure(engine, data_root, run_dir)
                self.assertEqual(report.status, "partial_failed")
                successful_id = report.items[0].document_id
                with self.assertRaises(ValueError):
                    confirm_collection_revision(
                        engine, collection.id, expected_selection_revision=1,
                        import_run_id=report.id, document_ids=(), data_root=data_root,
                    )
                with self.assertRaises(ValueError):
                    confirm_collection_revision(
                        engine, collection.id, expected_selection_revision=1,
                        import_run_id=report.id, document_ids=(successful_id, "not-successful"),
                        data_root=data_root,
                    )
                revision = confirm_collection_revision(
                    engine, collection.id, expected_selection_revision=1,
                    import_run_id=report.id, document_ids=(successful_id,), data_root=data_root,
                )
                self.assertEqual(len(revision.members), 1)
                self.assertEqual(revision.members[0].document_id, successful_id)
                self.assertEqual(revision.members[0].versioned_id, "2401.00001v1")
                self.assertEqual(len(revision.snapshot_hash), 64)
                repeated = confirm_collection_revision(
                    engine, collection.id, expected_selection_revision=1,
                    import_run_id=report.id, document_ids=(successful_id,), data_root=data_root,
                )
                self.assertEqual(revision, repeated)

                set_collection_selection(
                    engine, collection.id, expected_revision=1, data_root=data_root,
                    selections=(SelectedSource(run_dir, "2401.00001v3"),),
                )
                self.assertEqual(get_collection_revision(engine, revision.id), revision)
                with self.assertRaises(StaleSelection):
                    confirm_collection_revision(
                        engine, collection.id, expected_selection_revision=1,
                        import_run_id=report.id, document_ids=(successful_id,), data_root=data_root,
                    )
            finally:
                engine.dispose()

    async def test_confirm_rejects_missing_blob_and_old_report_after_new_import(self):
        from backend.app.collections.service import (
            StaleSelection,
            confirm_collection_revision,
        )
        from backend.app.ingestion.batch import import_collection
        from backend.app.ingestion.service import InvalidDocumentBlob, get_document_version
        from backend.app.models import CollectionRevision

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            engine = open_database(root / "papertrail.sqlite")
            try:
                collection, first = await self._import_two_with_one_failure(engine, data_root, run_dir)
                with httpx.Client(transport=httpx.MockTransport(
                    lambda _request: httpx.Response(404, content=b"missing")
                )) as client:
                    second = await import_collection(
                        engine, collection.id, expected_selection_revision=1,
                        data_root=data_root, client=client,
                    )
                self.assertNotEqual(first.id, second.id)
                with self.assertRaises(StaleSelection):
                    confirm_collection_revision(
                        engine, collection.id, expected_selection_revision=1,
                        import_run_id=first.id, document_ids=(first.items[0].document_id,),
                        data_root=data_root,
                    )
                document = get_document_version(engine, second.items[0].document_id)
                (data_root / document.blob_ref).write_bytes(b"changed")
                with self.assertRaises(InvalidDocumentBlob):
                    confirm_collection_revision(
                        engine, collection.id, expected_selection_revision=1,
                        import_run_id=second.id, document_ids=(second.items[0].document_id,),
                        data_root=data_root,
                    )
                with Session(engine) as session:
                    self.assertEqual(session.scalar(select(func.count()).select_from(CollectionRevision)), 0)
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
