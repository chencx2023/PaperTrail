"""T04: selected saved-search versions keep separate import results."""

from pathlib import Path
import tempfile
import unittest

import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.db import open_database
from backend.app.models import DocumentVersion
from tests.test_document_preparation import text_pdf_bytes
from tests.test_ingestion_selection import saved_search
from tests.test_ingestion_storage import pdf_bytes


class CollectionImportTest(unittest.IsolatedAsyncioTestCase):
    async def test_partial_download_failure_is_recorded_and_success_is_reused(self):
        from backend.app.collections.service import (
            SelectedSource,
            create_collection,
            set_collection_selection,
        )
        from backend.app.ingestion.batch import import_collection

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            engine = open_database(root / "papertrail.sqlite")
            try:
                collection = create_collection(engine)
                selected = set_collection_selection(
                    engine, collection.id, expected_revision=0, data_root=data_root,
                    selections=(
                        SelectedSource(run_dir, "2401.00001v1"),
                        SelectedSource(run_dir, "2401.00001v3"),
                    ),
                )
                self.assertEqual(selected.selection_revision, 1)
                requests = []

                def first_response(request):
                    requests.append(str(request.url))
                    if request.url.path.endswith("v1"):
                        return httpx.Response(200, headers={"Content-Type": "application/pdf"}, content=text_pdf_bytes())
                    return httpx.Response(200, headers={"Content-Type": "text/html"}, content=b"<html>not a PDF</html>")

                with httpx.Client(transport=httpx.MockTransport(first_response)) as client:
                    first = await import_collection(
                        engine, collection.id, expected_selection_revision=1,
                        data_root=data_root, client=client,
                    )
                self.assertEqual(first.status, "partial_failed")
                self.assertEqual([item.status for item in first.items], ["succeeded", "download_failed"])
                self.assertIsNotNone(first.items[0].document_id)
                self.assertIsNotNone(first.items[0].preparation_id)
                self.assertIsNone(first.items[1].document_id)
                self.assertEqual(len(requests), 2)

                def second_response(request):
                    self.assertTrue(request.url.path.endswith("v3"), "v1 should reuse its saved PDF")
                    return httpx.Response(200, headers={"Content-Type": "application/pdf"}, content=text_pdf_bytes())

                with httpx.Client(transport=httpx.MockTransport(second_response)) as client:
                    second = await import_collection(
                        engine, collection.id, expected_selection_revision=1,
                        data_root=data_root, client=client,
                    )
                self.assertEqual(second.status, "succeeded")
                self.assertEqual([item.status for item in second.items], ["succeeded", "succeeded"])
                self.assertTrue(second.items[0].reused_pdf)
                self.assertEqual(second.items[0].document_id, first.items[0].document_id)
                with Session(engine) as session:
                    self.assertEqual(session.scalar(select(func.count()).select_from(DocumentVersion)), 2)
            finally:
                engine.dispose()

    async def test_blank_pdf_is_downloaded_but_preparation_failure_is_kept(self):
        from backend.app.collections.service import SelectedSource, create_collection, set_collection_selection
        from backend.app.ingestion.batch import import_collection
        from backend.app.ingestion.service import get_document_version

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            engine = open_database(root / "papertrail.sqlite")
            try:
                collection = create_collection(engine)
                set_collection_selection(
                    engine, collection.id, expected_revision=0, data_root=data_root,
                    selections=(SelectedSource(run_dir, "2401.00001v3"),),
                )
                with httpx.Client(transport=httpx.MockTransport(
                    lambda _request: httpx.Response(200, headers={"Content-Type": "application/pdf"}, content=pdf_bytes())
                )) as client:
                    report = await import_collection(
                        engine, collection.id, expected_selection_revision=1,
                        data_root=data_root, client=client,
                    )
                self.assertEqual(report.status, "failed")
                self.assertEqual(report.items[0].status, "preparation_failed")
                self.assertEqual(get_document_version(engine, report.items[0].document_id).download_status, "downloaded")
                self.assertIsNotNone(report.items[0].preparation_id)
            finally:
                engine.dispose()

    async def test_selection_rejects_tampered_snapshot_and_stale_revision_without_request(self):
        from backend.app.collections.service import (
            SelectedSource,
            StaleSelection,
            create_collection,
            get_collection,
            set_collection_selection,
        )
        from backend.app.ingestion.batch import import_collection
        from backend.app.ingestion.selection import InvalidDocumentSelection

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            engine = open_database(root / "papertrail.sqlite")
            try:
                collection = create_collection(engine)
                choice = (SelectedSource(run_dir, "2401.00001v1"),)
                set_collection_selection(engine, collection.id, expected_revision=0, data_root=data_root, selections=choice)
                with self.assertRaises(StaleSelection):
                    set_collection_selection(engine, collection.id, expected_revision=0, data_root=data_root, selections=choice)
                self.assertEqual(get_collection(engine, collection.id).selection_revision, 1)

                (run_dir / "candidates.json").write_text("[]", encoding="utf-8")
                with self.assertRaises(InvalidDocumentSelection):
                    set_collection_selection(engine, collection.id, expected_revision=1, data_root=data_root, selections=choice)
                with httpx.Client(transport=httpx.MockTransport(
                    lambda _request: self.fail("tampered snapshot must never cause a PDF request")
                )) as client:
                    report = await import_collection(
                        engine, collection.id, expected_selection_revision=1,
                        data_root=data_root, client=client,
                    )
                self.assertEqual(report.status, "failed")
                self.assertEqual(report.items[0].status, "selection_failed")
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
