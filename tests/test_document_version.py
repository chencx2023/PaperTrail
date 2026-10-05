"""T04: persist a fixed downloaded version without claiming index readiness."""

from pathlib import Path
import tempfile
import unittest

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.db import open_database
from backend.app.ingestion.downloader import stage_pdf_download
from backend.app.ingestion.selection import InvalidDocumentSelection, select_fixed_version
from backend.app.ingestion.storage import publish_staged_pdf
from tests.test_ingestion_selection import saved_search
from tests.test_ingestion_storage import client_with_pdf, pdf_bytes


def publish_fixture(root: Path, run_dir: Path, versioned_id: str, pages: int):
    source = select_fixed_version(run_dir, versioned_id)
    with client_with_pdf(pdf_bytes(pages)) as client:
        with stage_pdf_download(
            source, client=client, staging_dir=root / "staging"
        ) as staged:
            return publish_staged_pdf(staged, store_dir=root / "blobs")


class DocumentVersionPersistenceTest(unittest.TestCase):
    def test_downloaded_version_survives_reopen_with_snapshot_and_relative_blob_ref(self):
        from backend.app.ingestion.service import (
            get_document_version,
            register_downloaded_version,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            stored = publish_fixture(data_root, run_dir, "2401.00001v3", 1)
            database_path = root / "papertrail.sqlite"
            engine = open_database(database_path)
            first = register_downloaded_version(
                engine,
                run_dir=run_dir,
                versioned_id="2401.00001v3",
                stored=stored,
                data_root=data_root,
            )
            repeated = register_downloaded_version(
                engine,
                run_dir=run_dir,
                versioned_id="2401.00001v3",
                stored=stored,
                data_root=data_root,
            )
            self.assertEqual(first, repeated)
            engine.dispose()

            reopened = open_database(database_path)
            record = get_document_version(reopened, first.id)
            self.assertEqual(record.paper_id, first.paper_id)
            self.assertEqual(record.versioned_id, "2401.00001v3")
            self.assertEqual(record.source_run_id, "search-selection-fixture")
            self.assertEqual(record.snapshot_sha256, select_fixed_version(run_dir, "2401.00001v3").snapshot_sha256)
            self.assertEqual(record.sha256, stored.sha256)
            self.assertEqual(record.blob_ref, f"blobs/{stored.sha256}.pdf")
            self.assertEqual(record.download_status, "downloaded")
            self.assertEqual(record.page_count, 1)
            self.assertIsNone(get_document_version(reopened, "missing"))
            reopened.dispose()

    def test_same_version_with_different_bytes_cannot_replace_existing_record(self):
        from backend.app.ingestion.service import (
            VersionContentConflict,
            get_document_version,
            register_downloaded_version,
        )
        from backend.app.models import DocumentVersion

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            first_pdf = publish_fixture(data_root, run_dir, "2401.00001v3", 1)
            second_pdf = publish_fixture(data_root, run_dir, "2401.00001v3", 2)
            engine = open_database(root / "papertrail.sqlite")
            first = register_downloaded_version(
                engine,
                run_dir=run_dir,
                versioned_id="2401.00001v3",
                stored=first_pdf,
                data_root=data_root,
            )
            with self.assertRaises(VersionContentConflict):
                register_downloaded_version(
                    engine,
                    run_dir=run_dir,
                    versioned_id="2401.00001v3",
                    stored=second_pdf,
                    data_root=data_root,
                )
            with Session(engine) as session:
                self.assertEqual(
                    session.scalar(select(func.count()).select_from(DocumentVersion)), 1
                )
            self.assertEqual(get_document_version(engine, first.id).sha256, first_pdf.sha256)
            engine.dispose()

    def test_two_versions_with_same_bytes_remain_distinct_under_one_paper(self):
        from backend.app.ingestion.service import (
            get_document_version,
            register_downloaded_version,
        )
        from backend.app.models import DocumentVersion, Paper

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            first_pdf = publish_fixture(data_root, run_dir, "2401.00001v1", 1)
            later_pdf = publish_fixture(data_root, run_dir, "2401.00001v3", 1)
            engine = open_database(root / "papertrail.sqlite")
            try:
                first = register_downloaded_version(
                    engine,
                    run_dir=run_dir,
                    versioned_id="2401.00001v1",
                    stored=first_pdf,
                    data_root=data_root,
                )
                later = register_downloaded_version(
                    engine,
                    run_dir=run_dir,
                    versioned_id="2401.00001v3",
                    stored=later_pdf,
                    data_root=data_root,
                )
                self.assertEqual(first.sha256, later.sha256)
                self.assertNotEqual(first.id, later.id)
                self.assertEqual(first.paper_id, later.paper_id)
                self.assertEqual(
                    get_document_version(engine, first.id).title,
                    "Memory for Agents: First Version",
                )
                self.assertEqual(
                    get_document_version(engine, later.id).title,
                    "Memory for Agents: Revised Version",
                )
                self.assertEqual(len(get_document_version(engine, first.id).authors), 2)
                self.assertEqual(len(get_document_version(engine, later.id).authors), 3)
                with Session(engine) as session:
                    self.assertEqual(session.scalar(select(func.count()).select_from(Paper)), 1)
                    self.assertEqual(
                        session.scalar(select(func.count()).select_from(DocumentVersion)), 2
                    )
            finally:
                engine.dispose()

    def test_corrupt_or_outside_blob_and_changed_snapshot_create_no_record(self):
        from backend.app.ingestion.service import (
            InvalidDocumentBlob,
            register_downloaded_version,
        )
        from backend.app.models import DocumentVersion, Paper

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            outside = publish_fixture(root / "outside", run_dir, "2401.00001v3", 1)
            inside = publish_fixture(data_root, run_dir, "2401.00001v3", 1)
            engine = open_database(root / "papertrail.sqlite")

            with self.assertRaises(InvalidDocumentBlob):
                register_downloaded_version(
                    engine,
                    run_dir=run_dir,
                    versioned_id="2401.00001v3",
                    stored=outside,
                    data_root=data_root,
                )
            inside.path.write_bytes(b"changed")
            with self.assertRaises(InvalidDocumentBlob):
                register_downloaded_version(
                    engine,
                    run_dir=run_dir,
                    versioned_id="2401.00001v3",
                    stored=inside,
                    data_root=data_root,
                )
            (run_dir / "candidates.json").write_text("[]", encoding="utf-8")
            with self.assertRaises(InvalidDocumentSelection):
                register_downloaded_version(
                    engine,
                    run_dir=run_dir,
                    versioned_id="2401.00001v3",
                    stored=inside,
                    data_root=data_root,
                )
            with Session(engine) as session:
                self.assertEqual(session.scalar(select(func.count()).select_from(Paper)), 0)
                self.assertEqual(
                    session.scalar(select(func.count()).select_from(DocumentVersion)), 0
                )
            engine.dispose()

    def test_pdf_staged_for_another_version_cannot_be_registered(self):
        from backend.app.ingestion.service import (
            InvalidDocumentBlob,
            register_downloaded_version,
        )
        from backend.app.models import DocumentVersion

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            first_version_pdf = publish_fixture(
                data_root, run_dir, "2401.00001v1", 1
            )
            engine = open_database(root / "papertrail.sqlite")
            try:
                with self.assertRaises(InvalidDocumentBlob):
                    register_downloaded_version(
                        engine,
                        run_dir=run_dir,
                        versioned_id="2401.00001v3",
                        stored=first_version_pdf,
                        data_root=data_root,
                    )
                with Session(engine) as session:
                    self.assertEqual(
                        session.scalar(select(func.count()).select_from(DocumentVersion)),
                        0,
                    )
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
