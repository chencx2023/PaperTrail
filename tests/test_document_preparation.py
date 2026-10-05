"""T04: a downloaded PDF can pass basic checks but fail PaperQA text preparation."""

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.db import open_database
from backend.app.ingestion.downloader import stage_pdf_download
from backend.app.ingestion.selection import select_fixed_version
from backend.app.ingestion.storage import publish_staged_pdf
from tests.test_ingestion_selection import saved_search
from tests.test_ingestion_storage import client_with_pdf, pdf_bytes


def text_pdf_bytes() -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})
    })
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 12 Tf 72 720 Td "
        b"(This paper studies memory for language agents with reliable evidence retrieval.) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def registered_document(root: Path, body: bytes):
    from backend.app.ingestion.service import register_downloaded_version

    data_root = root / "data"
    run_dir = saved_search(data_root)
    source = select_fixed_version(run_dir, "2401.00001v3")
    with client_with_pdf(body) as client:
        with stage_pdf_download(source, client=client, staging_dir=data_root / "staging") as staged:
            stored = publish_staged_pdf(staged, store_dir=data_root / "blobs")
    engine = open_database(root / "papertrail.sqlite")
    document = register_downloaded_version(
        engine, run_dir=run_dir, versioned_id=source.versioned_id,
        stored=stored, data_root=data_root,
    )
    return engine, data_root, document, stored


class DocumentPreparationTest(unittest.IsolatedAsyncioTestCase):
    async def test_text_pdf_is_prepared_once_for_a_named_parser_configuration(self):
        from backend.app.ingestion.preparation import (
            get_document_preparation,
            prepare_document_version,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine, data_root, document, _stored = registered_document(root, text_pdf_bytes())
            first = await prepare_document_version(engine, document.id, data_root=data_root)
            repeated = await prepare_document_version(engine, document.id, data_root=data_root)
            self.assertEqual(first, repeated)
            self.assertEqual(first.status, "succeeded")
            self.assertGreater(first.chunk_count, 0)
            self.assertEqual(len(first.parser_config_hash), 64)
            self.assertIn('"chunk_chars":3000', first.parser_config_json)
            self.assertIn("paperqa_pypdf", first.parser_config_json)
            engine.dispose()

            reopened = open_database(root / "papertrail.sqlite")
            self.assertEqual(
                get_document_preparation(reopened, document.id, first.parser_config_hash), first
            )
            reopened.dispose()

    async def test_blank_pdf_records_preparation_failure_but_remains_downloaded(self):
        from backend.app.ingestion.preparation import prepare_document_version
        from backend.app.ingestion.service import get_document_version

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine, data_root, document, _stored = registered_document(root, pdf_bytes())
            result = await prepare_document_version(engine, document.id, data_root=data_root)
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.error_kind, "UnusableDocumentText")
            self.assertIsNone(result.chunk_count)
            self.assertEqual(get_document_version(engine, document.id).download_status, "downloaded")
            engine.dispose()

    async def test_configuration_changes_identity_and_corrupt_blob_cannot_reuse_success(self):
        from backend.app.ingestion.preparation import (
            TextPreparationConfig,
            prepare_document_version,
        )
        from backend.app.ingestion.service import InvalidDocumentBlob
        from backend.app.models import DocumentPreparation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine, data_root, document, stored = registered_document(root, text_pdf_bytes())
            first = await prepare_document_version(engine, document.id, data_root=data_root)
            changed = await prepare_document_version(
                engine, document.id, data_root=data_root,
                config=TextPreparationConfig(chunk_chars=1000, overlap=100),
            )
            self.assertEqual(changed.status, "succeeded")
            self.assertNotEqual(first.parser_config_hash, changed.parser_config_hash)
            with Session(engine) as session:
                self.assertEqual(
                    session.scalar(select(func.count()).select_from(DocumentPreparation)), 2
                )

            stored.path.write_bytes(b"corrupted")
            with self.assertRaises(InvalidDocumentBlob):
                await prepare_document_version(engine, document.id, data_root=data_root)
            engine.dispose()

    async def test_blob_changed_during_extraction_creates_no_preparation_record(self):
        from backend.app.ingestion import preparation
        from backend.app.ingestion.service import InvalidDocumentBlob
        from backend.app.models import DocumentPreparation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine, data_root, document, stored = registered_document(root, text_pdf_bytes())
            try:
                original_read_doc = preparation.read_doc

                async def change_after_read(*args, **kwargs):
                    texts = await original_read_doc(*args, **kwargs)
                    stored.path.write_bytes(b"corrupted after read")
                    return texts

                with patch.object(preparation, "read_doc", side_effect=change_after_read):
                    with self.assertRaises(InvalidDocumentBlob):
                        await preparation.prepare_document_version(
                            engine, document.id, data_root=data_root
                        )
                with Session(engine) as session:
                    self.assertEqual(
                        session.scalar(select(func.count()).select_from(DocumentPreparation)), 0
                    )
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
