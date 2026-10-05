"""T04: parse before publishing immutable, content-addressed PDF bytes."""

from io import BytesIO
import hashlib
from pathlib import Path
import tempfile
import unittest

import httpx
from pypdf import PdfWriter

from backend.app.ingestion.downloader import stage_pdf_download
from tests.test_ingestion_downloader import fixed_source


def pdf_bytes(page_count: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=72, height=72)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def client_with_pdf(body: bytes) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"Content-Type": "application/pdf"},
                content=body,
            )
        )
    )


class PdfPublicationTest(unittest.TestCase):
    def test_valid_pdf_publishes_once_and_reuses_the_same_hash(self):
        from backend.app.ingestion.storage import publish_staged_pdf

        body = pdf_bytes()
        expected_hash = hashlib.sha256(body).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixed_source(root)
            stored = []
            for number in range(2):
                with client_with_pdf(body) as client:
                    with stage_pdf_download(
                        source,
                        client=client,
                        staging_dir=root / f"stage-{number}",
                    ) as staged:
                        stored.append(
                            publish_staged_pdf(staged, store_dir=root / "blobs")
                        )
                    self.assertFalse(staged.path.exists())

            self.assertEqual(stored[0].sha256, expected_hash)
            self.assertEqual(stored[0].path.name, f"{expected_hash}.pdf")
            self.assertEqual(stored[0].path.read_bytes(), body)
            self.assertEqual(stored[0].page_count, 1)
            self.assertFalse(stored[0].reused)
            self.assertTrue(stored[1].reused)
            self.assertEqual(stored[1].path, stored[0].path)
            self.assertEqual(list((root / "blobs").glob("*.pdf")), [stored[0].path])

    def test_pdf_header_without_parseable_pages_never_publishes(self):
        from backend.app.ingestion.storage import InvalidPdfContent, publish_staged_pdf

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixed_source(root)
            for index, body in enumerate((b"%PDF-1.4\nnot a PDF", pdf_bytes(0))):
                with self.subTest(index=index):
                    with client_with_pdf(body) as client:
                        with stage_pdf_download(
                            source,
                            client=client,
                            staging_dir=root / f"stage-{index}",
                        ) as staged:
                            with self.assertRaises(InvalidPdfContent):
                                publish_staged_pdf(staged, store_dir=root / "blobs")
                    self.assertEqual(list((root / "blobs").glob("*.pdf")), [])

    def test_changed_staged_bytes_cannot_be_published_under_old_hash(self):
        from backend.app.ingestion.storage import InvalidPdfContent, publish_staged_pdf

        body = pdf_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixed_source(root)
            with client_with_pdf(body) as client:
                with stage_pdf_download(
                    source, client=client, staging_dir=root / "stage"
                ) as staged:
                    staged.path.write_bytes(body + b"changed")
                    with self.assertRaises(InvalidPdfContent) as raised:
                        publish_staged_pdf(staged, store_dir=root / "blobs")
            self.assertEqual(raised.exception.kind, "staged_file_changed")
            self.assertEqual(list((root / "blobs").glob("*.pdf")), [])

    def test_corrupted_cached_file_is_not_silently_overwritten(self):
        from backend.app.ingestion.storage import CorruptStoredPdf, publish_staged_pdf

        body = pdf_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixed_source(root)
            with client_with_pdf(body) as client:
                with stage_pdf_download(
                    source, client=client, staging_dir=root / "first"
                ) as staged:
                    first = publish_staged_pdf(staged, store_dir=root / "blobs")
            first.path.write_bytes(b"corrupted after publication")

            with client_with_pdf(body) as client:
                with stage_pdf_download(
                    source, client=client, staging_dir=root / "second"
                ) as staged:
                    with self.assertRaises(CorruptStoredPdf):
                        publish_staged_pdf(staged, store_dir=root / "blobs")
            self.assertEqual(first.path.read_bytes(), b"corrupted after publication")
            self.assertEqual(list((root / "blobs").glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
