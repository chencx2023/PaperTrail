"""T04: downloading only stages bounded PDF-looking bytes for later parsing."""

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import time
import unittest

import httpx

from backend.app.ingestion.selection import select_fixed_version
from tests.test_ingestion_selection import saved_search


PDF_BYTES = b"%PDF-1.4\nsmall offline fixture\n%%EOF\n"


def fixed_source(root: Path):
    return select_fixed_version(saved_search(root), "2401.00001v3")


class StagedPdfDownloadTest(unittest.TestCase):
    def test_success_is_only_a_temporary_file_with_digest(self):
        from backend.app.ingestion.downloader import stage_pdf_download

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixed_source(root)
            calls = []

            def respond(request: httpx.Request) -> httpx.Response:
                calls.append(str(request.url))
                return httpx.Response(
                    200,
                    headers={"Content-Type": "application/pdf"},
                    content=PDF_BYTES,
                )

            with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                with stage_pdf_download(
                    source, client=client, staging_dir=root / "staging"
                ) as staged:
                    self.assertEqual(staged.path.read_bytes(), PDF_BYTES)
                    self.assertEqual(staged.byte_count, len(PDF_BYTES))
                    self.assertEqual(staged.sha256, hashlib.sha256(PDF_BYTES).hexdigest())
                    self.assertEqual(staged.versioned_id, "2401.00001v3")
                    self.assertEqual(staged.path.suffix, ".part")
                self.assertFalse(staged.path.exists())
                self.assertEqual(list((root / "staging").iterdir()), [])
            self.assertEqual(calls, ["https://arxiv.org/pdf/2401.00001v3"])

    def test_http_error_and_redirect_do_not_create_a_staged_file(self):
        from backend.app.ingestion.downloader import PdfStageError, stage_pdf_download

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixed_source(root)
            for status, expected_kind in ((404, "http_error"), (302, "redirect")):
                with self.subTest(status=status):
                    calls = []

                    def respond(request: httpx.Request) -> httpx.Response:
                        calls.append(str(request.url))
                        return httpx.Response(
                            status,
                            headers={"Location": "https://evil.example/file.pdf"},
                            content=b"not a PDF",
                        )

                    with httpx.Client(
                        transport=httpx.MockTransport(respond),
                        follow_redirects=True,
                    ) as client:
                        with self.assertRaises(PdfStageError) as raised:
                            with stage_pdf_download(
                                source, client=client, staging_dir=root / "staging"
                            ):
                                self.fail("An invalid response cannot be staged.")
                    self.assertEqual(raised.exception.kind, expected_kind)
                    self.assertEqual(raised.exception.status_code, status)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(list((root / "staging").iterdir()), [])

    def test_same_version_pdf_redirect_is_followed_once(self):
        from backend.app.ingestion.downloader import stage_pdf_download

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixed_source(root)
            canonical_url = "https://arxiv.org/pdf/2401.00001v3.pdf"
            for location in ("2401.00001v3.pdf", canonical_url):
                with self.subTest(location=location):
                    calls = []

                    def respond(request: httpx.Request) -> httpx.Response:
                        calls.append(str(request.url))
                        if len(calls) == 1:
                            return httpx.Response(302, headers={"Location": location})
                        return httpx.Response(
                            200,
                            headers={"Content-Type": "application/pdf"},
                            content=PDF_BYTES,
                        )

                    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                        with stage_pdf_download(
                            source, client=client, staging_dir=root / "staging"
                        ) as staged:
                            self.assertEqual(staged.path.read_bytes(), PDF_BYTES)
                        self.assertFalse(staged.path.exists())
                    self.assertEqual(calls, [source.download_url, canonical_url])

    def test_unsafe_redirect_targets_never_receive_a_second_request(self):
        from backend.app.ingestion.downloader import PdfStageError, stage_pdf_download

        targets = (
            "https://evil.example/pdf/2401.00001v3.pdf",
            "http://arxiv.org/pdf/2401.00001v3.pdf",
            "https://arxiv.org/pdf/2401.00001v4.pdf",
            "https://arxiv.org/abs/2401.00001v3",
            "https://arxiv.org/pdf/2401.00001v3.pdf?download=1",
            "https://arxiv.org:443/pdf/2401.00001v3.pdf",
            "//evil.example/pdf/2401.00001v3.pdf",
            None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixed_source(root)
            for target in targets:
                with self.subTest(target=target):
                    calls = []

                    def respond(request: httpx.Request) -> httpx.Response:
                        calls.append(str(request.url))
                        headers = {} if target is None else {"Location": target}
                        return httpx.Response(302, headers=headers)

                    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                        with self.assertRaises(PdfStageError) as raised:
                            with stage_pdf_download(
                                source, client=client, staging_dir=root / "staging"
                            ):
                                self.fail("An unsafe redirect cannot be staged.")
                    self.assertEqual(raised.exception.kind, "redirect")
                    self.assertEqual(calls, [source.download_url])
                    self.assertEqual(list((root / "staging").iterdir()), [])

    def test_redirect_loop_stops_before_repeating_a_request(self):
        from backend.app.ingestion.downloader import PdfStageError, stage_pdf_download

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixed_source(root)
            calls = []

            def respond(request: httpx.Request) -> httpx.Response:
                calls.append(str(request.url))
                if len(calls) == 1:
                    return httpx.Response(
                        307, headers={"Location": "2401.00001v3.pdf"}
                    )
                return httpx.Response(307, headers={"Location": source.download_url})

            with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                with self.assertRaises(PdfStageError) as raised:
                    with stage_pdf_download(
                        source, client=client, staging_dir=root / "staging"
                    ):
                        self.fail("A redirect loop cannot be staged.")
            self.assertEqual(raised.exception.kind, "redirect")
            self.assertEqual(
                calls,
                [source.download_url, "https://arxiv.org/pdf/2401.00001v3.pdf"],
            )
            self.assertEqual(list((root / "staging").iterdir()), [])

    def test_redirect_does_not_restart_the_download_time_limit(self):
        from backend.app.ingestion.downloader import PdfStageError, stage_pdf_download

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixed_source(root)
            calls = []

            def respond(request: httpx.Request) -> httpx.Response:
                calls.append(str(request.url))
                time.sleep(0.03)
                return httpx.Response(
                    302, headers={"Location": "2401.00001v3.pdf"}
                )

            with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                with self.assertRaises(PdfStageError) as raised:
                    with stage_pdf_download(
                        source,
                        client=client,
                        staging_dir=root / "staging",
                        timeout_seconds=0.005,
                    ):
                        self.fail("An expired download cannot follow another hop.")
            self.assertEqual(raised.exception.kind, "timeout")
            self.assertEqual(calls, [source.download_url])
            self.assertEqual(list((root / "staging").iterdir()), [])

    def test_html_response_and_pdf_label_with_html_body_are_rejected(self):
        from backend.app.ingestion.downloader import PdfStageError, stage_pdf_download

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixed_source(root)
            cases = (
                ("text/html", "invalid_content_type"),
                ("application/pdf", "invalid_signature"),
            )
            for content_type, expected_kind in cases:
                with self.subTest(content_type=content_type):
                    client = httpx.Client(
                        transport=httpx.MockTransport(
                            lambda _request: httpx.Response(
                                200,
                                headers={"Content-Type": content_type},
                                content=b"<html>error</html>",
                            )
                        )
                    )
                    try:
                        with self.assertRaises(PdfStageError) as raised:
                            with stage_pdf_download(
                                source, client=client, staging_dir=root / "staging"
                            ):
                                self.fail("HTML cannot be staged as a PDF.")
                    finally:
                        client.close()
                    self.assertEqual(raised.exception.kind, expected_kind)
                    self.assertEqual(list((root / "staging").iterdir()), [])

    def test_stream_limit_rejects_more_bytes_than_declared(self):
        from backend.app.ingestion.downloader import PdfStageError, stage_pdf_download

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixed_source(root)
            with httpx.Client(
                transport=httpx.MockTransport(
                    lambda _request: httpx.Response(
                        200,
                        headers={
                            "Content-Type": "application/pdf",
                            "Content-Length": "5",
                        },
                        content=PDF_BYTES,
                    )
                )
            ) as client:
                with self.assertRaises(PdfStageError) as raised:
                    with stage_pdf_download(
                        source,
                        client=client,
                        staging_dir=root / "staging",
                        max_bytes=10,
                    ):
                        self.fail("An oversized PDF cannot be staged.")
            self.assertEqual(raised.exception.kind, "too_large")
            self.assertEqual(list((root / "staging").iterdir()), [])

    def test_forged_target_is_rejected_before_a_request(self):
        from backend.app.ingestion.downloader import PdfStageError, stage_pdf_download

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = replace(
                fixed_source(root),
                download_url="https://evil.example/pdf/2401.00001v3",
            )
            calls = []

            def respond(_request: httpx.Request) -> httpx.Response:
                calls.append("sent")
                return httpx.Response(200, content=PDF_BYTES)

            with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                with self.assertRaises(PdfStageError) as raised:
                    with stage_pdf_download(
                        source, client=client, staging_dir=root / "staging"
                    ):
                        self.fail("A forged target cannot be sent.")
            self.assertEqual(raised.exception.kind, "invalid_source")
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
