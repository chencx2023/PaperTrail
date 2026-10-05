"""T04: a chosen search result must identify one saved PDF version."""

from pathlib import Path
import tempfile
import unittest

from backend.app.search.client import ArxivClient, TransportResponse
from backend.app.search.service import run_search


FIXTURE = Path(__file__).parent / "fixtures" / "arxiv" / "normal_with_versions.xml"


class FixedTransport:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def send(self, _request):
        return TransportResponse(
            status_code=200,
            body=self.body,
            headers=(("Content-Type", "application/atom+xml"),),
        )


def saved_search(root: Path, body: bytes | None = None) -> Path:
    result = run_search(
        "LLM agent memory",
        client=ArxivClient(
            transport=FixedTransport(
                FIXTURE.read_bytes() if body is None else body
            )
        ),
        output_root=root / "runs",
        run_id="search-selection-fixture",
    )
    assert result.status == "success"
    return result.run_dir


class FixedVersionSelectionTest(unittest.TestCase):
    def test_explicit_saved_version_is_not_silently_upgraded(self):
        from backend.app.ingestion.selection import select_fixed_version

        with tempfile.TemporaryDirectory() as directory:
            run_dir = saved_search(Path(directory))
            first = select_fixed_version(run_dir, "2401.00001v1")
            current = select_fixed_version(run_dir, "2401.00001v3")
            legacy = select_fixed_version(run_dir, "cs/9901001v2")

        self.assertEqual(first.version, 1)
        self.assertEqual(first.title, "Memory for Agents: First Version")
        self.assertEqual(first.download_url, "https://arxiv.org/pdf/2401.00001v1")
        self.assertEqual(current.version, 3)
        self.assertEqual(current.source_run_id, "search-selection-fixture")
        self.assertEqual(current.arxiv_id, "2401.00001")
        self.assertEqual(legacy.source_pdf_url, "http://arxiv.org/pdf/cs/9901001v2")
        self.assertEqual(legacy.download_url, "https://arxiv.org/pdf/cs/9901001v2")

    def test_unknown_or_unversioned_selection_is_rejected(self):
        from backend.app.ingestion.selection import (
            InvalidDocumentSelection,
            select_fixed_version,
        )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = saved_search(Path(directory))
            for versioned_id in ("2401.00001v2", "2401.00001"):
                with self.subTest(versioned_id=versioned_id):
                    with self.assertRaises(InvalidDocumentSelection):
                        select_fixed_version(run_dir, versioned_id)

    def test_changed_atom_or_candidate_list_is_rejected(self):
        from backend.app.ingestion.selection import (
            InvalidDocumentSelection,
            select_fixed_version,
        )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = saved_search(Path(directory))
            candidates_path = run_dir / "candidates.json"
            saved_candidates = candidates_path.read_bytes()
            candidates_path.write_text("[]", encoding="utf-8")
            with self.assertRaises(InvalidDocumentSelection):
                select_fixed_version(run_dir, "2401.00001v3")

            candidates_path.write_bytes(saved_candidates)
            (run_dir / "response.atom").write_bytes(b"tampered")
            with self.assertRaises(InvalidDocumentSelection):
                select_fixed_version(run_dir, "2401.00001v3")

    def test_pdf_link_outside_the_selected_arxiv_version_is_rejected(self):
        from backend.app.ingestion.selection import (
            InvalidDocumentSelection,
            select_fixed_version,
        )

        original = FIXTURE.read_bytes()
        wrong_host = original.replace(
            b"https://arxiv.org/pdf/2401.00001v3",
            b"https://evil.example/pdf/2401.00001v3",
        )
        wrong_version = original.replace(
            b"https://arxiv.org/pdf/2401.00001v3",
            b"https://arxiv.org/pdf/2401.00001v1",
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, body in enumerate((wrong_host, wrong_version)):
                with self.subTest(index=index):
                    run_dir = saved_search(Path(directory) / str(index), body)
                    with self.assertRaises(InvalidDocumentSelection):
                        select_fixed_version(run_dir, "2401.00001v3")


if __name__ == "__main__":
    unittest.main()
