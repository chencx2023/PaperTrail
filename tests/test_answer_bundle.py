import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.pqa.adapter import AnswerBundle, PaperQAAdapter


class AnswerBundleTest(unittest.TestCase):
    def setUp(self):
        self.report = {"status": "success", "documents": [
            {"filename": "A.pdf", "sha256": "sha-a", "dockey": "upstream-a"}],
            "token_counts": {"model": [10, 5]}, "models": {"llm": "model"}}
        self.session = {"question": "Q", "raw_answer": "Fact (pqac-11111111, pqac-missing). Again (pqac-11111111).",
                        "formatted_answer": "Fact (A page 1).", "has_successful_answer": True,
                        "context": "Final input", "contexts": [{"id": "pqac-11111111", "score": 9,
                        "context": "Summary", "text": {"text": "Original", "name": "A pages 1-2",
                        "doc": {"dockey": "upstream-a"}}}]}

    def test_mapping_keeps_invalid_ids_and_roundtrips_without_models(self):
        bundle = PaperQAAdapter.to_answer_bundle("run-a", self.report, self.session)
        self.assertEqual([c.mapping_status for c in bundle.citations], ["mapped", "missing_context", "mapped"])
        self.assertEqual(bundle.evidence[0].document_id, "sha-a")
        self.assertEqual(bundle.evidence[0].original_text, "Original")
        self.assertEqual(bundle.evidence[0].locator_status, "unverified_upstream_label")
        self.assertIsNone(bundle.evidence[0].pdf_page)
        self.assertEqual(bundle.usage["coverage"], "partial")
        self.assertIsNone(bundle.usage["requests_by_role"])
        self.assertEqual(AnswerBundle.model_validate_json(bundle.model_dump_json()), bundle)
        self.assertEqual(bundle.raw_answer, self.session["raw_answer"])

    def test_chinese_adjacent_citations_keep_offsets_without_ascii_false_positive(self):
        self.session["raw_answer"] = "结论pqac-11111111；未知pqac-missing；notpqac-11111111"
        bundle = PaperQAAdapter.to_answer_bundle("run-a", self.report, self.session)

        self.assertEqual(
            [citation.upstream_context_id for citation in bundle.citations],
            ["pqac-11111111", "pqac-missing"],
        )
        self.assertEqual(
            [citation.mapping_status for citation in bundle.citations],
            ["mapped", "missing_context"],
        )
        self.assertEqual(bundle.citations[0].start, len("结论"))
        self.assertEqual(
            [bundle.raw_answer[citation.start:citation.end] for citation in bundle.citations],
            ["pqac-11111111", "pqac-missing"],
        )
        self.assertEqual(bundle.raw_answer, self.session["raw_answer"])

    def test_foreign_evidence_fails_scope_even_when_not_cited(self):
        foreign = copy.deepcopy(self.session["contexts"][0])
        foreign["id"] = "pqac-22222222"
        foreign["text"]["doc"]["dockey"] = "outside"
        self.session["contexts"].append(foreign)
        bundle = PaperQAAdapter.to_answer_bundle("run-a", self.report, self.session)
        self.assertEqual(bundle.job_status, "failed")
        self.assertEqual(bundle.answer_outcome, "scope_violation")

    def test_status_does_not_treat_any_text_as_success(self):
        for status, job, outcome in [("truncated", "succeeded", "truncated"),
                                     ("unsure", "succeeded", "insufficient"),
                                     ("fail", "failed", "unknown"),
                                     ("interrupted", "interrupted", "unknown")]:
            with self.subTest(status=status):
                self.report["status"] = status
                b = PaperQAAdapter.to_answer_bundle("run-a", self.report, self.session)
                self.assertEqual((b.job_status, b.answer_outcome), (job, outcome))

    def test_evidence_ids_are_run_scoped_and_ambiguous_sources_rejected(self):
        a = PaperQAAdapter.to_answer_bundle("a", self.report, self.session)
        b = PaperQAAdapter.to_answer_bundle("b", self.report, self.session)
        self.assertNotEqual(a.evidence[0].evidence_id, b.evidence[0].evidence_id)
        self.report["documents"].append({"filename": "B.pdf", "sha256": "sha-b", "dockey": "upstream-a"})
        with self.assertRaises(ValueError):
            PaperQAAdapter.to_answer_bundle("a", self.report, self.session)
