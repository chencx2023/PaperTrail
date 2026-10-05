"""T05: map raw PaperQA references to the confirmed document version."""

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from backend.app.collections.indexing import build_collection_index
from backend.app.db import open_database
from backend.app.pqa.adapter import AnswerBundle
from backend.app.qa.scope import prepare_answer_scope
from tests.test_collection_index import FakeEmbedding, imported_revision, settings
from tests.test_ingestion_selection import saved_search
from tests.test_qa_execution import offline_answer


class EvidenceMappingTest(unittest.IsolatedAsyncioTestCase):
    async def test_raw_citation_maps_to_version_and_preserves_pre_filter_text(self):
        from backend.app.evidence.service import build_scoped_answer_bundle

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            engine = open_database(root / "papertrail.sqlite")
            try:
                revision = await imported_revision(engine, data_root, run_dir, "2401.00001v1")
                configured = settings()
                artifact = await build_collection_index(
                    engine, revision.id, data_root=data_root,
                    settings=configured, embedding_model=FakeEmbedding(),
                )
                scope = await prepare_answer_scope(
                    engine, revision.id, artifact.id,
                    data_root=data_root, settings=configured,
                )
                execution = await offline_answer(scope, raw_citations=True)
                self.assertEqual(execution.response.session.contexts[0].text.text, "")

                bundle = build_scoped_answer_bundle("answer-a", scope, execution)

                self.assertEqual(bundle.schema_version, 2)
                self.assertEqual(bundle.job_status, "succeeded")
                self.assertEqual(bundle.evidence[0].document_id, scope.documents[0].document_id)
                self.assertEqual(bundle.evidence[0].upstream_dockey, "2401.00001v1")
                self.assertEqual(bundle.evidence[0].summary, "Memory evidence")
                self.assertTrue(bundle.evidence[0].original_text)
                self.assertIn("memory", bundle.evidence[0].original_text.lower())
                self.assertEqual(bundle.evidence[0].locator_status, "unverified_upstream_label")
                self.assertIsNone(bundle.evidence[0].pdf_page)
                self.assertEqual(
                    [citation.mapping_status for citation in bundle.citations],
                    ["mapped", "missing_context"],
                )
                self.assertEqual(
                    bundle.citations[0].upstream_context_id,
                    bundle.evidence[0].upstream_context_id,
                )
                self.assertEqual(bundle.citations[0].evidence_id, bundle.evidence[0].evidence_id)
                self.assertEqual(bundle.citations[1].evidence_id, None)
                self.assertEqual(bundle.raw_answer[bundle.citations[0].start:bundle.citations[0].end],
                                 bundle.citations[0].upstream_context_id)
                self.assertEqual(bundle.allowed_documents[0]["versioned_id"], "2401.00001v1")
                self.assertEqual(AnswerBundle.model_validate_json(bundle.model_dump_json()), bundle)
            finally:
                engine.dispose()

    async def test_outside_context_stays_failed_even_without_an_answer_response(self):
        from backend.app.evidence.service import build_scoped_answer_bundle

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            engine = open_database(root / "papertrail.sqlite")
            try:
                revision = await imported_revision(engine, data_root, run_dir, "2401.00001v1")
                configured = settings()
                artifact = await build_collection_index(
                    engine, revision.id, data_root=data_root,
                    settings=configured, embedding_model=FakeEmbedding(),
                )
                scope = await prepare_answer_scope(
                    engine, revision.id, artifact.id,
                    data_root=data_root, settings=configured,
                )
                with self.assertLogs("paperqa.agents.main", level="ERROR"):
                    execution = await offline_answer(scope, foreign_context=True)
                bundle = build_scoped_answer_bundle("answer-failed", scope, execution)

                self.assertEqual(bundle.job_status, "failed")
                self.assertEqual(bundle.answer_outcome, "scope_violation")
                self.assertEqual(bundle.upstream_status, "unavailable")
                self.assertEqual(bundle.evidence[0].upstream_dockey, "foreign-version")
                self.assertIsNone(bundle.evidence[0].document_id)
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
