"""T05 truncated and provider-error executions keep distinct saved statuses."""

import os
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from backend.app.collections.indexing import build_collection_index
from backend.app.db import open_database
from backend.app.evidence.persistence import load_answer_run, save_answer_run
from backend.app.evidence.service import build_scoped_answer_bundle
from backend.app.qa.scope import prepare_answer_scope
from tests.test_collection_index import FakeEmbedding, imported_revision, settings
from tests.test_ingestion_selection import saved_search
from tests.test_qa_execution import offline_answer


class AnswerTerminalStateTest(unittest.IsolatedAsyncioTestCase):
    async def test_truncated_draft_and_provider_error_remain_distinct_after_reload(self):
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

                scope.settings.agent.max_timesteps = 2
                truncated = await offline_answer(scope, raw_citations=True)
                truncated_id = uuid4().hex
                truncated_bundle = build_scoped_answer_bundle(truncated_id, scope, truncated)
                save_answer_run(engine, scope, truncated, truncated_bundle, data_root=data_root)
                draft = load_answer_run(engine, truncated_id, data_root=data_root)
                self.assertEqual(draft.job_status, "succeeded")
                self.assertEqual(draft.answer_outcome, "truncated")
                self.assertEqual(draft.upstream_status, "truncated")
                self.assertTrue(draft.evidence[0].original_text)

                scope.settings.agent.max_timesteps = None
                with self.assertLogs("paperqa.agents.main", level="ERROR"):
                    failed = await offline_answer(scope, selector_error_after_search=True)
                failed_id = uuid4().hex
                failed_bundle = build_scoped_answer_bundle(failed_id, scope, failed)
                save_answer_run(engine, scope, failed, failed_bundle, data_root=data_root)
                failure = load_answer_run(engine, failed_id, data_root=data_root)
                self.assertEqual(failure.job_status, "failed")
                self.assertEqual(failure.upstream_status, "fail")
                self.assertNotEqual(failure.answer_outcome, "reported_sufficient")
                self.assertEqual(failure.scope_observation["failure_kind"], "upstream_failed")
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
