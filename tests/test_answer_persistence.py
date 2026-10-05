"""T05 answer files remain readable after restart without a model call."""

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from sqlalchemy.orm import Session

from backend.app.collections.indexing import build_collection_index
from backend.app.db import open_database
from backend.app.evidence.service import build_scoped_answer_bundle
from backend.app.models import AnswerRun
from backend.app.qa.scope import prepare_answer_scope
from tests.test_collection_index import FakeEmbedding, imported_revision, settings
from tests.test_ingestion_selection import saved_search
from tests.test_qa_execution import offline_answer


class AnswerPersistenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_saved_answer_reopens_with_original_evidence_and_safe_configuration(self):
        from backend.app.evidence.persistence import load_answer_run, save_answer_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            database_path = root / "papertrail.sqlite"
            engine = open_database(database_path)
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
                run_id = uuid4().hex
                bundle = build_scoped_answer_bundle(run_id, scope, execution)

                saved = save_answer_run(
                    engine, scope, execution, bundle, data_root=data_root,
                )
                answer_file = data_root / saved.artifact_ref
                self.assertEqual(saved.id, run_id)
                self.assertEqual(hashlib.sha256(answer_file.read_bytes()).hexdigest(),
                                 saved.artifact_sha256)
                self.assertNotIn("offline-secret", answer_file.read_text(encoding="utf-8"))
                self.assertEqual(list(answer_file.parent.glob("*.tmp")), [])
                with Session(engine) as session:
                    row = session.get(AnswerRun, run_id)
                    self.assertEqual(row.collection_revision_id, revision.id)
                    self.assertEqual(row.index_artifact_id, artifact.id)
                    self.assertEqual(row.artifact_sha256, saved.artifact_sha256)
                    self.assertEqual(row.execution_status, "succeeded")
                with self.assertRaises(FileExistsError):
                    save_answer_run(engine, scope, execution, bundle, data_root=data_root)
            finally:
                engine.dispose()

            engine = open_database(database_path, create_schema=False)
            try:
                with patch("backend.app.qa.service.run_agent", side_effect=AssertionError(
                    "reading an answer must not invoke PaperQA"
                )):
                    reloaded = load_answer_run(engine, run_id, data_root=data_root)
                self.assertEqual(reloaded.raw_answer, bundle.raw_answer)
                self.assertEqual(reloaded.display_answer, bundle.display_answer)
                self.assertEqual(reloaded.answer_outcome, bundle.answer_outcome)
                self.assertEqual(reloaded.evidence[0].original_text,
                                 bundle.evidence[0].original_text)
                self.assertEqual(reloaded.citations, bundle.citations)
                self.assertEqual(reloaded.usage["coverage"], "partial")
                self.assertIsNone(reloaded.usage["embedding_tokens"])
                self.assertEqual(reloaded.run_config["model_roles"]["agent"],
                                 configured.agent.agent_llm)
                self.assertEqual(reloaded.run_config["coverage"],
                                 "selected_effective_fields; provider_configs_excluded")
                self.assertIn("prompts", reloaded.run_config)
                self.assertEqual(
                    reloaded.run_config["prompts"]["qa"]["sha256"],
                    hashlib.sha256(configured.prompts.qa.encode("utf-8")).hexdigest(),
                )
                self.assertIn("context", reloaded.run_config["prompts"])
                from backend.app.qa.context import CONTEXT_POLICY
                self.assertEqual(reloaded.run_config["prompts"]["context"], CONTEXT_POLICY)
                selection = reloaded.run_config['context_selection']
                # This older fake execution supplies no serializer key pool: remain truthful.
                self.assertEqual(selection['status'], 'unknown')
                self.assertNotIn('selected_context_ids', selection)
                self.assertEqual(reloaded.scope_observation["revision_id"], revision.id)
                answer_file.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "hash"):
                    load_answer_run(engine, run_id, data_root=data_root)
            finally:
                engine.dispose()

    async def test_failed_database_commit_does_not_expose_an_orphan_file_as_saved_answer(self):
        from backend.app.evidence.persistence import load_answer_run, save_answer_run

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
                run_id = uuid4().hex
                bundle = build_scoped_answer_bundle(run_id, scope, execution)
                answer_file = data_root / "answers" / f"{run_id}.json"
                original_commit = Session.commit

                def fail_after_file_publish(session):
                    if any(isinstance(value, AnswerRun) for value in session.new):
                        self.assertTrue(answer_file.is_file())
                        raise RuntimeError("simulated database commit failure")
                    return original_commit(session)

                with patch.object(Session, "commit", fail_after_file_publish):
                    with self.assertRaisesRegex(RuntimeError, "commit failure"):
                        save_answer_run(
                            engine, scope, execution, bundle, data_root=data_root,
                        )
                self.assertTrue(answer_file.is_file())
                with self.assertRaises(LookupError):
                    load_answer_run(engine, run_id, data_root=data_root)
            finally:
                engine.dispose()

    async def test_scope_failure_is_saved_as_failure_without_an_upstream_answer(self):
        from backend.app.evidence.persistence import load_answer_run, save_answer_run

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
                run_id = uuid4().hex
                bundle = build_scoped_answer_bundle(run_id, scope, execution)
                save_answer_run(engine, scope, execution, bundle, data_root=data_root)
                reloaded = load_answer_run(engine, run_id, data_root=data_root)

                self.assertEqual(reloaded.job_status, "failed")
                self.assertEqual(reloaded.answer_outcome, "scope_violation")
                self.assertEqual(reloaded.upstream_status, "unavailable")
                self.assertEqual(reloaded.evidence[0].upstream_dockey, "foreign-version")
                self.assertIsNone(reloaded.evidence[0].document_id)
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
