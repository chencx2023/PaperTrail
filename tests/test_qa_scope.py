"""T05 preflight must tie one answer to one frozen, ready native index."""

from pathlib import Path
import os
import tempfile
import unittest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
from paperqa import Docs
from paperqa.agents.search import get_directory_index
from paperqa.types import Doc, Text

from backend.app.db import open_database
from tests.test_collection_index import FakeEmbedding, imported_revision, settings
from tests.test_ingestion_selection import saved_search


class AnswerScopeTest(unittest.IsolatedAsyncioTestCase):
    async def test_extra_native_index_document_is_rejected_before_answering(self):
        from backend.app.collections.indexing import build_collection_index
        from backend.app.qa.scope import prepare_answer_scope

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
                    engine, revision.id, artifact.id, data_root=data_root, settings=configured,
                )
                native = await get_directory_index(scope.settings, build=False)
                foreign_doc = Doc(docname="Foreign", citation="Foreign", dockey="foreign-version")
                foreign_text = Text(
                    name="Foreign page 1", text="foreign-only sentinel",
                    doc=foreign_doc, embedding=[1.0, 0.5],
                )
                foreign = Docs()
                await foreign.aadd_texts(
                    [foreign_text], foreign_doc, settings=scope.settings,
                    embedding_model=FakeEmbedding(),
                )
                await native.add_document(
                    {"title": "Foreign", "year": "2026", "file_location": "foreign.pdf",
                     "body": "foreign-only sentinel"},
                    document=foreign, lock_acquisition_max_retries=3,
                )
                await native.save_index()

                with self.assertRaisesRegex(ValueError, "outside the confirmed revision"):
                    await prepare_answer_scope(
                        engine, revision.id, artifact.id,
                        data_root=data_root, settings=configured,
                    )
            finally:
                engine.dispose()

    async def test_two_revisions_resolve_to_their_own_ready_index(self):
        from backend.app.collections.indexing import build_collection_index
        from backend.app.qa.scope import prepare_answer_scope

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            engine = open_database(root / "papertrail.sqlite")
            try:
                first = await imported_revision(engine, data_root, run_dir, "2401.00001v1")
                second = await imported_revision(engine, data_root, run_dir, "2401.00001v3")
                configured = settings()
                first_index = await build_collection_index(
                    engine, first.id, data_root=data_root,
                    settings=configured, embedding_model=FakeEmbedding(),
                )
                second_index = await build_collection_index(
                    engine, second.id, data_root=data_root,
                    settings=configured, embedding_model=FakeEmbedding(),
                )

                for revision, artifact, version in (
                    (first, first_index, "2401.00001v1"),
                    (second, second_index, "2401.00001v3"),
                ):
                    scope = await prepare_answer_scope(
                        engine, revision.id, artifact.id,
                        data_root=data_root, settings=configured,
                    )
                    self.assertEqual(scope.revision_id, revision.id)
                    self.assertEqual(scope.index_artifact_id, artifact.id)
                    self.assertEqual(
                        [(doc.document_id, doc.versioned_id, doc.sha256) for doc in scope.documents],
                        [(revision.members[0].document_id, version, revision.members[0].sha256)],
                    )
                    self.assertEqual(
                        Path(scope.settings.agent.index.paper_directory).resolve(),
                        (data_root / artifact.artifact_ref / "scope").resolve(),
                    )
                    self.assertEqual(
                        set(scope.settings.agent.tool_names),
                        {"paper_search", "gather_evidence", "gen_answer", "complete"},
                    )
                    self.assertFalse(scope.settings.agent.index.sync_with_paper_directory)
                    self.assertFalse(scope.settings.agent.rebuild_index)

                with self.assertRaises(ValueError):
                    await prepare_answer_scope(
                        engine, first.id, second_index.id,
                        data_root=data_root, settings=configured,
                    )
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
