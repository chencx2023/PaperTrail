"""T04: a frozen revision gets its own validated PaperQA index artifact."""

from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from importlib.metadata import version as installed_version

import httpx
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
from paperqa import Docs
from paperqa.agents.search import get_directory_index

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from model_config import build_settings

from backend.app.db import open_database
from tests.test_document_preparation import text_pdf_bytes
from tests.test_ingestion_selection import saved_search


class FakeEmbedding:
    def __init__(self, dimensions=2, *, fail=False):
        self.ndim = dimensions
        self.fail = fail
        self.calls = 0

    async def embed_documents(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError("offline embedding failure")
        return [[float((len(text) % 17) + 1), *([0.5] * (self.ndim - 1))] for text in texts]


def settings(embedding_name="text-embedding-v4", dimensions=2):
    return build_settings({
        "DEEPSEEK_MODEL": "deepseek-flash",
        "DEEPSEEK_BASE_URL": "https://example.invalid/chat",
        "DEEPSEEK_API_KEY": "offline-secret",
        "EMBEDDING_MODEL": embedding_name,
        "EMBEDDING_BASE_URL": "https://example.invalid/embedding",
        "EMBEDDING_API_KEY": "offline-secret",
        "EMBEDDING_DIMENSIONS": dimensions,
        "EMBEDDING_BATCH_SIZE": 10,
    })


async def imported_revision(engine, data_root, run_dir, versioned_id):
    from backend.app.collections.service import (
        SelectedSource,
        confirm_collection_revision,
        create_collection,
        set_collection_selection,
    )
    from backend.app.ingestion.batch import import_collection

    collection = create_collection(engine)
    set_collection_selection(
        engine, collection.id, expected_revision=0, data_root=data_root,
        selections=(SelectedSource(run_dir, versioned_id),),
    )
    with httpx.Client(transport=httpx.MockTransport(
        lambda _request: httpx.Response(200, headers={"Content-Type": "application/pdf"}, content=text_pdf_bytes())
    )) as client:
        report = await import_collection(
            engine, collection.id, expected_selection_revision=1,
            data_root=data_root, client=client,
        )
    assert report.status == "succeeded"
    return confirm_collection_revision(
        engine, collection.id, expected_selection_revision=1,
        import_run_id=report.id, document_ids=(report.items[0].document_id,),
        data_root=data_root,
    )


class CollectionIndexTest(unittest.IsolatedAsyncioTestCase):
    async def test_partial_import_confirmed_subset_reopens_as_only_that_scope(self):
        from backend.app.collections.service import (
            SelectedSource, confirm_collection_revision, create_collection, set_collection_selection,
        )
        from backend.app.collections.indexing import build_collection_index, settings_for_artifact
        from backend.app.ingestion.batch import import_collection

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            engine = open_database(root / "papertrail.sqlite")
            try:
                collection = create_collection(engine)
                set_collection_selection(
                    engine, collection.id, expected_revision=0, data_root=data_root,
                    selections=(
                        SelectedSource(run_dir, "2401.00001v1"),
                        SelectedSource(run_dir, "2401.00001v3"),
                    ),
                )

                def response(request):
                    if request.url.path.endswith("v1"):
                        return httpx.Response(
                            200, headers={"Content-Type": "application/pdf"}, content=text_pdf_bytes()
                        )
                    return httpx.Response(404, content=b"missing")

                with httpx.Client(transport=httpx.MockTransport(response)) as client:
                    report = await import_collection(
                        engine, collection.id, expected_selection_revision=1,
                        data_root=data_root, client=client,
                    )
                self.assertEqual(report.status, "partial_failed")
                revision = confirm_collection_revision(
                    engine, collection.id, expected_selection_revision=1,
                    import_run_id=report.id, document_ids=(report.items[0].document_id,),
                    data_root=data_root,
                )
                configured = settings()
                artifact = await build_collection_index(
                    engine, revision.id, data_root=data_root,
                    settings=configured, embedding_model=FakeEmbedding(),
                )
                reopened = await get_directory_index(
                    settings_for_artifact(configured, artifact, data_root=data_root), build=False
                )
                self.assertEqual(await reopened.count, 1)
                self.assertEqual(set(await reopened.index_files), {report.items[0].document_id + ".pdf"})
                saved = await reopened.get_saved_object(report.items[0].document_id + ".pdf")
                self.assertEqual(set(saved.docs), {"2401.00001v1"})
            finally:
                engine.dispose()

    async def test_two_revisions_build_distinct_reopenable_native_indexes(self):
        from backend.app.collections.indexing import (
            build_collection_index,
            settings_for_artifact,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            engine = open_database(root / "papertrail.sqlite")
            try:
                first = await imported_revision(engine, data_root, run_dir, "2401.00001v1")
                second = await imported_revision(engine, data_root, run_dir, "2401.00001v3")
                configured = settings()
                a = await build_collection_index(
                    engine, first.id, data_root=data_root,
                    settings=configured, embedding_model=FakeEmbedding(),
                )
                b = await build_collection_index(
                    engine, second.id, data_root=data_root,
                    settings=configured, embedding_model=FakeEmbedding(),
                )
                self.assertEqual((a.status, b.status), ("ready", "ready"))
                self.assertNotEqual(a.config_hash, b.config_hash)
                self.assertNotEqual(a.artifact_ref, b.artifact_ref)
                self.assertNotIn("offline-secret", a.config_json)
                for artifact, expected in ((a, "2401.00001v1"), (b, "2401.00001v3")):
                    native_settings = settings_for_artifact(configured, artifact, data_root=data_root)
                    index = await get_directory_index(native_settings, build=False)
                    self.assertEqual(await index.count, 1)
                    self.assertEqual(set(await index.index_files), {first.members[0].document_id + ".pdf"} if expected.endswith("v1") else {second.members[0].document_id + ".pdf"})
                    loaded = await index.get_saved_object(next(iter(await index.index_files)))
                    self.assertIsInstance(loaded, Docs)
                    self.assertEqual(set(loaded.docs), {expected})
                    self.assertTrue(all(len(text.embedding) == 2 for text in loaded.texts))
            finally:
                engine.dispose()

    async def test_config_change_rebuilds_and_same_config_reuses_ready_artifact(self):
        from backend.app.collections.indexing import build_collection_index

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            engine = open_database(root / "papertrail.sqlite")
            try:
                revision = await imported_revision(engine, data_root, run_dir, "2401.00001v1")
                first_embedding = FakeEmbedding()
                first = await build_collection_index(
                    engine, revision.id, data_root=data_root,
                    settings=settings(), embedding_model=first_embedding,
                )
                no_call = FakeEmbedding(fail=True)
                repeated = await build_collection_index(
                    engine, revision.id, data_root=data_root,
                    settings=settings(), embedding_model=no_call,
                )
                self.assertEqual(first.id, repeated.id)
                self.assertEqual(no_call.calls, 0)
                changed = await build_collection_index(
                    engine, revision.id, data_root=data_root,
                    settings=settings(dimensions=3), embedding_model=FakeEmbedding(3),
                )
                self.assertEqual(changed.status, "ready")
                self.assertNotEqual(first.config_hash, changed.config_hash)
                self.assertNotEqual(first.artifact_ref, changed.artifact_ref)
            finally:
                engine.dispose()

    async def test_failed_build_and_incomplete_final_directory_never_become_ready(self):
        from backend.app.collections.indexing import build_collection_index, get_index_artifact

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            engine = open_database(root / "papertrail.sqlite")
            try:
                revision = await imported_revision(engine, data_root, run_dir, "2401.00001v1")
                with self.assertRaisesRegex(RuntimeError, "offline embedding failure"):
                    await build_collection_index(
                        engine, revision.id, data_root=data_root,
                        settings=settings(), embedding_model=FakeEmbedding(fail=True),
                    )
                failed = get_index_artifact(engine, revision.id, settings=settings())
                self.assertEqual(failed.status, "failed")
                self.assertIsNone(failed.artifact_ref)
                self.assertEqual(failed.error_kind, "RuntimeError")
                self.assertFalse((data_root / "indexes" / "collections" / failed.config_hash).exists())

                incomplete = data_root / "indexes" / "collections" / failed.config_hash
                incomplete.mkdir(parents=True)
                (incomplete / "partial.txt").write_text("not an index", encoding="utf-8")
                with self.assertRaises(ValueError):
                    await build_collection_index(
                        engine, revision.id, data_root=data_root,
                        settings=settings(), embedding_model=FakeEmbedding(),
                    )
                self.assertEqual(get_index_artifact(engine, revision.id, settings=settings()).status, "failed")
            finally:
                engine.dispose()

    async def test_ready_artifact_rejects_changed_parser_settings_and_corrupt_pdf(self):
        from backend.app.collections.indexing import (
            build_collection_index,
            get_index_artifact,
            settings_for_artifact,
        )
        from backend.app.ingestion.service import InvalidDocumentBlob, get_document_version

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            engine = open_database(root / "papertrail.sqlite")
            try:
                revision = await imported_revision(engine, data_root, run_dir, "2401.00001v1")
                configured = settings()
                ready = await build_collection_index(
                    engine, revision.id, data_root=data_root,
                    settings=configured, embedding_model=FakeEmbedding(),
                )
                changed = settings()
                changed.parsing.reader_config["chunk_chars"] = 5000
                with self.assertRaises(ValueError):
                    settings_for_artifact(changed, ready, data_root=data_root)

                def changed_parser_version(package):
                    if package == "paper-qa-pypdf":
                        return "new-parser-version"
                    return installed_version(package)

                with patch("backend.app.collections.indexing.version", side_effect=changed_parser_version):
                    with self.assertRaisesRegex(ValueError, "parser"):
                        settings_for_artifact(configured, ready, data_root=data_root)

                document = get_document_version(engine, revision.members[0].document_id)
                (data_root / document.blob_ref).write_bytes(b"corrupted")
                with self.assertRaisesRegex(ValueError, "PDF"):
                    settings_for_artifact(configured, ready, data_root=data_root)
                with self.assertRaises(InvalidDocumentBlob):
                    await build_collection_index(
                        engine, revision.id, data_root=data_root,
                        settings=configured, embedding_model=FakeEmbedding(fail=True),
                    )
                self.assertEqual(get_index_artifact(engine, revision.id, settings=configured).status, "failed")
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
