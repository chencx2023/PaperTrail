"""T05: every answer starts fresh and observes the frozen document scope."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
import logging
from io import StringIO
from unittest.mock import patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
from aviary.core import ToolCall, ToolRequestMessage
from paperqa import Docs
from paperqa.agents.env import PaperQAEnvironment
from paperqa.types import Context, Doc, Text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from model_config import PaperTrailSettings

from backend.app.collections.indexing import build_collection_index
from backend.app.db import open_database
from backend.app.qa.scope import prepare_answer_scope
from tests.test_collection_index import FakeEmbedding, imported_revision, settings
from tests.test_ingestion_selection import saved_search


async def offline_answer(
    scope, *, foreign_context=False, raise_after_foreign_context=False,
    answer_calls=None, raw_citations=False, selector_error_after_search=False,
):
    from backend.app.qa.service import execute_scoped_answer

    decisions = iter([
        ("paper_search", {"query": "memory", "min_year": None, "max_year": None}),
        ("gather_evidence", {"question": "What is studied?"}),
        ("gen_answer", {}),
        ("complete", {"has_successful_answer": True}),
    ])

    selector_calls = 0

    async def selector(_messages, tools):
        nonlocal selector_calls
        selector_calls += 1
        if selector_error_after_search and selector_calls > 1:
            raise RuntimeError("offline simulated model-provider failure")
        name, arguments = next(decisions)
        tool = next(tool for tool in tools if tool.info.name == name)
        return ToolRequestMessage(tool_calls=[ToolCall.from_tool(tool, **arguments)])

    async def evidence(docs, query, **_kwargs):
        if foreign_context:
            foreign_doc = Doc(docname="Foreign", citation="Foreign", dockey="foreign-version")
            text = Text(
                name="Foreign page 1", text="foreign-only sentinel",
                doc=foreign_doc, embedding=[1.0, 0.5],
            )
        else:
            text = docs.texts[0]
        query.contexts = [Context(context="Memory evidence", text=text, score=9)]
        if raise_after_foreign_context:
            raise RuntimeError("evidence tool failed after changing contexts")
        return query

    async def answer(_docs, query, **_kwargs):
        if answer_calls is not None:
            answer_calls.append("called")
        query.answer = "The paper studies agent memory."
        query.raw_answer = (
            f"The paper studies agent memory ("
            f"{query.contexts[0].id}, pqac-missing)."
            if raw_citations else "The paper studies agent memory."
        )
        query.context = "Observed final generation context"
        return query

    with patch.object(PaperTrailSettings, "make_aviary_tool_selector", return_value=selector), \
         patch.object(Docs, "aget_evidence", evidence), \
         patch.object(Docs, "aquery", answer):
        return await execute_scoped_answer(scope, "What is studied?")


class AnswerExecutionTest(unittest.IsolatedAsyncioTestCase):
    async def test_agent_run_does_not_log_model_credentials_at_info_level(self):
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
                stream = StringIO()
                logger = logging.getLogger("paperqa.agents.main")
                previous_level = logger.level
                handler = logging.StreamHandler(stream)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
                try:
                    result = await offline_answer(scope)
                    self.assertEqual(result.execution_status, "succeeded")
                    self.assertNotIn("offline-secret", stream.getvalue())
                finally:
                    logger.removeHandler(handler)
                    logger.setLevel(previous_level)
            finally:
                engine.dispose()

    async def test_foreign_context_fails_and_blocks_fallback_answer_generation(self):
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
                answer_calls = []
                with self.assertLogs("paperqa.agents.main", level="ERROR"):
                    result = await offline_answer(
                        scope, foreign_context=True, answer_calls=answer_calls,
                    )
                self.assertEqual(result.execution_status, "failed")
                self.assertEqual(result.failure_kind, "scope_violation")
                self.assertEqual(result.snapshots[-1].context_dockeys, ("foreign-version",))
                self.assertEqual(answer_calls, [])
            finally:
                engine.dispose()

    async def test_foreign_context_is_seen_when_evidence_tool_then_raises(self):
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
                answer_calls = []
                with self.assertLogs("paperqa.agents.main", level="ERROR"):
                    result = await offline_answer(
                        scope, foreign_context=True,
                        raise_after_foreign_context=True, answer_calls=answer_calls,
                    )
                self.assertEqual(result.failure_kind, "scope_violation")
                self.assertEqual(result.snapshots[-1].context_dockeys, ("foreign-version",))
                self.assertEqual(answer_calls, [])
            finally:
                engine.dispose()

    async def test_scope_is_checked_when_environment_tool_call_raises(self):
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
                original = PaperQAEnvironment.exec_tool_calls

                async def interrupted(env, message, *args, **kwargs):
                    if message.tool_calls[0].function.name == "gather_evidence":
                        foreign_doc = Doc(
                            docname="Foreign", citation="Foreign", dockey="foreign-version",
                        )
                        env.state.session.contexts = [Context(
                            context="Foreign evidence",
                            text=Text(
                                name="Foreign page 1", text="foreign-only sentinel",
                                doc=foreign_doc, embedding=[1.0, 0.5],
                            ),
                            score=9,
                        )]
                        raise RuntimeError("environment stopped after changing contexts")
                    return await original(env, message, *args, **kwargs)

                answer_calls = []
                with patch.object(PaperQAEnvironment, "exec_tool_calls", interrupted), \
                     self.assertLogs("paperqa.agents.main", level="ERROR"):
                    result = await offline_answer(scope, answer_calls=answer_calls)
                self.assertEqual(result.failure_kind, "scope_violation")
                self.assertEqual(result.snapshots[-1].context_dockeys, ("foreign-version",))
                self.assertEqual(answer_calls, [])
            finally:
                engine.dispose()

    async def test_sequential_a_b_a_answers_use_new_sessions_and_only_their_own_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            engine = open_database(root / "papertrail.sqlite")
            try:
                scopes = []
                for version in ("2401.00001v1", "2401.00001v3"):
                    revision = await imported_revision(engine, data_root, run_dir, version)
                    configured = settings()
                    artifact = await build_collection_index(
                        engine, revision.id, data_root=data_root,
                        settings=configured, embedding_model=FakeEmbedding(),
                    )
                    scopes.append(await prepare_answer_scope(
                        engine, revision.id, artifact.id,
                        data_root=data_root, settings=configured,
                    ))

                results = [await offline_answer(scopes[i]) for i in (0, 1, 0)]
                self.assertEqual([r.execution_status for r in results], ["succeeded"] * 3)
                self.assertEqual([r.failure_kind for r in results], [None] * 3)
                self.assertEqual(
                    [r.snapshots[-1].loaded_dockeys for r in results],
                    [("2401.00001v1",), ("2401.00001v3",), ("2401.00001v1",)],
                )
                self.assertEqual(
                    [r.snapshots[-1].context_dockeys for r in results],
                    [("2401.00001v1",), ("2401.00001v3",), ("2401.00001v1",)],
                )
                self.assertEqual(
                    [r.final_context for r in results],
                    ["Observed final generation context"] * 3,
                )
                self.assertEqual(
                    [r.final_context_coverage for r in results],
                    ["paperqa_session_context_only; full_model_request_unobservable"] * 3,
                )
                self.assertEqual(
                    len({str(r.response.session.id) for r in results}), 3,
                )
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
