"""Offline integration of source-led generation with the scoped Agent and artifacts."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from aviary.core import ToolCall, ToolRequestMessage
from lmi import LiteLLMModel
from lmi.types import LLMResult
from paperqa import Docs
from paperqa.types import Context

from backend.app.collections.indexing import build_collection_index
from backend.app.db import open_database
from backend.app.evidence.persistence import load_answer_run, save_answer_run
from backend.app.evidence.service import build_scoped_answer_bundle
from backend.app.qa.grounded import grounded_policy_identity
from backend.app.qa.scope import prepare_answer_scope
from backend.app.qa.service import execute_scoped_answer
from tests.test_collection_index import FakeEmbedding, imported_revision, settings
from tests.test_ingestion_selection import saved_search
from tests.test_qa_execution import PaperTrailSettings, offline_answer


OFFLINE_MODEL = "offline-grounded-test-model"


def model_stub(calls, *, invalid_outline=False, invalid_focus=False, focus_raw=None):
    async def call_single(_model, *, messages, name=None, **_kwargs):
        calls.append({"name": name, "messages": messages})
        if invalid_outline or (invalid_focus and name == "grounded_focus"):
            text = '{"parts": ['
        elif name == "grounded_outline":
            payload = json.loads(messages[1].content)
            unit = payload["original_source_units"][0]
            text = json.dumps({"parts": [{
                "id": "p1", "question_part": "What does the paper study?",
                "answer_kind": "fact", "operating_premise": None,
                "direct_answer": {"statement": "The paper studies agent memory.",
                                  "scope": "method", "sentence_ids": [unit["id"]]},
                "status": "supported", "gap": "",
                "facts": [],
            }]})
        elif name == "grounded_focus":
            payload = json.loads(messages[1].content)
            identities = payload["target_facts"]
            assert len(identities) == 1
            identity = identities[0]
            text = focus_raw if focus_raw is not None else json.dumps({"fact_checks": [{
                "fact_id": identity["fact_id"],
                "source_sentence_ids": identity["original"]["sentence_ids"],
                "conditions_comparison": "The fixed fixture only claims that the paper studies agent memory.",
                "counterexample": "", "verdict": "supported",
            }], "corrections": []})
        else:
            raise AssertionError("Unexpected provider call outside the stubbed generation stages")
        return LLMResult(model=OFFLINE_MODEL, name=name, text=text,
                         prompt_count=11, completion_count=5)

    return call_single


async def answer_with_two_generation_calls(scope):
    """Use the real native tools, requesting gen_answer twice in sequential Agent steps."""
    decisions = iter([
        ("paper_search", {"query": "memory", "min_year": None, "max_year": None}),
        ("gather_evidence", {"question": "What is studied?"}),
        ("gen_answer", {}),
        ("gen_answer", {}),
        ("complete", {"has_successful_answer": True}),
    ])

    async def selector(_messages, tools):
        name, arguments = next(decisions)
        tool = next(tool for tool in tools if tool.info.name == name)
        return ToolRequestMessage(tool_calls=[ToolCall.from_tool(tool, **arguments)])

    async def evidence(docs, query, **_kwargs):
        query.contexts = [Context(context="Memory evidence", text=docs.texts[0], score=9)]
        return query

    async def native_answer_must_not_run(*_args, **_kwargs):
        raise AssertionError("Enabled generation must not fall back to the native answer model")

    with patch.object(PaperTrailSettings, "make_aviary_tool_selector", return_value=selector), \
         patch.object(Docs, "aget_evidence", evidence), \
         patch.object(Docs, "aquery", native_answer_must_not_run):
        return await execute_scoped_answer(scope, "What is studied?")


class ScopedGroundedTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        self.engine = open_database(self.root / "papertrail.sqlite")
        run_dir = saved_search(self.data_root)
        revision = await imported_revision(
            self.engine, self.data_root, run_dir, "2401.00001v1",
        )
        configured = settings()
        configured.grounded_answers = True
        artifact = await build_collection_index(
            self.engine, revision.id, data_root=self.data_root,
            settings=configured, embedding_model=FakeEmbedding(),
        )
        self.scope = await prepare_answer_scope(
            self.engine, revision.id, artifact.id,
            data_root=self.data_root, settings=configured,
        )

    async def asyncTearDown(self):
        self.engine.dispose()
        self.temporary.cleanup()

    async def test_compact_profile_uses_two_real_tools_stages_and_persists_its_policy(self):
        self.scope.settings.__dict__["grounded_profile"] = "compact"
        calls = []
        async def compact_stub(_model, *, messages, name=None, **_kwargs):
            calls.append(name)
            payload = json.loads(messages[1].content)
            if name == "grounded_outline":
                unit = payload["original_source_units"][0]
                data = {"parts":[{"id":"p1","question_part":"What is studied?","status":"supported",
                    "answer_kind":"criterion","direct_answer":{"statement":"The paper studies agent memory.","scope":"method","sentence_ids":[unit["id"]]},"facts":[],"gap":""}]}
            elif name == "grounded_focus":
                fact = payload["target_facts"][0]
                data = {"fact_checks":[{"fact_id":fact["fact_id"],"source_sentence_ids":fact["original"]["sentence_ids"],
                    "conditions_comparison":"The original supports the fixed memory statement.","counterexample":"","verdict":"supported"}],"corrections":[],"additions":[]}
            else: raise AssertionError("Unexpected additional generation stage")
            return LLMResult(model=OFFLINE_MODEL,name=name,text=json.dumps(data),prompt_count=11,completion_count=5)
        with patch.object(LiteLLMModel,"call_single",compact_stub):
            execution = await answer_with_two_generation_calls(self.scope)
        self.assertEqual(calls,["grounded_outline","grounded_focus"])
        self.assertEqual(execution.execution_status,"succeeded")
        run_id=uuid4().hex;bundle=build_scoped_answer_bundle(run_id,self.scope,execution)
        save_answer_run(self.engine,self.scope,execution,bundle,data_root=self.data_root)
        saved=load_answer_run(self.engine,run_id,data_root=self.data_root)
        self.assertEqual(saved.run_config["prompts"]["generation"]["max_model_calls"],2)
        self.assertEqual(saved.run_config["prompts"]["generation"]["version"],"source-led-answer-v27-compact")
        self.assertIn("Original source condition:",saved.raw_answer)
        self.assertTrue(all(c.mapping_status=="mapped" for c in saved.citations))

    async def test_enabled_generation_runs_real_tool_chain_and_persists_citations_audit_and_usage(self):
        calls, native_answer_calls = [], []
        with patch.object(LiteLLMModel, "call_single", model_stub(calls)):
            execution = await offline_answer(self.scope, answer_calls=native_answer_calls)
        self.assertEqual(execution.execution_status, "succeeded")
        self.assertEqual(native_answer_calls, [])
        self.assertEqual([call["name"] for call in calls],
                         ["grounded_outline", "grounded_focus"])
        self.assertEqual(len(execution.generation_records), 1)
        audit = execution.generation_records[0]
        self.assertTrue(audit["answer_published"])
        self.assertEqual(audit["version"], grounded_policy_identity()["version"])
        self.assertEqual(audit["final_context_sha256"],
                         hashlib.sha256(execution.final_context.encode()).hexdigest())
        self.assertTrue(all(stage["structurally_valid"] for stage in audit["stages"].values()))
        self.assertEqual(audit["stages"]["grounded_focus"]["fact_checks"][0]["verdict"], "supported")
        self.assertEqual(len(audit["stages"]["grounded_focus"]["fact_checks"]), 1)
        self.assertTrue(execution.contexts[0].original_text)
        self.assertIn("Original passage", execution.final_context)
        source_id = audit["rendered_claims"][0]["context_ids"][0]
        self.assertIn("(" + source_id + ")", execution.response.session.raw_answer)
        self.assertIn(audit["stages"]["grounded_focus"]["parsed"]["parts"][0]["facts"][0]["statement"],
                      execution.response.session.raw_answer)
        self.assertEqual(execution.response.session.token_counts[OFFLINE_MODEL], [22, 10])

        run_id = uuid4().hex
        bundle = build_scoped_answer_bundle(run_id, self.scope, execution)
        self.assertTrue(bundle.citations)
        self.assertTrue(all(c.mapping_status == "mapped" for c in bundle.citations))
        save_answer_run(self.engine, self.scope, execution, bundle, data_root=self.data_root)
        saved = load_answer_run(self.engine, run_id, data_root=self.data_root)
        self.assertEqual(saved.raw_answer, execution.response.session.raw_answer)
        self.assertEqual(saved.final_context, execution.final_context)
        self.assertEqual(saved.scope_observation["generation_records"], list(execution.generation_records))
        self.assertEqual(saved.run_config["prompts"]["generation"], grounded_policy_identity())
        self.assertEqual(saved.usage["tokens_by_upstream_model"][OFFLINE_MODEL], [22, 10])
        self.assertTrue(all(e.original_text for e in saved.evidence))

    async def test_window_ids_originals_and_policy_survive_real_tool_chain_and_reload(self):
        from tests.test_source_windows import fixture
        from backend.app.qa.context import window_context_serializer, WINDOW_CONTEXT_POLICY
        self.scope.settings.custom_context_serializer = window_context_serializer
        decisions = iter([("paper_search", {"query": "memory", "min_year": None, "max_year": None}),
            ("gather_evidence", {"question": "What is studied?"}), ("gen_answer", {}),
            ("gather_evidence", {"question": "Another aspect?"}), ("gen_answer", {}),
            ("complete", {"has_successful_answer": True})])
        async def selector(_messages, tools):
            name, arguments = next(decisions)
            return ToolRequestMessage(tool_calls=[ToolCall.from_tool(next(t for t in tools if t.info.name == name), **arguments)])
        collected = 0
        async def evidence(docs, query, **_kwargs):
            nonlocal collected
            collected += 1
            if collected == 2:
                extra = docs.texts[0].model_copy(update={"name": "additional", "text": "Another source sentence."})
                docs.texts.append(extra)
                query.contexts = [*query.contexts, Context(context="Additional evidence", score=1, text=extra)]
                return query
            context, native = fixture()
            fixed_doc = docs.texts[0].doc
            native = [text.model_copy(update={"doc": fixed_doc}) for text in native]
            docs.texts = native
            query.contexts = [context.model_copy(update={"text": native[0]})]
            return query
        calls, focus_count = [], 0
        stub = model_stub(calls)
        async def first_focus_invalid(model, *, messages, name=None, **kwargs):
            nonlocal focus_count
            if name == "grounded_focus":
                focus_count += 1
                if focus_count == 1:
                    return await model_stub(calls, invalid_focus=True)(model, messages=messages, name=name, **kwargs)
            return await stub(model, messages=messages, name=name, **kwargs)
        with patch.object(PaperTrailSettings, "make_aviary_tool_selector", return_value=selector), \
             patch.object(Docs, "aget_evidence", evidence), \
             patch.object(LiteLLMModel, "call_single", first_focus_invalid):
            execution = await execute_scoped_answer(self.scope, "What is studied?")
        self.assertEqual(execution.execution_status, "succeeded")
        self.assertEqual(len(execution.generation_records), 2)
        record = execution.generation_records[0]
        window = record["source_window_expansions"][0]
        self.assertTrue(record["source_window_policy"]["enabled"])
        self.assertEqual(execution.generation_records[1]["source_window_expansions"], [window])
        captured = next(c for c in execution.contexts if c.upstream_context_id == window["window_context_id"])
        self.assertEqual(len(captured.original_text), 3700)
        self.assertIn("separate value", captured.original_text)
        self.assertIn("continuation", captured.upstream_locator)
        run_id = uuid4().hex
        bundle = build_scoped_answer_bundle(run_id, self.scope, execution)
        save_answer_run(self.engine, self.scope, execution, bundle, data_root=self.data_root)
        saved = load_answer_run(self.engine, run_id, data_root=self.data_root)
        self.assertEqual(saved.run_config["prompts"]["context"], WINDOW_CONTEXT_POLICY)
        self.assertEqual(saved.run_config["context_selection"]["status"], "known")
        self.assertEqual(saved.evidence[0].original_text, captured.original_text)
        self.assertTrue(all(c.mapping_status == "mapped" for c in saved.citations))
        self.assertTrue(all(c.upstream_context_id == window["window_context_id"] for c in saved.citations))
        self.assertEqual(saved.scope_observation["generation_records"][0]["source_window_expansions"], [window])

    async def test_invalid_generation_cannot_become_success_when_agent_calls_complete_true(self):
        calls = []
        with patch.object(LiteLLMModel, "call_single", model_stub(calls, invalid_outline=True)):
            execution = await offline_answer(self.scope)
        self.assertEqual(len(calls), 1)
        self.assertEqual(execution.execution_status, "failed")
        self.assertEqual(execution.failure_kind, "grounded_generation_failed")
        self.assertIsNone(execution.response)
        self.assertEqual(len(execution.generation_records), 1)
        self.assertFalse(execution.generation_records[0]["answer_published"])
        self.assertEqual(execution.generation_records[0]["status"], "grounded_outline_failed")
        self.assertIn("complete", [snapshot.stage for snapshot in execution.snapshots])
        run_id = uuid4().hex
        bundle = build_scoped_answer_bundle(run_id, self.scope, execution)
        self.assertEqual(bundle.job_status, "failed")
        self.assertEqual(bundle.raw_answer, "")
        self.assertEqual(bundle.citations, [])
        save_answer_run(self.engine, self.scope, execution, bundle, data_root=self.data_root)
        saved = load_answer_run(self.engine, run_id, data_root=self.data_root)
        self.assertEqual(saved.job_status, "failed")
        self.assertFalse(saved.scope_observation["generation_records"][0]["answer_published"])
        self.assertEqual(saved.usage["tokens_by_upstream_model"][OFFLINE_MODEL], [11, 5])
        self.assertEqual(saved.raw_answer, "")
        self.assertEqual(saved.display_answer, "")
        self.assertEqual(saved.citations, [])
        self.assertEqual(saved.scope_observation["generation_records"][0]["model_usage"], [{
            "stage": "grounded_outline", "model": OFFLINE_MODEL,
            "prompt_tokens": 11, "completion_tokens": 5,
        }])

    async def test_validated_single_generation_finishes_before_another_agent_request(self):
        calls = []
        with patch.object(LiteLLMModel, "call_single", model_stub(calls)):
            execution = await answer_with_two_generation_calls(self.scope)
        self.assertEqual(execution.execution_status, "succeeded")
        self.assertEqual([call["name"] for call in calls],
                         ["grounded_outline", "grounded_focus"])
        self.assertEqual(len(execution.generation_records), 1)
        self.assertEqual([snapshot.stage for snapshot in execution.snapshots].count("gen_answer"), 1)
        self.assertEqual(execution.generation_records[0]["completion_mode"], "local_native_complete_after_validated_generation")
        self.assertEqual(execution.response.session.token_counts[OFFLINE_MODEL], [22, 10])

    async def test_focus_failure_preserves_both_model_usages_without_publishing_an_answer(self):
        calls = []
        with patch.object(LiteLLMModel, "call_single", model_stub(calls, invalid_focus=True)):
            execution = await offline_answer(self.scope)
        self.assertEqual([call["name"] for call in calls], ["grounded_outline", "grounded_focus"])
        self.assertEqual(execution.execution_status, "failed")
        self.assertEqual(execution.failure_kind, "grounded_generation_failed")
        self.assertIsNone(execution.response)
        audit = execution.generation_records[0]
        self.assertFalse(audit["answer_published"])
        self.assertEqual(audit["status"], "grounded_focus_failed")
        self.assertTrue(audit["stages"]["grounded_outline"]["structurally_valid"])
        self.assertFalse(audit["stages"]["grounded_focus"]["structurally_valid"])
        self.assertEqual(audit["model_usage"], [
            {"stage": "grounded_outline", "model": OFFLINE_MODEL, "prompt_tokens": 11, "completion_tokens": 5},
            {"stage": "grounded_focus", "batch_id": 1, "model": OFFLINE_MODEL, "prompt_tokens": 11, "completion_tokens": 5}])
        self.assertEqual(execution.session_token_counts[OFFLINE_MODEL], [22, 10])
        run_id = uuid4().hex
        bundle = build_scoped_answer_bundle(run_id, self.scope, execution)
        save_answer_run(self.engine, self.scope, execution, bundle, data_root=self.data_root)
        saved = load_answer_run(self.engine, run_id, data_root=self.data_root)
        self.assertEqual(saved.job_status, "failed")
        self.assertEqual(saved.raw_answer, "")
        self.assertEqual(saved.citations, [])
        self.assertEqual(saved.usage["tokens_by_upstream_model"][OFFLINE_MODEL], [22, 10])
        self.assertEqual(saved.scope_observation["generation_records"][0]["model_usage"], audit["model_usage"])

    async def test_missing_fact_checks_remains_failure_after_agent_complete_and_persistence(self):
        calls = []
        with patch.object(LiteLLMModel, "call_single", model_stub(calls, focus_raw='{"corrections":[]}')):
            execution = await offline_answer(self.scope)
        self.assertEqual([call["name"] for call in calls], ["grounded_outline", "grounded_focus"])
        self.assertEqual(execution.execution_status, "failed")
        self.assertIsNone(execution.response)
        self.assertEqual(execution.session_token_counts[OFFLINE_MODEL], [22, 10])
        run_id = uuid4().hex
        bundle = build_scoped_answer_bundle(run_id, self.scope, execution)
        save_answer_run(self.engine, self.scope, execution, bundle, data_root=self.data_root)
        saved = load_answer_run(self.engine, run_id, data_root=self.data_root)
        self.assertEqual(saved.job_status, "failed")
        self.assertEqual(saved.raw_answer, "")
        self.assertFalse(saved.scope_observation["generation_records"][0]["answer_published"])

    async def test_repeated_failed_context_does_not_repeat_provider_calls_or_token_charges(self):
        calls = []
        with patch.object(LiteLLMModel, "call_single", model_stub(calls, invalid_focus=True)):
            execution = await answer_with_two_generation_calls(self.scope)
        self.assertEqual(execution.execution_status, "failed")
        self.assertEqual(execution.failure_kind, "grounded_generation_failed")
        self.assertEqual([call["name"] for call in calls], ["grounded_outline", "grounded_focus"])
        self.assertEqual(len(execution.generation_records), 1)
        self.assertEqual([snapshot.stage for snapshot in execution.snapshots].count("gen_answer"), 2)
        self.assertEqual(execution.session_token_counts[OFFLINE_MODEL], [22, 10])

    async def test_disabled_strategy_preserves_native_execution_without_grounded_calls(self):
        self.scope.settings.grounded_answers = False
        calls, native_answer_calls = [], []
        with patch.object(LiteLLMModel, "call_single", model_stub(calls)):
            execution = await offline_answer(self.scope, answer_calls=native_answer_calls)
        self.assertEqual(execution.execution_status, "succeeded")
        self.assertEqual(native_answer_calls, ["called"])
        self.assertEqual(calls, [])
        self.assertEqual(execution.generation_records, ())
        self.assertEqual(execution.final_context, "Observed final generation context")

    async def test_configured_attempt_bound_stops_failed_agent_without_reporting_success(self):
        self.scope.settings.answer.max_answer_attempts = 1
        calls = []
        with patch.object(LiteLLMModel, "call_single", model_stub(calls, invalid_focus=True)):
            execution = await answer_with_two_generation_calls(self.scope)
        self.assertEqual(execution.execution_status, "failed")
        self.assertEqual(execution.failure_kind, "grounded_generation_failed")
        self.assertEqual([call["name"] for call in calls], ["grounded_outline", "grounded_focus"])
        self.assertEqual([snapshot.stage for snapshot in execution.snapshots].count("gen_answer"), 2)
        self.assertNotIn("complete", [snapshot.stage for snapshot in execution.snapshots])
        self.assertFalse(execution.generation_records[-1]["answer_published"])


if __name__ == "__main__":
    unittest.main()
