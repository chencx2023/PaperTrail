"""Ordinary worker defaults select the accepted profile without API access."""
from contextlib import ExitStack
import json
import unittest
from unittest.mock import patch

from aviary.core import ToolCall, ToolRequestMessage
from lmi import LiteLLMModel
from lmi.types import LLMResult
from paperqa import Docs
from paperqa.types import Context

from backend.app.evidence.persistence import load_answer_run
from backend.app.jobs.answer_worker import _answer_deadline_seconds
from backend.app.jobs.dispatcher import run_one_job
from backend.app.jobs.index_worker import _runtime_settings
from backend.app.qa.service import execute_scoped_answer
from backend.app.qa.submission import submit_answer
import tests.test_api_revision as revision_fixture


CONFIG = {
    "DEEPSEEK_MODEL": "deepseek-flash", "DEEPSEEK_BASE_URL": "https://example.invalid/chat",
    "DEEPSEEK_API_KEY": "offline-secret", "EMBEDDING_MODEL": "text-embedding-v4",
    "EMBEDDING_BASE_URL": "https://example.invalid/embedding", "EMBEDDING_API_KEY": "offline-secret",
    "EMBEDDING_DIMENSIONS": 2, "EMBEDDING_BATCH_SIZE": 10,
}


class CompactActivationTest(unittest.TestCase):
    def test_ordinary_factory_defaults_to_flash_profile_but_preserves_explicit_off(self):
        with patch("scripts.check_models.load_config", return_value=dict(CONFIG)):
            active = _runtime_settings()
        self.assertTrue(active.grounded_answers)
        self.assertEqual(active.grounded_profile, "compact")
        self.assertEqual(active.llm, "openai/deepseek-flash")
        self.assertEqual(active.summary_llm, active.llm)
        self.assertEqual(active.agent.agent_llm, active.llm)
        self.assertEqual(_answer_deadline_seconds(active), 875)
        with patch("scripts.check_models.load_config", return_value={**CONFIG,
                "PAPERTRAIL_GROUNDED_ANSWERS": "false"}):
            disabled = _runtime_settings()
        self.assertFalse(disabled.grounded_answers)
        self.assertEqual(_answer_deadline_seconds(disabled), 620)

    def test_explicit_operator_settings_survive_the_factory_defaults(self):
        overrides = {**CONFIG, "PAPERTRAIL_GROUNDED_PROFILE": "audited",
                     "PAPERTRAIL_QA_REASONING": "disabled", "PAPERTRAIL_AGENT_TIMEOUT": "700"}
        with patch("scripts.check_models.load_config", return_value=overrides):
            configured = _runtime_settings()
        self.assertEqual(configured.grounded_profile, "audited")
        self.assertEqual(configured.agent.timeout, 700)
        params = configured.llm_config["model_list"][0]["litellm_params"]
        self.assertEqual(params["extra_body"]["thinking"]["type"], "disabled")

    def test_dispatcher_without_injected_settings_publishes_compact_answer_and_uses_deadline(self):
        with ExitStack() as resources:
            fixture = revision_fixture.ApiRevisionTest()
            fixture.setUp()
            resources.callback(fixture.doCleanups)
            revision_id, artifact_id = fixture.ready_revision()
            engine, data_root = fixture.engine, fixture.data_root
            submitted = submit_answer(engine, revision_id=revision_id, index_artifact_id=artifact_id,
                question="What is studied?", idempotency_key="ordinary-compact")
            calls, native_calls, deadlines = [], [], []

            async def model_stub(_model, *, messages, name=None, **_kwargs):
                calls.append(name); payload = json.loads(messages[1].content)
                if name == "grounded_outline":
                    key = payload["original_source_units"][0]["id"]
                    data = {"parts": [{"id": "p1", "question_part": "What is studied?", "status": "supported",
                        "answer_kind": "fact", "direct_answer": {"statement": "The paper studies agent memory.",
                        "scope": "method", "sentence_ids": [key]}, "facts": [], "gap": ""}]}
                elif name == "grounded_focus":
                    target = payload["target_facts"][0]
                    data = {"fact_checks": [{"fact_id": target["fact_id"], "source_sentence_ids": target["original"]["sentence_ids"],
                        "conditions_comparison": "Fixture evidence supports the statement.", "counterexample": "", "verdict": "supported"}],
                        "corrections": [], "additions": []}
                else:
                    raise AssertionError("Unexpected model stage")
                return LLMResult(model="offline-compact", name=name, text=json.dumps(data), prompt_count=11, completion_count=5)

            async def executor(scope, question):
                self.assertEqual(question, "What is studied?")
                self.assertTrue(scope.settings.grounded_answers)
                self.assertEqual(scope.settings.grounded_profile, "compact")
                decisions = iter([
                    ("paper_search", {"query": "memory", "min_year": None, "max_year": None}),
                    ("gather_evidence", {"question": question}), ("gen_answer", {}),
                    ("complete", {"has_successful_answer": True}),
                ])
                async def selector(_messages, tools):
                    name, arguments = next(decisions)
                    tool = next(t for t in tools if t.info.name == name)
                    return ToolRequestMessage(tool_calls=[ToolCall.from_tool(tool, **arguments)])
                async def evidence(docs, query, **_kwargs):
                    query.contexts = [Context(context="Memory evidence", text=docs.texts[0], score=9)]
                    return query
                async def native_forbidden(*_args, **_kwargs):
                    native_calls.append("called")
                    raise AssertionError("Compact default must use its own generation stages")
                with patch.object(type(scope.settings), "make_aviary_tool_selector", return_value=selector), \
                     patch.object(Docs, "aget_evidence", evidence), \
                     patch.object(Docs, "aquery", native_forbidden):
                    return await execute_scoped_answer(scope, question)

            def deadline(configured):
                deadlines.append(_answer_deadline_seconds(configured))
                return deadlines[-1]

            with patch("scripts.check_models.load_config", return_value=dict(CONFIG)), \
                 patch.object(LiteLLMModel, "call_single", model_stub), \
                 patch("backend.app.jobs.answer_worker._answer_deadline_seconds", side_effect=deadline):
                # answer_settings is intentionally omitted: use the ordinary factory.
                completed = run_one_job(engine, worker_id="ordinary-compact", data_root=data_root,
                    answer_executor=executor)
            self.assertEqual(completed.status, "succeeded")
            self.assertEqual(calls, ["grounded_outline", "grounded_focus"])
            self.assertEqual(native_calls, [])
            self.assertEqual(deadlines, [875])
            saved = load_answer_run(engine, submitted.answer_run_id, data_root=data_root)
            self.assertEqual(saved.run_config["prompts"]["generation"]["version"], "source-led-answer-v27-compact")
            self.assertIn("agent memory", saved.raw_answer)
            self.assertTrue(saved.citations)
            self.assertTrue(all(c.mapping_status == "mapped" for c in saved.citations))


if __name__ == "__main__":
    unittest.main()
