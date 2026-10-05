"""Offline native-index/agent integration; model decisions and generation are fakes."""
import hashlib
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from model_config import PaperTrailSettings, build_settings
from paperqa_agent import Recorder, configure_agent, execute_agent, prepare_index, recording_environment
from paperqa import Docs
from paperqa.types import Context, Doc, Text
from aviary.core import ToolCall, ToolRequestMessage


class NativeAgentTest(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, max_steps, mode="normal"):
        config = {"DEEPSEEK_MODEL": "deepseek-flash", "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
                  "DEEPSEEK_API_KEY": "offline-secret", "EMBEDDING_MODEL": "text-embedding-v4",
                  "EMBEDDING_BASE_URL": "https://example.invalid/v1", "EMBEDDING_API_KEY": "offline-secret",
                  "EMBEDDING_DIMENSIONS": 2, "EMBEDDING_BATCH_SIZE": 10}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            doc = Doc(docname="Mem0", citation="Local PDF: Mem0.pdf", dockey=hashlib.sha256(b"fixture").hexdigest())
            text = Text(name="Mem0 page 1", text="Mem0 extracts memories and updates existing memories.",
                        doc=doc, embedding=[1.0, 0.0])
            papers = [{"spec": {"filename": "Mem0.pdf"}, "cache_sha256": "fixture",
                       "doc": doc, "texts": [text]}]
            settings = build_settings(config)
            manifest = configure_agent(settings, papers, root=root, max_steps=max_steps)
            if mode == "timeout":
                settings.agent.timeout = 1
            await prepare_index(settings, papers, manifest)
            recorder = Recorder(root, ["offline-secret"], {str(doc.dockey)})
            decisions = iter([("paper_search", {"query": "Mem0", "min_year": None, "max_year": None}),
                              ("gather_evidence", {"question": "How are memories updated?"}),
                              ("gen_answer", {}), ("complete", {"has_successful_answer": True})])

            async def selector(messages, tools):
                name, arguments = next(decisions)
                if name == "gather_evidence" and mode == "provider_error":
                    raise RuntimeError("Offline simulated provider failure")
                if name == "gather_evidence" and mode == "timeout":
                    await asyncio.sleep(2)
                tool = next(t for t in tools if t.info.name == name)
                return ToolRequestMessage(tool_calls=[ToolCall.from_tool(tool, **arguments)])

            async def evidence(docs, query, **kwargs):
                self.assertEqual(len(docs.docs), 1)  # Actual native paper_search loaded the snapshot.
                query.contexts = [Context(context="Memories are extracted and updated.", text=docs.texts[0], score=9)]
                return query

            async def answer(docs, query, **kwargs):
                if not query.contexts:
                    await evidence(docs, query)
                query.answer = "Memories are updated from extracted facts."
                query.context = "Offline evidence"
                return query

            with patch.object(PaperTrailSettings, "make_aviary_tool_selector", return_value=selector), \
                 patch.object(Docs, "aget_evidence", evidence), patch.object(Docs, "aquery", answer):
                response = await execute_agent(settings, "How are memories updated?", recorder)
            events = [json.loads(line) for line in (root / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
            saved = json.loads((root / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["contexts"][0]["text"]["text"], text.text)
            self.assertEqual(saved["tool_history"], response.session.tool_history)
            self.assertNotIn("embedding", saved["contexts"][0]["text"])
            return response, events

    async def test_native_tools_finish_and_preserve_evidence(self):
        response, events = await self.run_case(8)
        self.assertEqual(response.status.value, "success")
        starts = [e for e in events if e["event"] == "tools_started"]
        self.assertEqual(len(starts), 4)
        self.assertTrue(all(e["origin"] == "agent" for e in starts))
        self.assertIn("complete", events[0]["tools"])

    async def test_truncation_records_framework_fallback(self):
        response, events = await self.run_case(1)
        self.assertEqual(response.status.value, "truncated")
        starts = [e for e in events if e["event"] == "tools_started"]
        self.assertEqual([e["origin"] for e in starts], ["agent", "framework_fallback"])

    async def test_timeout_retains_truncated_status_despite_fallback_answer(self):
        response, events = await self.run_case(8, "timeout")
        self.assertEqual(response.status.value, "truncated")
        self.assertTrue(any(e.get("origin") == "framework_fallback" for e in events))

    async def test_provider_failure_is_not_success_despite_fallback_answer(self):
        with self.assertLogs("paperqa.agents.main", level="ERROR"):
            response, events = await self.run_case(8, "provider_error")
        self.assertEqual(response.status.value, "fail")
        self.assertTrue(any(e.get("origin") == "framework_fallback" for e in events))

    async def test_two_collection_indexes_and_sessions_do_not_cross_read(self):
        config = {"DEEPSEEK_MODEL": "deepseek-flash", "DEEPSEEK_BASE_URL": "https://example.invalid",
                  "DEEPSEEK_API_KEY": "fake", "EMBEDDING_MODEL": "text-embedding-v4",
                  "EMBEDDING_BASE_URL": "https://example.invalid/v1", "EMBEDDING_API_KEY": "fake",
                  "EMBEDDING_DIMENSIONS": 2, "EMBEDDING_BATCH_SIZE": 10}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings_list = []
            for label in ("A", "B"):
                doc = Doc(docname=label, citation=label, dockey=label)
                text = Text(name=label, text=f"sharedtopic secret{label}", doc=doc, embedding=[1.0, 0.0])
                papers = [{"spec": {"filename": label + ".pdf", "sha256": label},
                           "cache_sha256": label, "doc": doc, "texts": [text]}]
                settings = build_settings(config)
                manifest = configure_agent(settings, papers, root=root)
                await prepare_index(settings, papers, manifest)
                settings_list.append(settings)
            self.assertNotEqual(settings_list[0].agent.index.name, settings_list[1].agent.index.name)
            for label, settings in zip(("A", "B", "A"), (*settings_list, settings_list[0])):
                recorder = Recorder(root, [], {label})
                env = recording_environment(recorder)("sharedtopic", settings, Docs())
                _, tools = await env.reset()
                self.assertFalse(env.state.docs.docs)
                tool = next(t for t in tools if t.info.name == "paper_search")
                action = ToolRequestMessage(tool_calls=[ToolCall.from_tool(tool, query="sharedtopic", min_year=None, max_year=None)])
                await recorder.action(action, None)
                await env.step(action)
                self.assertEqual(set(env.state.docs.docs), {label})
                self.assertTrue(all(t.doc.dockey == label for t in env.state.docs.texts))


if __name__ == "__main__":
    unittest.main()
