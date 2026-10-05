import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.app.api import create_app
from backend.app.db import open_database
from backend.app.search.suggestions import SuggestionError, generate_suggestion, read_suggestion


CONFIG = {"DEEPSEEK_MODEL": "test-model", "DEEPSEEK_BASE_URL": "https://example.test", "DEEPSEEK_API_KEY": "test-secret"}


class QuerySuggestionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def generate(self, content, **kwargs):
        def upstream(request):
            payload = json.loads(request.content)
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            self.assertEqual(payload["model"], "test-model")
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 25, "completion_tokens": 8, "total_tokens": 33}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            return await generate_suggestion("如何评估智能体记忆？", self.root, config=CONFIG, client=client, **kwargs)

    async def test_valid_suggestion_preserves_question_model_usage_without_search(self):
        result = await self.generate('{"keywords":["agent","memory","evaluation"]}')
        self.assertEqual(result["query_text"], "agent memory evaluation")
        saved = read_suggestion(self.root, result["suggestion_id"])
        self.assertEqual(saved["question"], "如何评估智能体记忆？")
        self.assertEqual(saved["model"], "test-model")
        self.assertEqual(saved["usage"]["total_tokens"], 33)
        self.assertIsNone(saved["cost_usd"])
        self.assertNotIn("test-secret", json.dumps(saved))
        self.assertFalse((self.root / "searches").exists())

    async def test_invalid_json_operator_phrase_and_overspecified_query_are_rejected(self):
        for value in ('not json', '{"keywords":["AND"]}', '{"keywords":["agent memory"]}', '{"keywords":["a","b","c","d","e","f","g"]}'):
            with self.subTest(value=value), self.assertRaises(SuggestionError):
                await self.generate(value)
        self.assertFalse((self.root / "query-suggestions").exists())

    async def test_timeout_has_no_retry_or_record(self):
        calls = []
        async def slow(request):
            calls.append(request)
            await asyncio.sleep(1)
            return httpx.Response(200, json={})
        async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
            with self.assertRaisesRegex(SuggestionError, "超时"):
                await generate_suggestion("question", self.root, config=CONFIG, client=client, timeout_seconds=.01)
        self.assertEqual(len(calls), 1)
        self.assertFalse((self.root / "query-suggestions").exists())

    async def test_provider_error_never_leaks_body_or_key(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(401, text="test-secret"))) as client:
            with self.assertRaises(SuggestionError) as caught:
                await generate_suggestion("question", self.root, config=CONFIG, client=client)
        self.assertNotIn("test-secret", str(caught.exception))


class SuggestionAdoptionTest(unittest.TestCase):
    def test_api_suggestion_does_not_enqueue_and_search_records_edited_adoption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            engine = open_database(root / "db.sqlite")
            async def fixed(question, data_root):
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"choices": [{"message": {"content": '{"keywords":["agent","memory"]}'}}]}))) as client:
                    return await generate_suggestion(question, data_root, config=CONFIG, client=client)
            with TestClient(create_app(root / "db.sqlite", root)) as client, patch("backend.app.api.generate_suggestion", fixed):
                result = client.post("/api/query-suggestions", json={"question": "memory question"})
                self.assertEqual(result.status_code, 200)
                suggestion = result.json()
                with engine.connect() as connection:
                    self.assertEqual(connection.execute(text("SELECT COUNT(*) FROM jobs")).scalar_one(), 0)
                body = {"question": "memory question", "query_text": "agent evaluation", "query_suggestion_id": suggestion["suggestion_id"]}
                result = client.post("/api/searches", json=body, headers={"Idempotency-Key": "adoption"})
                self.assertEqual(result.status_code, 202)
                search_id = result.json()["search_id"]
                adoption = json.loads((root / "query-suggestions/adoptions" / f"{search_id}.json").read_text(encoding="utf-8"))
                self.assertEqual(adoption["final_query_text"], "agent evaluation")
                self.assertEqual(adoption["suggestion_id"], suggestion["suggestion_id"])
                body["question"] = "changed question"
                self.assertEqual(client.post("/api/searches", json=body, headers={"Idempotency-Key": "mismatch"}).status_code, 400)
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
