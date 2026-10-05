import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.api import create_app
from backend.app.db import open_database
from backend.app.jobs.service import get_job
from backend.app.jobs.worker import run_one_search
from backend.app.models import JobRequest
from backend.app.search.client import ArxivClient
from backend.app.search.suggestions import SuggestionError, generate_expansion, read_suggestion
from tests.test_multi_search import Responses
from tests.test_query_plan import EXPANSIONS

CONFIG = {"DEEPSEEK_MODEL": "test-model", "DEEPSEEK_BASE_URL": "https://example.test", "DEEPSEEK_API_KEY": "test-secret"}


class ExpansionTest(unittest.IsolatedAsyncioTestCase):
    async def test_one_explicit_model_call_preserves_baseline_and_records_validated_plan(self):
        calls = []
        def respond(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"expansions": EXPANSIONS})}}], "usage": {"total_tokens": 42}})
        with tempfile.TemporaryDirectory() as temp, httpx.MockTransport(respond) as transport:
            async with httpx.AsyncClient(transport=transport) as client:
                result = await generate_expansion("coding agent的发展潜力", "coding agent", Path(temp), config=CONFIG, client=client)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result["query_plan"]["baseline_query_text"], "coding agent")
            self.assertEqual(read_suggestion(Path(temp), result["suggestion_id"]), result)
            self.assertEqual(result["usage"]["total_tokens"], 42)
            self.assertIsNone(result["cost_usd"])
            self.assertNotIn("test-secret", json.dumps(result))

    async def test_invalid_json_or_timeout_keeps_single_query_and_does_not_retry(self):
        for content in ["not-json", json.dumps({"expansions": EXPANSIONS * 2})]:
            calls = []
            def respond(request):
                calls.append(request)
                return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 42}})
            with tempfile.TemporaryDirectory() as temp:
                async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                    with self.assertRaises(SuggestionError):
                        await generate_expansion("question", "coding agent", Path(temp), config=CONFIG, client=client)
                self.assertEqual(len(calls), 1)
                self.assertEqual(list((Path(temp) / "query-suggestions").glob("*.json")), [])
                failures = list((Path(temp) / "query-suggestions/attempts").glob("*.json"))
                self.assertEqual(len(failures), 1)
                failed = json.loads(failures[0].read_text(encoding="utf-8"))
                self.assertEqual(failed["usage"]["total_tokens"], 42)
                self.assertEqual(failed["status"], "failed")
                self.assertIsNone(failed["cost_usd"])
                self.assertEqual(failed["prompt_version"], "arxiv-expansion-v1")
                self.assertNotIn("test-secret", json.dumps(failed))

    async def test_expansion_timeout_is_one_call_without_automatic_retry(self):
        calls = []
        async def slow(request):
            calls.append(request)
            await asyncio.sleep(1)
            return httpx.Response(200, json={})
        with tempfile.TemporaryDirectory() as temp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
                with self.assertRaisesRegex(SuggestionError, "超时"):
                    await generate_expansion("question", "coding agent", Path(temp), config=CONFIG,
                        client=client, timeout_seconds=.01)
            self.assertEqual(len(calls), 1)
            failures = list((Path(temp) / "query-suggestions/attempts").glob("*.json"))
            self.assertEqual(len(failures), 1)
            failed = json.loads(failures[0].read_text(encoding="utf-8"))
            self.assertEqual(failed["error_kind"], "timeout")
            self.assertEqual(failed["usage"], {})
            self.assertIsNone(failed["cost_usd"])


class ApiMultiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "db.sqlite"
        self.engine = open_database(self.db)
        self.addCleanup(self.engine.dispose)
        self.client = TestClient(create_app(self.db, self.root / "data"))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.body = {"question": "coding agent的发展潜力", "query_text": "coding agent", "query_plan":
            {"version": 1, "baseline_query_text": "coding agent", "expansions": EXPANSIONS}}

    def test_idempotent_plan_runs_and_refreshes_without_queries(self):
        first = self.client.post("/api/searches", json=self.body, headers={"Idempotency-Key": "multi"})
        self.assertEqual(first.status_code, 202, first.text)
        self.assertEqual(self.client.post("/api/searches", json=self.body, headers={"Idempotency-Key": "multi"}).json(), first.json())
        with Session(self.engine) as session:
            self.assertIsNotNone(session.get(JobRequest, first.json()["job_id"]))
        transport = Responses()
        run_one_search(self.engine, worker_id="offline", data_root=self.root / "data",
            client=ArxivClient(transport=transport, max_attempts=1, minimum_interval_seconds=0))
        self.assertEqual(get_job(self.engine, first.json()["job_id"]).status, "succeeded")
        for _ in range(2):
            result = self.client.get("/api/searches/" + first.json()["search_id"])
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.json()["fusion"]["successful_queries"], 3)
            self.assertEqual(result.json()["query_plan"], self.body["query_plan"])
        self.assertEqual(len(transport.calls), 3)
        saved = result.json()
        collection = self.client.post("/api/collections", json={"selections": [{
            "search_id": saved["search_id"], "versioned_id": saved["candidates"][0]["versioned_id"]}]},
            headers={"Idempotency-Key": "multi-fixed-selection"})
        self.assertEqual(collection.status_code, 202, collection.text)
        draft = self.client.get("/api/collections/" + collection.json()["collection_id"]).json()
        self.assertEqual(draft["selections"][0]["snapshot_sha256"], saved["snapshot_sha256"])
        changed = {**self.body, "query_plan": {**self.body["query_plan"], "expansions": EXPANSIONS[:1]}}
        self.assertEqual(self.client.post("/api/searches", json=changed, headers={"Idempotency-Key": "multi"}).status_code, 409)

    def test_stale_baseline_or_invalid_plan_never_enqueues(self):
        for body in [{**self.body, "query_text": "memory agent"},
                     {**self.body, "query_plan": {**self.body["query_plan"], "expansions": []}}]:
            self.assertEqual(self.client.post("/api/searches", json=body, headers={"Idempotency-Key": "bad"}).status_code, 400)

    def test_all_failed_search_reads_child_causes_and_rejects_changed_records(self):
        submitted = self.client.post("/api/searches", json=self.body,
            headers={"Idempotency-Key": "limited"}).json()
        transport = Responses((429,))
        executed = run_one_search(self.engine, worker_id="offline", data_root=self.root / "data",
            client=ArxivClient(transport=transport, max_attempts=1, minimum_interval_seconds=0))
        endpoint = "/api/searches/" + submitted["search_id"]
        for _ in range(2):
            saved = self.client.get(endpoint).json()
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["failure"]["kind"], "rate_limited")
            self.assertEqual(saved["failure"]["http_attempts"], 1)
            self.assertIsNone(saved["candidates"])
        self.assertEqual(len(transport.calls), 1)
        child = self.root / "data/runs" / f"search-{executed.attempt_id}-q1" / "run.json"
        original = child.read_bytes()
        child.write_bytes(original + b" ")
        self.assertIsNone(self.client.get(endpoint).json()["failure"])
        child.write_bytes(original)
        response = child.parent / "response.atom"
        response.write_bytes(b"changed upstream response")
        self.assertIsNone(self.client.get(endpoint).json()["failure"])


if __name__ == "__main__":
    unittest.main()
