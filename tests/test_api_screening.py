import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.app.api import create_app
from backend.app.db import open_database
from backend.app.jobs.worker import run_one_search
from backend.app.public_access import PublicAccess
from tests.test_api_search import FixedTransport, fixed_client
from tests.test_abstract_screening import CONFIG, span_output


class ApiScreeningTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.database_path = self.root / "db.sqlite"
        self.data_root = self.root / "data"
        self.engine = open_database(self.database_path)
        self.addCleanup(self.engine.dispose)
        self.client = TestClient(create_app(self.database_path, self.data_root))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        result = self.client.post("/api/searches", json={"question": "如何评估 Agent 记忆？", "query_text": "agent memory"}, headers={"Idempotency-Key": "screen-test"})
        self.search_id = result.json()["search_id"]
        self.path = f"/api/searches/{self.search_id}/screening"

    def complete(self):
        run_one_search(self.engine, worker_id="test-worker", data_root=self.data_root, client=fixed_client(FixedTransport()))

    def test_get_is_read_only_and_pending_search_cannot_start_screening(self):
        self.assertEqual(self.client.get(self.path).json(), {"screening": None})
        self.assertEqual(self.client.post(self.path, json={}).status_code, 409)
        self.assertFalse((self.data_root / "abstract-screenings").exists())
        self.assertEqual(self.client.get('/api/searches/unknown/screening').status_code, 404)

    def test_completed_search_get_plans_all_candidates_without_model_or_writes(self):
        self.complete()
        search = self.client.get(f'/api/searches/{self.search_id}').json()
        count = len(search['candidates'])
        with patch('backend.app.api.generate_screening', side_effect=AssertionError('GET is read-only')):
            value = self.client.get(self.path).json()
        self.assertEqual(value['screening'], None)
        self.assertEqual(value['screening_plan'], {'total_count': count, 'verified_count': 0,
            'remaining_count': count, 'batch_count': 1, 'wait_timeout_ms': 75000})
        self.assertFalse((self.data_root / 'abstract-screenings').exists())

    def test_success_reopens_without_model_or_new_job_and_rejects_client_supplied_question(self):
        from tests.test_abstract_screening import screening as feature
        self.assertIsNotNone(feature, "摘要筛选模块尚未实现")
        from backend.app.search.screening import generate_screening
        self.complete()
        before = self.client.get(f"/api/searches/{self.search_id}").json()
        calls = []
        async def fixed(search, data_root):
            def upstream(request):
                calls.append(json.loads(request.content))
                return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(span_output(search))}}], "usage": {"total_tokens": 100}})
            async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
                return await generate_screening(search, data_root, config=CONFIG, client=client)
        with patch("backend.app.api.generate_screening", fixed):
            result = self.client.post(self.path, json={})
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(self.client.post(self.path, json={}).json()["screening_id"], result.json()["screening_id"])
        with patch("backend.app.api.generate_screening", side_effect=AssertionError("GET must not invoke model")):
            saved = self.client.get(self.path)
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["screening"]["question"], before["question"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.client.get(f"/api/searches/{self.search_id}").json(), before)
        self.assertEqual(self.client.post(self.path, json={"question": "改掉原问题"}).status_code, 422)
        with self.engine.connect() as connection:
            self.assertEqual(connection.execute(text("SELECT count(*) FROM jobs")).scalar(), 1)

    def test_invalid_search_snapshot_blocks_model(self):
        from backend.app import api as api_module
        self.assertTrue(hasattr(api_module, "generate_screening"), "筛选 API 尚未实现")
        self.complete()
        candidate_path = next((self.data_root / "runs").glob("*/candidates.json"))
        candidate_path.write_text("[]", encoding="utf-8")
        with patch("backend.app.api.generate_screening", side_effect=AssertionError("must validate snapshot first")):
            self.assertEqual(self.client.post(self.path, json={}).status_code, 409)
            self.assertEqual(self.client.get(self.path).status_code, 409)

    def test_public_screening_uses_costly_operation_budget(self):
        gate = PublicAccess("x" * 48)
        with TestClient(create_app(self.database_path, self.data_root, public_access=gate)) as client:
            gate.expensive["screening"] = 10
            result = client.post(self.path, json={}, headers={"Authorization": "Bearer " + "x" * 48})
        self.assertEqual(result.status_code, 429)


if __name__ == "__main__":
    unittest.main()
