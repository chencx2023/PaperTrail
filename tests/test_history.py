import hashlib
import unittest
from sqlalchemy import text

from backend.app.jobs.claim import claim_next_job
from backend.app.jobs.worker import interrupt_expired_attempt
from backend.app.history import read_history
from datetime import datetime, timedelta, timezone
from tests import test_api_revision


class HistoryTest(unittest.TestCase):
    def setUp(self):
        self.fixture = test_api_revision.ApiRevisionTest(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_read_only_history_covers_search_collection_revision_and_distinct_attempts(self):
        f = self.fixture
        revision, index = f.ready_revision()
        submitted = f.client.post("/api/answers", json={"revision_id": revision, "index_artifact_id": index, "question": "How is memory studied?"}, headers={"Idempotency-Key": "history"}).json()
        claim = claim_next_job(f.engine, worker_id="history", supported_kinds=("answer",))
        interrupt_expired_attempt(f.engine, claim.attempt_id, worker_stopped_confirmed=True, now=datetime.now(timezone.utc) + timedelta(minutes=1))
        retried = f.client.post(f"/api/jobs/{submitted['job_id']}/retry", json={"expected_attempt_id": claim.attempt_id}).json()
        before = hashlib.sha256(f.database_path.read_bytes()).hexdigest()
        result = f.client.get("/api/history")
        self.assertEqual(result.status_code, 200)
        items = result.json()["items"]
        self.assertEqual({item["kind"] for item in items}, {"search", "collection", "revision", "answer"})
        answers = {item["id"]: item for item in items if item["kind"] == "answer"}
        self.assertEqual(answers[submitted["answer_run_id"]]["status"], "interrupted")
        self.assertEqual(answers[retried["answer_run_id"]]["status"], "queued")
        self.assertEqual(answers[retried["answer_run_id"]]["previous_answer_run_id"], submitted["answer_run_id"])
        self.assertEqual(answers[submitted["answer_run_id"]]["question"], "How is memory studied?")
        self.assertEqual(hashlib.sha256(f.database_path.read_bytes()).hexdigest(), before)

    def test_cursor_ties_are_stable_and_filters_and_limits_are_validated(self):
        f = self.fixture
        for number in range(5):
            f.client.post("/api/searches", json={"question": f"question {number}", "query_text": "agent"}, headers={"Idempotency-Key": f"search-{number}"})
        with f.engine.begin() as connection:
            connection.execute(text("UPDATE search_runs SET created_at = '2026-10-01T00:00:00Z'"))
        first = f.client.get("/api/history?kind=search&limit=2").json()
        second = f.client.get("/api/history", params={"kind": "search", "limit": 2, "cursor": first["next_cursor"]}).json()
        third = read_history(f.engine, kind="search", limit=2, cursor=second["next_cursor"])
        ids = [item["id"] for page in (first, second, third) for item in page["items"]]
        self.assertEqual(len(ids), 5)
        self.assertEqual(len(set(ids)), 5)
        self.assertEqual(ids, sorted(ids, reverse=True))
        self.assertIsNone(third["next_cursor"])
        for query in ("limit=0", "limit=101", "kind=unknown", "cursor=broken"):
            self.assertIn(f.client.get("/api/history?" + query).status_code, (400, 422))


if __name__ == "__main__":
    unittest.main()
