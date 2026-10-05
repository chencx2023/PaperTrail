import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from backend.app.api import create_app
from backend.app.db import open_database
from backend.app.public_access import PublicAccess
from sqlalchemy import text


class PublicAccessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        engine = open_database(self.root / "db.sqlite")
        engine.dispose()
        self.gate = PublicAccess("x" * 48, mutation_limit=2, suggestion_limit=1)
        self.client = TestClient(create_app(self.root / "db.sqlite", self.root, public_access=self.gate), base_url="https://testserver")
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def test_no_credential_denies_ui_api_and_pdf(self):
        for url in ("/", "/api/history", "/api/health", "/api/answers/test/evidence/test/pdf", "/docs", "/.env"):
            self.assertEqual(self.client.get(url).status_code, 401)
        self.assertEqual(self.client.post("/api/searches", json={}).status_code, 401)

    def test_bootstrap_sets_http_only_cookie_and_removes_token_from_url(self):
        response = self.client.get('/?answer=example&access=' + 'x' * 48, follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers['location'], '/?answer=example')
        self.assertIn('HttpOnly', response.headers['set-cookie'])
        self.assertIn('Secure', response.headers['set-cookie'])
        self.assertEqual(self.client.get('/api/history').status_code, 200)
        self.assertEqual(response.headers['referrer-policy'], 'no-referrer')

    def test_tls_termination_keeps_cookie_secure_for_http_backend(self):
        response = self.client.get('http://testserver/?access=' + 'x' * 48, follow_redirects=False)
        self.assertIn('Secure', response.headers['set-cookie'])
        self.assertEqual(self.client.get('http://testserver/api/history').status_code, 401)
        self.assertEqual(self.client.get('/api/history').status_code, 200)

    def test_bearer_allows_full_submission_and_request_limits_apply(self):
        headers = {'Authorization': 'Bearer ' + 'x' * 48, 'Idempotency-Key': 'public'}
        body = {'question': 'memory question', 'query_text': 'agent memory'}
        self.assertEqual(self.client.post('/api/searches', json=body, headers=headers).status_code, 202)
        self.assertEqual(self.client.post('/api/searches', json=body, headers=headers).status_code, 202)
        self.assertEqual(self.client.post('/api/searches', json=body, headers=headers).status_code, 429)

    def test_cookie_cannot_submit_cross_origin_and_get_cannot_smuggle_mutation(self):
        self.client.get('/?access=' + 'x' * 48)
        self.assertEqual(self.client.post('/api/searches', json={}, headers={'Origin':'https://attacker.test'}).status_code, 403)
        self.client.cookies.clear()
        self.assertEqual(self.client.post('/api/searches?access=' + 'x' * 48, json={}).status_code, 401)

    def test_expansion_shares_suggestion_cap_without_calling_a_model(self):
        self.gate.suggestions = self.gate.suggestion_limit
        with patch('backend.app.api.generate_expansion', new_callable=AsyncMock) as upstream:
            upstream.return_value = {"suggestion_id": "test"}
            response = self.client.post('/api/query-expansions', json={"question": "q", "query_text": "agent"},
                headers={'Authorization': 'Bearer ' + 'x' * 48})
        upstream.assert_not_awaited()
        self.assertEqual(response.status_code, 429)

    def test_costly_retry_cannot_bypass_answer_operation_cap(self):
        engine = open_database(self.root / 'db.sqlite', create_schema=False)
        with engine.begin() as connection:
            connection.execute(text("INSERT INTO jobs (id, kind, resource_id, status, idempotency_key, request_hash, created_at, updated_at) VALUES ('job', 'answer', 'answer', 'failed', 'cap', 'hash', 'now', 'now')"))
        engine.dispose()
        self.gate.expensive['answer'] = 10
        response = self.client.post('/api/jobs/job/retry', json={'expected_attempt_id':'old'}, headers={'Authorization':'Bearer ' + 'x' * 48})
        self.assertEqual(response.status_code, 429)


if __name__ == '__main__':
    unittest.main()
