"""Public action budgets cover both the two-page flow and legacy submissions."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.app.api import create_app
from backend.app.db import open_database
from backend.app.public_access import PublicAccess


class PublicResearchLimitsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = self.root / 'db.sqlite'
        engine = open_database(self.database)
        engine.dispose()
        self.gate = PublicAccess('x' * 48)
        self.client = TestClient(create_app(self.database, self.root,
            public_access=self.gate), base_url='https://testserver')
        self.client.__enter__()
        self.headers = {'Authorization': 'Bearer ' + 'x' * 48,
                        'Idempotency-Key': 'public-action'}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def seed_operation(self, kind):
        engine = open_database(self.database, create_schema=False)
        with engine.begin() as connection:
            connection.execute(text("INSERT INTO research_records (id, title, title_is_custom, operation_id, created_at, updated_at) VALUES ('research', 'Public test', 0, 'operation', 'now', 'now')"))
            connection.execute(text("INSERT INTO research_operations (id, research_id, idempotency_key, request_hash, request_json, kind, phase, status, created_at, updated_at) VALUES ('operation', 'research', 'saved', 'hash', '{}', :kind, 'start', 'failed', 'now', 'now')"), {'kind': kind})
        engine.dispose()

    def test_new_prepare_and_answer_routes_stop_before_submission_at_cap(self):
        cases = [
            ('prepare', 'prepare', 'prepare_research',
             {'versioned_ids': ['1908.06600v1'], 'expected_pool_revision_id': None}),
            ('answer', 'answers', 'answer_research',
             {'question': 'What does this paper study?', 'document_ids': ['doc'],
              'expected_pool_revision_id': 'pool'}),
        ]
        for budget, suffix, handler, body in cases:
            with self.subTest(budget=budget):
                self.gate.expensive[budget] = 10
                with patch('backend.app.api.' + handler) as submit:
                    response = self.client.post('/api/researches/research/' + suffix,
                        json=body, headers=self.headers)
                self.assertEqual(response.status_code, 429)
                submit.assert_not_called()
                self.assertEqual(response.json()['code'], 'PUBLIC_ACCESS_LIMIT')

    def test_legacy_import_and_index_share_the_preparation_budget(self):
        self.gate.expensive['prepare'] = 9
        # Invalid bodies reach validation but cannot invoke downloads or embeddings.
        self.assertEqual(self.client.post('/api/collections', json={},
            headers=self.headers).status_code, 422)
        self.assertEqual(self.gate.expensive['prepare'], 10)
        for path in ('/api/collections/collection/imports',
                     '/api/collections/collection/revisions',
                     '/api/researches/research/prepare'):
            with self.subTest(path=path):
                self.assertEqual(self.client.post(path, json={},
                    headers=self.headers).status_code, 429)

    def test_one_preparation_action_is_charged_once_and_reads_are_free(self):
        with patch('backend.app.api.prepare_research', return_value={
            'research_id': 'research', 'operation': {'kind': 'prepare', 'phase': 'import'}}) as submit:
            response = self.client.post('/api/researches/research/prepare',
                json={'versioned_ids': ['1908.06600v1'],
                      'expected_pool_revision_id': None}, headers=self.headers)
        self.assertEqual(response.status_code, 202)
        submit.assert_called_once()
        self.assertEqual(self.gate.expensive.get('prepare'), 1)
        self.assertEqual(self.gate.mutations, 1)
        self.client.get('/api/runtime', headers=self.headers)
        self.client.get('/api/researches', headers=self.headers)
        self.assertEqual(self.gate.expensive['prepare'], 1)
        self.assertEqual(self.gate.mutations, 1)

    def test_research_retry_uses_the_saved_operation_kind(self):
        self.seed_operation('answer')
        self.gate.expensive['answer'] = 10
        with patch('backend.app.api.retry_research') as submit:
            response = self.client.post('/api/researches/research/retry',
                json={'operation_id': 'operation'}, headers=self.headers)
        self.assertEqual(response.status_code, 429)
        submit.assert_not_called()

    def test_preparation_retry_uses_the_same_budget_as_legacy_index(self):
        self.seed_operation('prepare')
        self.gate.expensive['prepare'] = 10
        with patch('backend.app.api.retry_research') as submit:
            response = self.client.post('/api/researches/research/retry',
                json={'operation_id': 'operation'}, headers=self.headers)
        self.assertEqual(response.status_code, 429)
        submit.assert_not_called()

    def test_legacy_job_retry_normalizes_import_and_index_to_preparation(self):
        engine = open_database(self.database, create_schema=False)
        with engine.begin() as connection:
            for kind in ('import', 'index'):
                connection.execute(text("INSERT INTO jobs (id, kind, resource_id, status, idempotency_key, request_hash, created_at, updated_at) VALUES (:kind, :kind, :kind, 'failed', :kind, 'hash', 'now', 'now')"), {'kind': kind})
        engine.dispose()
        self.gate.expensive['prepare'] = 10
        for kind in ('import', 'index'):
            with self.subTest(kind=kind):
                response = self.client.post('/api/jobs/' + kind + '/retry',
                    json={'expected_attempt_id': 'old'}, headers=self.headers)
                self.assertEqual(response.status_code, 429)

    def test_retry_lookup_failure_does_not_admit_an_unmetered_submission(self):
        self.gate.database_path = self.root / 'missing.sqlite'
        with patch('backend.app.api.retry_research') as submit:
            response = self.client.post('/api/researches/research/retry',
                json={'operation_id': 'operation'}, headers=self.headers)
        self.assertEqual(response.status_code, 503)
        submit.assert_not_called()
        self.assertFalse(self.gate.database_path.exists())

    def test_screening_budget_rejects_before_any_model_request(self):
        self.gate.expensive['screening'] = 10
        with patch('backend.app.api.generate_screening', new_callable=AsyncMock) as model:
            response = self.client.post('/api/searches/search/screening',
                json={}, headers=self.headers)
        self.assertEqual(response.status_code, 429)
        model.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
