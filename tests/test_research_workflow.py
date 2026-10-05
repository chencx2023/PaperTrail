"""Authorized research preparation persists across reads and worker turns."""
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from datetime import datetime, timezone
from unittest.mock import patch
import httpx
from sqlalchemy import select, func, event
from sqlalchemy.orm import Session
from backend.app.jobs.dispatcher import run_one_job
from backend.app.models import Job, CollectionRevision
from backend.app.search.client import ArxivClient
import tests.test_api_revision as revision_fixture
from tests.test_api_revision import AtomTransport
from tests.test_collection_index import FakeEmbedding, settings
from tests.test_document_preparation import text_pdf_bytes
from tests.test_qa_execution import offline_answer


class ResearchWorkflowTest(unittest.TestCase):
    setUp = revision_fixture.ApiRevisionTest.setUp

    def searched(self):
        response = self.client.post('/api/searches', headers={'Idempotency-Key': 'research-search'},
            json={'question': '离线研究测试', 'query_text': 'LLM agent memory'})
        self.assertEqual(response.status_code, 202, response.text)
        data = response.json()
        self.assertIn('research_id', data)
        run_one_job(self.engine, worker_id='search', data_root=self.data_root,
            search_client=ArxivClient(transport=AtomTransport(), clock=lambda: datetime(2026, 9, 25, tzinfo=timezone.utc)))
        return data['research_id']

    def record(self, id):
        response = self.client.get(f'/api/researches/{id}')
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def prepare(self, id, versions=('2401.00001v1', '2401.00001v3'), key='prepare'):
        response = self.client.post(f'/api/researches/{id}/prepare', headers={'Idempotency-Key': key},
            json={'versioned_ids': list(versions), 'expected_pool_revision_id': self.record(id)['pool_revision_id']})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()

    def tick(self, *, failed_versions=(), embedding=None, executor=None):
        def download(request):
            if any(request.url.path.endswith(v) for v in failed_versions): return httpx.Response(404)
            return httpx.Response(200, headers={'Content-Type': 'application/pdf'}, content=text_pdf_bytes())
        with httpx.Client(transport=httpx.MockTransport(download)) as client:
            return run_one_job(self.engine, worker_id='research', data_root=self.data_root,
                import_client=client, index_settings=settings(), index_embedding_model=embedding or FakeEmbedding(),
                answer_settings=settings(), answer_executor=executor or (lambda scope, _question: offline_answer(scope)))

    def test_partial_import_automatically_indexes_successes_once_and_reads_never_queue(self):
        id = self.searched()
        first = self.prepare(id)
        self.assertEqual(self.prepare(id)['operation']['operation_id'], first['operation']['operation_id'])
        self.tick(failed_versions=('2401.00001v3',))
        self.tick()
        self.tick()
        record = self.record(id)
        self.assertEqual(record['operation']['status'], 'succeeded')
        self.assertEqual(len(record['papers']), 1)
        self.assertEqual(record['papers'][0]['versioned_id'], '2401.00001v1')
        self.assertNotEqual(record['import_report']['items'][1]['status'], 'succeeded')
        with Session(self.engine) as s:
            before = s.scalar(select(func.count()).select_from(Job))
            self.assertEqual(s.scalar(select(func.count()).select_from(Job).where(Job.kind == 'index')), 1)
        for _ in range(3): self.record(id)
        self.tick()
        with Session(self.engine) as s: self.assertEqual(s.scalar(select(func.count()).select_from(Job)), before)

    def test_all_failed_stops_without_revision_or_index_and_explicit_retry_recovers(self):
        id = self.searched()
        self.prepare(id)
        self.tick(failed_versions=('2401.00001v1', '2401.00001v3'))
        self.tick()
        record = self.record(id)
        self.assertEqual(record['operation']['status'], 'failed')
        self.assertFalse(record['papers'])
        with Session(self.engine) as s: self.assertEqual(s.scalar(select(func.count()).select_from(CollectionRevision)), 0)
        retry = self.client.post(f'/api/researches/{id}/retry', headers={'Idempotency-Key': 'retry'},
            json={'operation_id': record['operation']['operation_id']})
        self.assertEqual(retry.status_code, 202, retry.text)
        self.tick(); self.tick(); self.tick()
        self.assertEqual(len(self.record(id)['papers']), 2)

    def test_subset_index_reuses_cached_vectors_and_freezes_actual_scope(self):
        id = self.searched()
        self.prepare(id)
        self.tick(); self.tick(); self.tick()
        pool = self.record(id)
        selected = [pool['papers'][1]['document_id']]
        response = self.client.post(f'/api/researches/{id}/answers', headers={'Idempotency-Key': 'qa'},
            json={'question': 'What is studied?', 'document_ids': selected,
                  'expected_pool_revision_id': pool['pool_revision_id']})
        self.assertEqual(response.status_code, 202, response.text)
        embedding = FakeEmbedding(fail=True)
        scopes = []
        async def answer(scope, question):
            scopes.append([doc.document_id for doc in scope.documents])
            return await offline_answer(scope)
        self.tick(embedding=embedding, executor=answer)
        self.tick(embedding=embedding, executor=answer)
        self.tick(embedding=embedding, executor=answer)
        saved = self.record(id)
        self.assertEqual(saved['operation']['status'], 'succeeded')
        self.assertEqual(scopes, [selected])
        self.assertEqual(embedding.calls, 0)
        self.assertNotEqual(saved['scope_revision_id'], saved['pool_revision_id'])
        self.assertEqual(len(saved['papers']), 2)
        self.assertTrue(saved['answer_run_id'])

    def test_index_failure_is_persisted_before_worker_exits_and_explicit_retry_recovers(self):
        id = self.searched(); self.prepare(id); self.tick()
        with self.assertRaises(Exception): self.tick(embedding=FakeEmbedding(fail=True))
        record = self.record(id)
        self.assertEqual(record['operation']['status'], 'failed')
        response = self.client.post(f'/api/researches/{id}/answers', headers={'Idempotency-Key': 'not-ready'},
            json={'question': 'What is studied?', 'document_ids': [record['papers'][0]['document_id']],
                  'expected_pool_revision_id': record['pool_revision_id']})
        self.assertEqual(response.status_code, 409)
        retried = self.client.post(f'/api/researches/{id}/retry', headers={'Idempotency-Key': 'index-retry'},
            json={'operation_id': record['operation']['operation_id']})
        self.assertEqual(retried.status_code, 202, retried.text)
        self.tick(); self.tick(); self.tick()
        self.assertEqual(self.record(id)['operation']['status'], 'succeeded')

    def test_stale_pool_unknown_documents_and_changed_idempotency_are_rejected(self):
        id = self.searched(); self.prepare(id); self.tick(); self.tick(); self.tick()
        record = self.record(id)
        for pool, ids, expected in [('stale-pool', [record['papers'][0]['document_id']], 409),
                (record['pool_revision_id'], ['foreign-document'], 400)]:
            response = self.client.post(f'/api/researches/{id}/answers', headers={'Idempotency-Key': f'invalid-{expected}'},
                json={'question': 'What is studied?', 'document_ids': ids, 'expected_pool_revision_id': pool})
            self.assertEqual(response.status_code, expected, response.text)
        changed = self.client.post(f'/api/researches/{id}/prepare', headers={'Idempotency-Key': 'prepare'},
            json={'versioned_ids': ['2401.00001v1'], 'expected_pool_revision_id': record['pool_revision_id']})
        self.assertEqual(changed.status_code, 409)

    def test_worker_restart_continues_persisted_intent_without_duplicate_import(self):
        from backend.app.db import open_database
        id = self.searched(); self.prepare(id); self.tick()
        self.engine.dispose(); self.engine = open_database(self.database_path, create_schema=False)
        self.addCleanup(self.engine.dispose)
        self.tick(); self.tick()
        self.assertEqual(self.record(id)['operation']['status'], 'succeeded')
        with Session(self.engine) as s:
            self.assertEqual(s.scalar(select(func.count()).select_from(Job).where(Job.kind == 'import')), 1)

    def test_first_answer_fills_record_and_followup_is_new_with_full_pool_and_custom_title_preserved(self):
        id = self.searched(); self.prepare(id); self.tick(); self.tick(); self.tick()
        pool = self.record(id)
        renamed = self.client.patch(f'/api/researches/{id}', json={'title': '我的论文笔记'})
        self.assertEqual(renamed.status_code, 200)
        def ask(target, key, docs):
            response = self.client.post(f'/api/researches/{target}/answers', headers={'Idempotency-Key': key},
                json={'question': 'What is studied?', 'document_ids': docs, 'expected_pool_revision_id': pool['pool_revision_id']})
            self.assertEqual(response.status_code, 202, response.text)
            return response.json()
        first = ask(id, 'first-answer', [pool['papers'][0]['document_id']])
        self.assertEqual(first['research_id'], id)
        self.assertEqual(first['title'], '我的论文笔记')
        self.tick(); self.tick(); self.tick()
        saved = self.record(id)
        followup = ask(id, 'followup-answer', [pool['papers'][1]['document_id']])
        self.assertNotEqual(followup['research_id'], id)
        self.assertEqual(followup['parent_id'], id)
        self.assertEqual(len(followup['papers']), 2)
        self.assertEqual(followup['title'], 'What is studied?')
        self.assertEqual(ask(id, 'followup-answer', [pool['papers'][1]['document_id']])['research_id'], followup['research_id'])
        self.tick(); self.tick(); self.tick()
        self.assertEqual(self.record(id)['answer_run_id'], saved['answer_run_id'])
        self.assertEqual(self.record(id)['title'], '我的论文笔记')
        history = self.client.get('/api/researches').json()
        self.assertEqual(len(history['items']), 2)
        for _ in range(2):
            self.record(id); self.client.get('/api/researches')
        self.assertEqual(len(self.client.get('/api/researches').json()['items']), 2)

    def test_legacy_answer_submission_resolves_immediately_without_service_restart(self):
        id = self.searched(); self.prepare(id); self.tick(); self.tick(); self.tick()
        pool = self.record(id)
        submitted = self.client.post('/api/answers', headers={'Idempotency-Key': 'legacy-qa'},
            json={'question': 'What is studied?', 'revision_id': pool['pool_revision_id'],
                  'index_artifact_id': pool['preparation']['index_artifact_id']})
        self.assertEqual(submitted.status_code, 202, submitted.text)
        answer_id = submitted.json()['answer_run_id']
        resolved = self.client.get(f'/api/researches/resolve?answer={answer_id}').json()['research_id']
        self.assertIsNotNone(resolved)
        self.assertEqual(self.record(resolved)['answer_run_id'], answer_id)

    def test_simultaneous_same_prepare_key_replays_one_operation(self):
        id = self.searched()
        barrier = Barrier(2); lock = Lock(); seen = 0
        def synchronized_first_reads(_connection, _cursor, sql, parameters, _context, _many):
            nonlocal seen
            if 'research_operations.idempotency_key =' not in sql or 'parallel-prepare' not in parameters: return
            with lock:
                seen += 1; first_pair = seen <= 2
            if first_pair: barrier.wait(timeout=5)
        # API requests open their own engines; coordinate the empty key read
        # across connections rather than relying on thread scheduling luck.
        from sqlalchemy.engine import Engine
        event.listen(Engine, 'after_cursor_execute', synchronized_first_reads)
        try:
            def prepare(): return self.client.post(f'/api/researches/{id}/prepare',
                headers={'Idempotency-Key': 'parallel-prepare'},
                json={'versioned_ids': ['2401.00001v1'], 'expected_pool_revision_id': None})
            with ThreadPoolExecutor(max_workers=2) as workers:
                responses = list(workers.map(lambda _: prepare(), range(2)))
            self.assertEqual([r.status_code for r in responses], [202, 202], [r.text for r in responses])
            self.assertEqual(responses[0].json()['operation']['operation_id'], responses[1].json()['operation']['operation_id'])
        finally: event.remove(Engine, 'after_cursor_execute', synchronized_first_reads)

    def test_legacy_subset_answer_keeps_the_existing_complete_prepared_pool(self):
        from backend.app.research.workflow import freeze_answer_subset, _scope_index
        from backend.app.collections.revision_submission import read_revision_state
        id = self.searched(); self.prepare(id); self.tick(); self.tick(); self.tick()
        pool = self.record(id)
        subset = freeze_answer_subset(self.engine, pool['pool_revision_id'], [pool['papers'][0]['document_id']])
        _scope_index(self.engine, subset, 'legacy-subset-index', settings()); self.tick()
        scope = read_revision_state(self.engine, pool['collection_id'], subset)
        response = self.client.post('/api/answers', headers={'Idempotency-Key': 'legacy-subset-qa'},
            json={'question': 'What is studied?', 'revision_id': subset, 'index_artifact_id': scope['index_artifact_id']})
        self.assertEqual(response.status_code, 202, response.text)
        saved = self.record(id)
        self.assertEqual(len(saved['papers']), 2)
        self.assertEqual(saved['pool_revision_id'], pool['pool_revision_id'])
        self.assertEqual(saved['scope_revision_id'], subset)
        self.assertEqual(len(self.client.get('/api/researches').json()['items']), 1)

    def test_crash_after_answer_enqueue_is_not_migrated_into_duplicate_history(self):
        import backend.app.research.workflow as workflow
        from backend.app.research.records import backfill_research_records
        id = self.searched(); self.prepare(id); self.tick(); self.tick(); self.tick()
        pool = self.record(id)
        response = self.client.post(f'/api/researches/{id}/answers', headers={'Idempotency-Key': 'crash-answer'},
            json={'question': 'What is studied?', 'document_ids': [p['document_id'] for p in pool['papers']],
                  'expected_pool_revision_id': pool['pool_revision_id']})
        self.assertEqual(response.status_code, 202)
        workflow.advance_research_operations(self.engine, data_root=self.data_root, index_settings=settings())
        progress = workflow._progress
        def crash_before_binding(*args, **kwargs):
            if 'answer_run_id' in (kwargs.get('record_values') or {}): raise RuntimeError('simulated process crash')
            return progress(*args, **kwargs)
        with patch.object(workflow, '_progress', crash_before_binding):
            with self.assertRaises(RuntimeError): workflow.advance_research_operations(self.engine, data_root=self.data_root, index_settings=settings())
        backfill_research_records(self.engine)
        self.assertEqual(len(self.client.get('/api/researches').json()['items']), 1)
        self.tick(); self.tick()
        self.assertTrue(self.record(id)['answer_run_id'])
        self.assertEqual(len(self.client.get('/api/researches').json()['items']), 1)

    def test_singleton_takeover_interrupts_expired_research_attempt_without_auto_retry(self):
        from backend.app.research.workflow import advance_research_operations, recover_orphan_research_attempts
        from backend.app.jobs.claim import claim_next_job
        from backend.app.jobs.process_lock import worker_singleton
        from backend.app.models import JobAttempt
        id = self.searched(); self.prepare(id)
        advance_research_operations(self.engine, data_root=self.data_root, index_settings=settings())
        claimed = claim_next_job(self.engine, worker_id='dead-worker', supported_kinds=('import',))
        with Session(self.engine) as session:
            attempt = session.get(JobAttempt, claimed.attempt_id); attempt.lease_until = '2099-01-01T00:00:00Z'; session.commit()
        with worker_singleton(self.database_path) as acquired:
            self.assertTrue(acquired)
            self.assertEqual(recover_orphan_research_attempts(self.engine, worker_id='replacement', singleton_acquired=acquired), [])
            with Session(self.engine) as session:
                session.get(JobAttempt, claimed.attempt_id).lease_until = '2000-01-01T00:00:00Z'; session.commit()
            self.assertEqual(recover_orphan_research_attempts(self.engine, worker_id='replacement', singleton_acquired=acquired), [claimed.attempt_id])
        self.tick()
        stopped = self.record(id)
        self.assertEqual(stopped['operation']['status'], 'failed')
        with Session(self.engine) as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(Job).where(Job.kind == 'import')), 1)
        response = self.client.post(f'/api/researches/{id}/retry', headers={'Idempotency-Key': 'after-takeover'}, json={'operation_id': stopped['operation']['operation_id']})
        self.assertEqual(response.status_code, 202)
        self.tick(); self.tick()
        self.assertEqual(self.record(id)['operation']['status'], 'succeeded')
