import json
import asyncio
from pathlib import Path
import tempfile
import unittest

import httpx
from backend.app.search import screening

CONFIG = {'DEEPSEEK_MODEL': 'test', 'DEEPSEEK_BASE_URL': 'https://example.test', 'DEEPSEEK_API_KEY': 'never-record'}


def search(count=2):
    return {'search_id': 'a' * 32, 'status': 'succeeded', 'question': '有哪些知识图谱方法？',
        'snapshot_sha256': 'b' * 64, 'candidates': [
            {'versioned_id': f'2501.{n:05d}v1', 'display_rank': n,
             'title': f'Paper {n}', 'abstract': 'We investigate awide range of approaches. We study KG- enhanced tasks.'}
            for n in range(1, count + 1)]}


def response_for(payload, invalid=None):
    rows = [{'paper_id': p['paper_id'], 'label': 'high', 'reason': '提供知识图谱方法的依据。',
        'evidence_span_id': 'nonexistent' if p['paper_id'] == invalid else
            (p.get('abstract_spans') or [{'id': 'a1'}])[0]['id']} for p in payload['papers']]
    result = {'papers': rows}
    if not payload.get('criteria'):
        result.update(interpretation='按方法及应用核对知识图谱研究。', criteria=['对象与知识图谱有关', '给出具体方法或应用'])
    return result


class ScreeningSpanTest(unittest.IsolatedAsyncioTestCase):
    async def test_server_recovers_exact_saved_excerpt_without_model_copying(self):
        calls = []
        def upstream(request):
            payload = json.loads(json.loads(request.content)['messages'][1]['content']); calls.append(payload)
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(response_for(payload))}}]})
        with tempfile.TemporaryDirectory() as root:
            async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
                result = await screening.generate_screening(search(), root, config=CONFIG, client=client)
            self.assertEqual(len(result['papers']), 2)
            paper = result['papers'][0]
            self.assertIn(paper['evidence_quote'], search()['candidates'][0]['abstract'])
            self.assertEqual(paper['evidence_quote'], search()['candidates'][0]['abstract'][paper['evidence_span']['start']:paper['evidence_span']['end']])
            self.assertIn('awide', calls[0]['papers'][0]['abstract_spans'][0]['text'])
            self.assertEqual(screening.read_screening(search(), root)['screening_id'], result['screening_id'])

    async def test_one_invalid_paper_does_not_discard_verified_paper_and_retry_is_explicit(self):
        calls = []
        def upstream(request):
            payload = json.loads(json.loads(request.content)['messages'][1]['content']); calls.append(payload)
            output = response_for(payload, '2501.00002v1' if len(calls) == 1 else None)
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(output)}}]})
        with tempfile.TemporaryDirectory() as root:
            async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
                partial = await screening.generate_screening(search(), root, config=CONFIG, client=client)
                self.assertEqual(partial['status'], 'partial')
                self.assertEqual([p['paper_id'] for p in partial['papers']], ['2501.00001v1'])
                self.assertEqual([p['paper_id'] for p in partial['unverified_papers']], ['2501.00002v1'])
                self.assertEqual(len(calls), 1)
                attempts = [json.loads(p.read_text()) for p in (Path(root) / 'abstract-screenings/attempts').glob('*.json')]
                self.assertEqual(attempts[0]['status'], 'failed')
                self.assertIn('model_output', attempts[0])
                self.assertNotIn(CONFIG['DEEPSEEK_API_KEY'], json.dumps(attempts))
                self.assertEqual(screening.read_screening(search(), root)['status'], 'partial')
                complete = await screening.generate_screening(search(), root, config=CONFIG, client=client)
                self.assertEqual(complete['status'], 'succeeded')
                self.assertEqual([p['paper_id'] for p in calls[1]['papers']], ['2501.00002v1'])
                self.assertEqual(calls[1]['criteria'], partial['criteria'])
                again = await screening.generate_screening(search(), root, config=CONFIG, client=client)
                self.assertTrue(again['cache_hit']); self.assertEqual(len(calls), 2)

    async def test_batches_share_criteria_and_preserve_all_paper_identities(self):
        calls = []
        def upstream(request):
            payload = json.loads(json.loads(request.content)['messages'][1]['content']); calls.append(payload)
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(response_for(payload))}}]})
        with tempfile.TemporaryDirectory() as root:
            async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
                result = await screening.generate_screening(search(21), root, config=CONFIG, client=client)
            self.assertEqual([len(c['papers']) for c in calls], [10, 10, 1])
            self.assertEqual(calls[1]['criteria'], result['criteria'])
            self.assertEqual(calls[2]['criteria'], result['criteria'])
            self.assertEqual([p['paper_id'] for p in result['papers']], [p['versioned_id'] for p in search(21)['candidates']])

    async def test_truncated_middle_batch_keeps_both_verified_batches_and_retries_only_the_gap(self):
        calls = []
        def upstream(request):
            payload = json.loads(json.loads(request.content)['messages'][1]['content']); calls.append(payload)
            return httpx.Response(200, json={'choices': [{'finish_reason': 'length' if len(calls) == 2 else 'stop',
                'message': {'content': json.dumps(response_for(payload))}}]})
        with tempfile.TemporaryDirectory() as root:
            async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
                result = await screening.generate_screening(search(30), root, config=CONFIG, client=client)
                self.assertEqual(result['status'], 'partial')
                self.assertEqual(len(result['papers']), 20)
                missing = [p['paper_id'] for p in result['unverified_papers']]
                self.assertEqual(missing, [p['versioned_id'] for p in search(30)['candidates'][10:20]])
                self.assertEqual(len(calls), 3)
                self.assertEqual(screening.read_screening(search(30), root)['status'], 'partial')
                complete = await screening.generate_screening(search(30), root, config=CONFIG, client=client)
                self.assertEqual(complete['status'], 'succeeded')
                self.assertEqual([p['paper_id'] for p in calls[3]['papers']], missing)
                self.assertEqual(calls[3]['criteria'], result['criteria'])

    async def test_cancellation_after_a_saved_batch_resumes_without_repeating_verified_papers(self):
        calls = []
        def upstream(request):
            payload = json.loads(json.loads(request.content)['messages'][1]['content']); calls.append(payload)
            if len(calls) == 2: raise asyncio.CancelledError()
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(response_for(payload))}}]})
        with tempfile.TemporaryDirectory() as root:
            async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
                with self.assertRaises(asyncio.CancelledError):
                    await screening.generate_screening(search(20), root, config=CONFIG, client=client)
                partial = screening.read_screening(search(20), root)
                self.assertEqual(len(partial['papers']), 10)
                self.assertEqual(partial['status'], 'partial')
                complete = await screening.generate_screening(search(20), root, config=CONFIG, client=client)
                self.assertEqual(complete['status'], 'succeeded')
                self.assertEqual([p['paper_id'] for p in calls[2]['papers']], [p['versioned_id'] for p in search(20)['candidates'][10:]])

    async def test_saved_excerpt_offset_corruption_is_rejected_without_rewriting_the_record(self):
        def upstream(request):
            payload = json.loads(json.loads(request.content)['messages'][1]['content'])
            return httpx.Response(200, json={'choices': [{'finish_reason':'stop','message':{'content':json.dumps(response_for(payload))}}]})
        with tempfile.TemporaryDirectory() as root:
            async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
                result = await screening.generate_screening(search(), root, config=CONFIG, client=client)
            path = Path(root) / 'abstract-screenings' / (result['input_hash'] + '.json')
            altered = json.loads(path.read_text(encoding='utf-8'))
            altered['papers'][0]['evidence_span']['start'] += 1
            path.write_text(json.dumps(altered), encoding='utf-8'); before = path.read_bytes()
            with self.assertRaises(screening.ScreeningError) as caught:
                screening.read_screening(search(), root)
            self.assertEqual(caught.exception.kind, 'invalid_record')
            self.assertEqual(path.read_bytes(), before)

    async def test_reading_a_legacy_success_does_not_call_or_rewrite_it(self):
        from tests.test_abstract_screening import saved_search, model_output
        source = saved_search(); inputs, digest = screening._input(source, legacy=True)
        # Legacy identity remains available even after the active prompt changes.
        inputs['prompt_version'] = 'abstract-relevance-v1'
        inputs['prompt_sha256'] = getattr(screening, 'LEGACY_PROMPT_SHA256', screening.PROMPT_SHA256)
        import hashlib
        digest = hashlib.sha256(json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        record = {**inputs, **model_output(source), 'screening_id': 'c' * 32, 'input_hash': digest,
            'status': 'succeeded', 'model': 'test', 'candidate_count': 2, 'validation_version': 'abstract-relevance-validation-v2'}
        # Inputs store the original candidates; the successful papers are assessments.
        record['papers'] = model_output(source)['papers']
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'abstract-screenings' / f'{digest}.json'; path.parent.mkdir(); path.write_text(json.dumps(record), encoding='utf-8')
            before = path.read_bytes()
            self.assertEqual(screening.read_screening(source, root)['screening_id'], record['screening_id'])
            self.assertEqual(path.read_bytes(), before)
