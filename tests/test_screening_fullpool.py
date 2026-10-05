import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import httpx
from backend.app.search import screening, screening_spans
from tests.test_abstract_screening import CONFIG, saved_search
from tests.test_screening_spans import response_for


def pool(count=60, long=False):
    value = saved_search()
    value['candidates'] = [{**value['candidates'][0], 'versioned_id': f'2501.{i:05d}v1',
        'display_rank': i, 'abstract': ('long word ' * 1500 if long else value['candidates'][0]['abstract'])}
        for i in range(1, count + 1)]
    value['fusion'] = {'method': 'rrf', 'candidate_count': count}
    return value


def old_record(root, value, *, v1=False, partial=False):
    papers = [{'paper_id': p['versioned_id'], 'title': p['title'], 'abstract': p['abstract']}
        for p in value['candidates'][:30]]
    inputs = {'search_id': value['search_id'], 'snapshot_sha256': value['snapshot_sha256'],
        'question': value['question'], 'papers': papers,
        'prompt_version': screening.LEGACY_PROMPT_VERSION if v1 else screening_spans.PROMPT_VERSION,
        'prompt_sha256': screening.LEGACY_PROMPT_SHA256 if v1 else hashlib.sha256(screening_spans.SYSTEM_PROMPT.encode()).hexdigest(),
        'screening_scope': {'method': 'rrf_top', 'limit': 30, 'source_candidate_count': len(value['candidates']),
            'paper_ids': [p['paper_id'] for p in papers]}}
    digest = hashlib.sha256(json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    rows = [{'paper_id': p['paper_id'], 'label': 'high', 'reason': '提供实际评测依据。',
        'evidence_quote': p['abstract']} for p in papers]
    if not v1:
        for row in rows: row['evidence_span'] = {'id': 'a1', 'start': 0, 'end': len(row['evidence_quote'])}
    missing = []
    if partial:
        missing = [{'paper_id': rows.pop()['paper_id'], 'kind': 'unverified', 'message': '尚未核实'}]
    record = {k: inputs[k] for k in ('search_id', 'snapshot_sha256', 'question', 'prompt_version', 'prompt_sha256', 'screening_scope')}
    record.update(input_hash=digest, screening_id='c'*32, validation_version='abstract-relevance-validation-v2' if v1 else 'abstract-relevance-spans-validation-v3',
        candidate_count=30, status='partial' if missing else 'succeeded', papers=rows,
        interpretation='核对编码代理的实际评测。', criteria=['研究对象一致', '提供实证依据'], model='old-model',
        created_at='2026-10-02T00:00:00Z', elapsed_ms=100, usage={'total_tokens': 100}, cost_usd=None)
    if not v1: record['unverified_papers'] = missing
    folder = Path(root)/'abstract-screenings'; folder.mkdir(exist_ok=True)
    path = folder/f'{digest}.json'; path.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
    return path


class FullPoolScreeningTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.calls = []

    async def generate(self, value, fail_batch=None, long_criteria=False):
        def respond(request):
            data = json.loads(json.loads(request.content)['messages'][1]['content']); self.calls.append(data)
            output = response_for(data)
            if long_criteria and not data.get('criteria'):
                output.update(criteria=['\x00'*160]*4, interpretation='\x00'*400)
            return httpx.Response(200, json={'choices':[{'finish_reason':'length' if len(self.calls)==fail_batch else 'stop',
                'message':{'content':json.dumps(output)}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await screening.generate_screening(value, self.root, config=CONFIG, client=client)

    async def test_all_sixty_and_read_only_plan_complete_without_resubmitting(self):
        value = pool()
        plan = screening.screening_plan(value, None)
        self.assertEqual((plan['total_count'],plan['verified_count'],plan['remaining_count'],plan['batch_count'],plan['wait_timeout_ms']), (60,0,60,6,375000))
        result = await self.generate(value)
        self.assertEqual(result['candidate_count'],60)
        self.assertEqual(result['screening_scope']['method'],'all_candidates')
        self.assertEqual([p['paper_id'] for p in result['papers']], [f'2501.{i:05d}v1' for i in range(1,61)])
        self.assertEqual([len(c['papers']) for c in self.calls], [10]*6)
        reread = screening.read_screening(value,self.root)
        self.assertEqual(screening.screening_plan(value,reread)['remaining_count'],0)
        await self.generate(value)
        self.assertEqual(len(self.calls),6)

    async def test_long_batches_fit_with_worst_shared_criteria_without_truncating(self):
        value = pool(long=True)
        plan = screening.screening_plan(value,None)
        result = await self.generate(value,long_criteria=True)
        self.assertEqual(len(self.calls),plan['batch_count'])
        self.assertGreater(len(self.calls),6)
        self.assertEqual(len(result['papers']),60)
        restored = {}
        for batch in self.calls:
            self.assertLessEqual(len(batch['papers']),10)
            self.assertLessEqual(len(json.dumps(batch,ensure_ascii=False)),120000)
            for paper in batch['papers']: restored[paper['paper_id']] = ''.join(s['text'] for s in paper['abstract_spans'])
        self.assertEqual(restored, {p['versioned_id']:p['abstract'] for p in value['candidates']})

    async def test_middle_failure_preserves_other_fifty_and_retry_only_missing_ten(self):
        value=pool(); result=await self.generate(value,fail_batch=3)
        self.assertEqual((result['status'],len(result['papers'])),('partial',50))
        self.assertEqual(screening.screening_plan(value,result)['batch_count'],1)
        complete=await self.generate(value)
        self.assertEqual([p['paper_id'] for p in self.calls[-1]['papers']], [f'2501.{i:05d}v1' for i in range(21,31)])
        self.assertEqual((complete['status'],len(complete['papers'])),('succeeded',60))

    async def test_both_old_protocols_reuse_verified_rows_and_never_overwrite_source(self):
        for v1 in (False,True):
            with self.subTest(v1=v1), tempfile.TemporaryDirectory() as root:
                self.root=Path(root); self.calls=[]; value=pool()
                old=old_record(self.root,value,v1=v1); before=old.read_bytes()
                cached=screening.read_screening(value,self.root)
                self.assertEqual(screening.screening_plan(value,cached)['remaining_count'],30)
                result=await self.generate(value)
                self.assertEqual([p['paper_id'] for c in self.calls for p in c['papers']], [f'2501.{i:05d}v1' for i in range(31,61)])
                self.assertEqual(old.read_bytes(),before)
                self.assertEqual(result['reused_screening']['input_hash'], old.stem)
                self.assertEqual(result['reused_screening']['artifact_sha256'],hashlib.sha256(before).hexdigest())
                self.assertEqual(len(screening.read_screening(value,self.root)['papers']),60)
                damaged=json.loads(old.read_text(encoding='utf-8')); damaged['papers'][0]['label']='partial'
                old.write_text(json.dumps(damaged,ensure_ascii=False),encoding='utf-8')
                with self.assertRaises(screening.ScreeningError): screening.read_screening(value,self.root)
                with self.assertRaises(screening.ScreeningError): await self.generate(value)
                self.assertEqual(len(self.calls),3)

    async def test_old_partial_reuses_only_verified_and_inherits_shared_criteria(self):
        value=pool(); old=old_record(self.root,value,partial=True); before=old.read_bytes()
        result=await self.generate(value)
        self.assertEqual([p['paper_id'] for c in self.calls for p in c['papers']], [f'2501.{i:05d}v1' for i in range(30,61)])
        self.assertTrue(all(c['criteria']==['研究对象一致','提供实证依据'] for c in self.calls))
        self.assertEqual(len(result['papers']),60); self.assertEqual(old.read_bytes(),before)

    async def test_cancel_and_overlapping_submit_do_not_repeat_finished_batches(self):
        value=pool(); entered=asyncio.Event(); release=asyncio.Event()
        async def respond(request):
            data=json.loads(json.loads(request.content)['messages'][1]['content']); self.calls.append(data)
            if len(self.calls)==2:
                entered.set(); await release.wait()
            return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(response_for(data))}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            task=asyncio.create_task(screening.generate_screening(value,self.root,config=CONFIG,client=client))
            await asyncio.wait_for(entered.wait(),3)
            with self.assertRaises(screening.ScreeningError): await screening.generate_screening(value,self.root,config=CONFIG,client=client)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
        saved=screening.read_screening(value,self.root)
        self.assertEqual(screening.screening_plan(value,saved)['verified_count'],10)
        self.assertEqual(len(saved['papers']),10)
        result=await self.generate(value)
        self.assertEqual(self.calls[2]['papers'][0]['paper_id'],'2501.00011v1')
        self.assertEqual(len(result['papers']),60)

    async def test_new_scope_and_inherited_rows_cannot_be_tampered(self):
        value=pool(); old_record(self.root,value); result=await self.generate(value)
        path=self.root/'abstract-screenings'/f"{result['input_hash']}.json"
        original=path.read_bytes()
        for kind in ('scope','inherited-row','criteria','source-id'):
            damaged=json.loads(original)
            if kind=='scope': damaged['screening_scope']['paper_ids'][0]='outside'
            elif kind=='inherited-row': damaged['papers'][0]['label']='low'
            elif kind=='criteria': damaged['criteria']=['不同标准','换了问题']
            else: damaged['reused_screening']['input_hash']='d'*64
            path.write_text(json.dumps(damaged,ensure_ascii=False),encoding='utf-8')
            with self.subTest(kind=kind),self.assertRaises(screening.ScreeningError): screening.read_screening(value,self.root)
            path.write_bytes(original)

    async def test_removing_provenance_cannot_make_inherited_v2_rows_native(self):
        value = pool(); old = old_record(self.root, value)
        result = await self.generate(value, fail_batch=1)
        path = self.root/'abstract-screenings'/f"{result['input_hash']}.json"
        original = path.read_bytes(); old.unlink()
        with self.assertRaises(screening.ScreeningError): screening.read_screening(value, self.root)
        for mutation in ('missing', 'null'):
            damaged = json.loads(original)
            if mutation == 'missing': damaged.pop('reused_screening')
            else: damaged['reused_screening'] = None
            path.write_text(json.dumps(damaged, ensure_ascii=False), encoding='utf-8')
            with self.subTest(mutation=mutation):
                with self.assertRaises(screening.ScreeningError): screening.read_screening(value, self.root)
                with self.assertRaises(screening.ScreeningError): await self.generate(value)
        self.assertEqual(len(self.calls), 3)

    async def test_native_records_explicitly_have_no_reused_source(self):
        result = await self.generate(pool())
        self.assertTrue('reused_screening' in result, 'Native records must explicitly identify no reused source.')
        self.assertIsNone(result['reused_screening'])

    async def test_mutating_a_valid_native_label_is_detected_before_cache_reuse(self):
        value = pool(); result = await self.generate(value)
        path = self.root/'abstract-screenings'/f"{result['input_hash']}.json"
        damaged = json.loads(path.read_text(encoding='utf-8'))
        damaged['papers'][0]['label'] = 'partial'
        path.write_text(json.dumps(damaged, ensure_ascii=False), encoding='utf-8')
        with self.assertRaises(screening.ScreeningError): screening.read_screening(value, self.root)
        with self.assertRaises(screening.ScreeningError): await self.generate(value)
        self.assertEqual(len(self.calls), 6)


if __name__=='__main__': unittest.main()
