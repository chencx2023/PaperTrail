"""Source binding, review ordering and ordinary-version isolation; no API calls."""
import copy
import json
import unittest
from scripts.model_config import build_settings
from backend.app.qa.citation_review import source_first_payload
from backend.app.qa.compact import compact_policy_identity, generate_compact_grounded_answer
from backend.app.qa.grounded import source_units
from tests.test_compact_activation import CONFIG
from tests.test_grounded_generation import FakeModel, packed_context, support_check

TEXT = 'The method is evaluated in one trial in region A.'
STATEMENT = 'The result concerns one trial in region A.'


def materials():
    units = source_units(packed_context(a=TEXT))
    payload = {'question': 'What is supported?', 'program_filled_original_units': units,
        'target_facts': [{'fact_id': 'p1f1', 'part_id': 'p1', 'original': {
            'statement': STATEMENT, 'scope': 'experimental_result', 'sentence_ids': ['s0001']}, 'protected': True}]}
    return payload, units


class CitationScopeReviewTest(unittest.TestCase):
    def test_actual_sources_precede_draft_without_changing_sources_or_claim(self):
        payload, units = materials(); before = copy.deepcopy(payload)
        result = source_first_payload(payload, units); row = result['target_facts'][0]
        self.assertLess(list(result).index('target_facts'), list(result).index('program_filled_original_units'))
        self.assertLess(list(row).index('cited_original_units'), list(row).index('original'))
        self.assertEqual(row['cited_original_units'][0]['text'], TEXT)
        self.assertEqual(row['original'], payload['target_facts'][0]['original'])
        self.assertEqual(result['program_filled_original_units'], payload['program_filled_original_units'])
        self.assertEqual(payload, before)

    def test_unknown_source_cannot_be_bound(self):
        payload, units = materials(); payload['target_facts'][0]['original']['sentence_ids'] = ['s9999']
        with self.assertRaises(KeyError): source_first_payload(payload, units)

    def test_preexisting_packet_cannot_override_program_binding(self):
        payload, units = materials(); payload['target_facts'][0]['cited_original_units'] = [{'id': 's9999', 'text': 'foreign'}]
        result = source_first_payload(payload, units)
        self.assertEqual(result['target_facts'][0]['cited_original_units'][0]['text'], TEXT)

    def test_candidate_changes_only_the_documented_answer_response_budget(self):
        config = {**CONFIG, 'PAPERTRAIL_GROUNDED_ANSWERS': 'true', 'PAPERTRAIL_GROUNDED_PROFILE': 'compact_repair',
            'PAPERTRAIL_INSUFFICIENT_FACTS_COMPAT': 'true', 'PAPERTRAIL_QA_REASONING': 'high'}
        before = build_settings(config); after = build_settings({**config, 'PAPERTRAIL_CITATION_SCOPE_REVIEW': 'true'})
        a, b = before.model_dump(), after.model_dump()
        for obj in (a, b): obj.pop('citation_scope_review'); obj.pop('md5', None)
        self.assertEqual(a['llm_config']['model_list'][0]['litellm_params']['max_tokens'], 24000)
        self.assertEqual(b['llm_config']['model_list'][0]['litellm_params']['max_tokens'], 32768)
        b['llm_config']['model_list'][0]['litellm_params']['max_tokens'] = 24000
        self.assertEqual(a, b)
        self.assertEqual(compact_policy_identity(repair=True, insufficient_facts_compat=True)['version'], 'source-led-answer-v32-compact')
        self.assertEqual(compact_policy_identity(repair=True, insufficient_facts_compat=True, citation_scope_review=True)['max_model_calls'], 2)
        for value in ['yes', 'true']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_settings({**config, 'PAPERTRAIL_CITATION_SCOPE_REVIEW': value, 'PAPERTRAIL_INSUFFICIENT_FACTS_COMPAT': 'false'})


class SourceBoundGeneratorTest(unittest.IsolatedAsyncioTestCase):
    async def test_same_two_calls_and_four_field_review_preserve_raw_outputs(self):
        raw = json.dumps({'parts': [{'id': 'p1', 'question_part': 'What is supported?', 'answer_kind': 'fact',
            'direct_answer': {'statement': STATEMENT, 'scope': 'experimental_result', 'sentence_ids': ['s0001']},
            'status': 'supported', 'facts': [], 'gap': ''}]})
        review = json.dumps({'fact_checks': [support_check('p1f1', ['s0001'])], 'corrections': [], 'additions': [], 'gap_reviews': []})
        model = FakeModel(raw, review)
        result = await generate_compact_grounded_answer(question='What is supported?', final_context=packed_context(a=TEXT),
            model=model, repair=True, insufficient_facts_compat=True, citation_scope_review=True)
        self.assertIsNotNone(result.answer)
        self.assertEqual(model.stages, ['grounded_outline', 'grounded_focus'])
        payload = json.loads(model.calls[1]['messages'][1].content)
        self.assertEqual(payload['target_facts'][0]['cited_original_units'][0]['text'], TEXT)
        self.assertEqual(set(json.loads(result.audit['stages']['grounded_focus']['raw'])), {'fact_checks', 'corrections', 'additions', 'gap_reviews'})
        self.assertEqual(result.audit['version'], 'source-led-answer-v34-compact')


if __name__ == '__main__': unittest.main()
