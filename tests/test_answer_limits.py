"""Exercise token packing through the application's real PaperQA serializer."""

import os
import unittest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from scripts.model_config import build_settings
from paperqa.types import Context, Doc, Text


def configured():
    return build_settings({
            "DEEPSEEK_MODEL": "offline-model", "DEEPSEEK_BASE_URL": "https://example.invalid",
            "DEEPSEEK_API_KEY": "offline", "EMBEDDING_MODEL": "offline-embedding",
            "EMBEDDING_BASE_URL": "https://example.invalid", "EMBEDDING_API_KEY": "offline",
            "EMBEDDING_DIMENSIONS": 2, "EMBEDDING_BATCH_SIZE": 10,
    })


def passage(index, *, document=None, score=9, text=None):
    return Context(context=f"Saved summary {index}.", score=score,
        text=Text(name=f"Paper {index}", text=text or f"Original passage {index}.",
            doc=Doc(docname=f"Paper{document or index}", citation=f"Fixed source {document or index}",
                dockey=f"doc-{document or index}")))


class AnswerLimitsTest(unittest.IsolatedAsyncioTestCase):
    async def test_more_than_six_relevant_fragments_are_available_without_a_piece_cap(self):
        contexts = [passage(index) for index in range(1, 14)] + [passage(99, score=0)]
        serialized = await configured().context_serializer(contexts, "Summarize the evidence", None)
        keys = serialized.split("\n\nValid Keys: ")[-1].split(", ")
        self.assertEqual(set(keys), {context.id for context in contexts[:-1]})
        for context in contexts[:-1]:
            self.assertIn(context.text.text, serialized)
            self.assertNotIn(context.context, serialized)
        self.assertNotIn(contexts[-1].id, keys)

    async def test_original_duplicates_only_collapse_within_the_same_fixed_document(self):
        contexts = [passage(1, document='a', text='Same source text'),
            passage(2, document='a', text='Same  source text', score=8),
            passage(3, document='b', text='Same source text', score=8),
            passage(4, document='a', text='Additional useful result', score=7)]
        serialized = await configured().context_serializer(contexts, "Question", None)
        keys = serialized.split("\n\nValid Keys: ")[-1].split(", ")
        self.assertEqual(set(keys), {contexts[i].id for i in [0, 2, 3]})

    async def test_budget_preserves_relevant_source_coverage_without_a_per_paper_cap(self):
        from backend.app.qa.context import pack_contexts, context_tokens
        contexts = [passage(i, document='a', score=10, text=(f'Specific finding {i}. ' * 25)) for i in range(1, 9)]
        contexts.append(passage(20, document='b', score=9, text='Independent result. ' * 25))
        contexts.append(passage(21, document='outside', score=0))
        full, _ = pack_contexts(contexts, cutoff=1, budget=24000)
        budget = context_tokens(full) // 2
        packed, report = pack_contexts(contexts, cutoff=1, budget=budget)
        self.assertLessEqual(context_tokens(packed), budget)
        self.assertIn(contexts[-2].id, report['selected_context_ids'])
        self.assertGreater(report['selected_fragments'], 2)
        self.assertEqual(report['budget_tokens'], budget)
        self.assertGreater(report['excluded_budget'], 0)
        self.assertEqual(report['excluded_irrelevant'], 1)
        self.assertNotIn(contexts[-1].id, report['selected_context_ids'])
        self.assertEqual(contexts[0].text.text, 'Specific finding 1. ' * 25)

    def test_answer_output_budget_does_not_change_agent_or_summary_roles(self):
        settings = configured()
        maximum = lambda cfg: cfg['model_list'][0]['litellm_params']['max_tokens']
        self.assertEqual(maximum(settings.llm_config), 6000)
        self.assertEqual(maximum(settings.summary_llm_config), 1800)
        self.assertEqual(maximum(settings.agent.agent_llm_config), 1800)
        self.assertEqual(settings.answer.evidence_k, 10)
        self.assertEqual(settings.answer.answer_max_sources, 0)
        self.assertNotIn('300', settings.answer.answer_length)
        self.assertNotIn('compact', settings.prompts.qa)

    def test_selection_record_is_derived_from_the_saved_context_not_a_hypothetical_request(self):
        from types import SimpleNamespace
        from backend.app.qa.context import pack_contexts, saved_selection_report
        contexts = [passage(1), passage(2, score=0)]
        serialized, packed = pack_contexts(contexts, cutoff=1)
        evidence = [SimpleNamespace(upstream_context_id=c.id, upstream_dockey=c.text.doc.dockey,
            original_text=c.text.text, summary=c.context, score=c.score, upstream_locator=c.text.name) for c in contexts]
        report = saved_selection_report(SimpleNamespace(final_context=serialized, evidence=evidence), 1)
        self.assertEqual(report['status'], 'known')
        self.assertEqual(report['selected_context_ids'], packed['selected_context_ids'])
        self.assertEqual(report['estimated_tokens'], packed['estimated_tokens'])
        self.assertEqual(report['excluded'][0]['reason'], 'below_relevance_cutoff')
