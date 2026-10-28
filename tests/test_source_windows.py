"""Offline source continuity, budget and reference integrity checks."""

import copy
import hashlib
import unittest

from paperqa.types import Context, Doc, Text
from backend.app.qa.context import (CONTEXT_POLICY, WINDOW_CONTEXT_POLICY, evidence_excerpt,
                                    pack_contexts, context_policy_identity, window_context_serializer)
from backend.app.qa.answer_review import original_passages
from backend.app.qa.grounded import source_units
from backend.app.qa.windows import expand_original_windows


def fixture():
    doc = Doc(docname="Fixed", citation="Fixed source", dockey="fixed-v1")
    tail = "x" * 180 + "The time limit is se"
    original = "a" * (3000 - len(tail)) + tail
    successor = original[-200:] + "t to a separate value for an explicitly named system. " + "z" * 1000
    a = Text(name="native-first", text=original, doc=doc, embedding=[1., 0.])
    b = Text(name="native-next", text=successor, doc=doc, embedding=[0., 1.])
    context = Context(id="pqac-original", context="Original relevance assessment", score=8, text=a)
    return context, [a, b]


def expand(contexts, texts, **kwargs):
    return expand_original_windows(contexts, texts, overlap=200, chunk_chars=3000, cutoff=1, **kwargs)


class SourceWindowTest(unittest.TestCase):
    def test_exact_continuation_offsets_and_native_objects_are_unchanged(self):
        context, texts = fixture()
        before = copy.deepcopy([c.model_dump() for c in [context, *texts]])
        result, records = expand([context], texts)
        window, record = result[0], records[0]
        self.assertEqual(window.text.text, texts[0].text + texts[1].text[200:900])
        self.assertEqual(len(window.text.text), 3700)
        self.assertIsNone(window.text.embedding)
        self.assertNotEqual(window.id, context.id)
        self.assertEqual(record["successor_span"], [200, 900])
        self.assertEqual(record["window_sha256"], hashlib.sha256(window.text.text.encode()).hexdigest())
        self.assertEqual([c.model_dump() for c in [context, *texts]], before)
        self.assertEqual(expand([context], texts)[0][0].id, window.id)

    def test_missing_ambiguous_wrong_order_or_cross_document_never_guesses(self):
        context, texts = fixture()
        other = texts[1].model_copy(update={"name": "another matching successor"})
        wrong_doc = texts[1].model_copy(update={"doc": Doc(docname="Other", citation="Other", dockey="other-v1")})
        filler = texts[1].model_copy(update={"name": "in-between", "text": "unrelated"})
        for native in (texts[:1], texts + [other], [*texts[:1], filler, texts[1]], [texts[0], wrong_doc]):
            with self.subTest(names=[text.name for text in native]):
                output, records = expand([context], native)
                self.assertEqual(records, [])
                self.assertIs(output[0], context)

    def test_duplicate_anchor_contexts_keep_unique_ids_then_deduplicate_originals(self):
        context, texts = fixture()
        duplicate = context.model_copy(update={"id": "pqac-second"})
        output, records = expand([context, duplicate], texts)
        self.assertEqual(len({c.id for c in output}), 2)
        self.assertEqual(len(records), 2)
        _, selection = pack_contexts(output, cutoff=1, policy=WINDOW_CONTEXT_POLICY)
        self.assertEqual(selection["selected_fragments"], 1)
        self.assertEqual(selection["excluded_duplicates"], 1)

    def test_short_successor_bounds_low_score_and_no_second_hop(self):
        context, texts = fixture()
        short = texts[1].model_copy(update={"text": texts[1].text[:240]})
        output, records = expand([context], [texts[0], short])
        self.assertEqual(len(output[0].text.text), 3040)
        self.assertEqual(records[0]["successor_span"], [200, 240])
        self.assertEqual(expand(output, texts)[1], [])
        self.assertEqual(expand([context.model_copy(update={"score": 0})], texts)[1], [])

    def test_window_is_atomic_in_budget_and_source_offsets_replay_across_join(self):
        context, texts = fixture()
        output, _ = expand([context], texts)
        rendered, selection = pack_contexts(output, cutoff=1, policy=WINDOW_CONTEXT_POLICY)
        key = output[0].id
        self.assertEqual(original_passages(rendered)[key], output[0].text.text)
        for unit in source_units(rendered):
            self.assertEqual(unit["text"], output[0].text.text[unit["start"]:unit["end"]])
        denied, report = pack_contexts(output, cutoff=1, budget=selection["estimated_tokens"] - 1,
                                       policy=WINDOW_CONTEXT_POLICY)
        self.assertEqual(report["selected_fragments"], 0)
        self.assertNotIn(key + ": Original", denied)
        accepted, report = pack_contexts(output, cutoff=1, budget=selection["estimated_tokens"],
                                         policy=WINDOW_CONTEXT_POLICY)
        self.assertEqual(accepted, rendered)

    def test_new_policy_retains_tail_while_legacy_excerpt_is_unchanged(self):
        context, texts = fixture()
        output, _ = expand([context], texts)
        window = output[0]
        legacy = evidence_excerpt(window.id, window.text.text, window.context)
        current = evidence_excerpt(window.id, window.text.text, window.context, policy=WINDOW_CONTEXT_POLICY)
        self.assertIn("first 3000 of 3700", legacy)
        self.assertNotIn("separate value", legacy)
        self.assertIn("separate value", current)
        self.assertEqual(CONTEXT_POLICY["original_character_limit"], 3000)
        self.assertEqual(context_policy_identity(window_context_serializer), WINDOW_CONTEXT_POLICY)


if __name__ == "__main__":
    unittest.main()
