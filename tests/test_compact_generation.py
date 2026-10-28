"""Offline behavior checks of the two-call candidate; not LLM quality scores."""
import asyncio
import copy
import json
import unittest

from backend.app.qa.compact import (generate_compact_grounded_answer, merge_review,
    model_units, normalize_plan, OUTLINE_SYSTEM, FOCUS_SYSTEM)
from backend.app.qa.grounded import source_units, validate_outline
from tests.test_grounded_generation import FakeModel, packed_context, outline_payload, support_check


def planned(data, kind="criterion"):
    data=copy.deepcopy(data)
    for part in data["parts"]:
        direct=part["facts"].pop(0) if part["facts"] else None
        part.update(answer_kind=kind,direct_answer=direct)
    return json.dumps(data,ensure_ascii=False)


def review(checks, corrections=(), additions=()):
    return json.dumps({"fact_checks": checks, "corrections": list(corrections),
                       "additions": list(additions)}, ensure_ascii=False)


class CompactGenerationTest(unittest.IsolatedAsyncioTestCase):
    async def test_decision_rule_paraphrase_is_not_published_when_a_qualifier_is_lost(self):
        original="A response is non-direct when it hedges using modal verbs."
        context=packed_context(a=original)
        data=outline_payload()
        data["parts"][0]["facts"][0].update(statement="Any modal verb makes the response non-direct.",
                                            sentence_ids=["s0001"])
        raw=planned(data)
        result=await generate_compact_grounded_answer(question="如何判定回答？",final_context=context,
            model=FakeModel(raw,review([support_check("p1f1",["s0001"])])))
        self.assertEqual(result.status,"generated_not_independently_graded")
        self.assertNotIn("Any modal verb",result.answer)
        self.assertIn(original,result.answer)
        self.assertIn("Any modal verb",result.audit["stages"]["grounded_outline"]["raw"])
        self.assertEqual(result.audit["rendered_claims"][0]["statement"],original)
        self.assertEqual(result.audit["rendered_claims"][0]["rendering_mode"],"verbatim_original_rule")
        self.assertEqual(len(result.llm_results),2)

    async def test_two_calls_only_preserve_original_units_once_and_raw_responses(self):
        raw = planned(outline_payload())
        focused = review([support_check("p1f1", ["s0001", "s0002", "s0003"])])
        model = FakeModel(raw, focused)
        result = await generate_compact_grounded_answer(question="How is performance judged?",
            final_context=packed_context(), model=model)
        self.assertEqual(result.status, "generated_not_independently_graded")
        self.assertEqual(model.stages, ["grounded_outline", "grounded_focus"])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(result.audit["stages"]["grounded_outline"]["raw"], raw)
        self.assertEqual(result.audit["stages"]["grounded_focus"]["raw"], focused)
        payload = json.loads(model.calls[1]["messages"][1].content)
        self.assertNotIn("parent_original_passages_for_scope", payload)
        self.assertNotIn("start", payload["program_filled_original_units"][0])
        self.assertIn("start", result.audit["units"][0])
        self.assertEqual(payload["program_filled_original_units"], model_units(result.audit["units"]))
        self.assertIn("(pqac-alpha)", result.answer)
        self.assertIn("(pqac-beta)", result.answer)
        self.assertTrue(all(c["response_format"] == {"type": "json_object"} for c in model.calls))

    async def test_literal_condition_error_is_corrected_without_rewriting_other_facts(self):
        context = packed_context(a="A response is non-direct when it hedges using modal verbs. Direct answers entail the gold answer.")
        data = outline_payload()
        data["parts"][0]["facts"] = [
            {"statement": "Any modal verb makes the response non-direct.", "scope": "response_evaluation", "sentence_ids": ["s0001"]},
            {"statement": "Direct answers entail the gold answer.", "scope": "response_evaluation", "sentence_ids": ["s0002"]}]
        replacement = {**data["parts"][0]["facts"][0], "statement": "A response is non-direct when it hedges using modal verbs."}
        correction = {"fact_id": "p1f1", "action": "replace", "reason_type": "evidence",
            "reason": "The governing action was omitted.", "replacement": replacement, "gap": ""}
        focused = review([support_check("p1f1", ["s0001"], verdict="replace"),
                         support_check("p1f2", ["s0002"])], [correction])
        result = await generate_compact_grounded_answer(question="What is the decision rule?",
            final_context=context, model=FakeModel(planned(data), focused))
        self.assertIn("when it hedges", result.answer)
        final = result.audit["stages"]["grounded_focus"]["parsed"]["parts"][0]["facts"]
        self.assertEqual(final[1], data["parts"][0]["facts"][1])

    async def test_missing_requested_fact_can_be_added_with_sources_in_same_review(self):
        data = outline_payload()
        data["parts"][0]["facts"][0].update(statement="Verified nodes earn points.", sentence_ids=["s0001"])
        fact = {"statement": "The completion fraction uses the proportion of passed nodes.",
                "scope": "metric_definition", "sentence_ids": ["s0003"]}
        addition = {"part_id": "p1", "reason": "The requested partial-completion definition is absent.", "fact": fact}
        model = FakeModel(planned(data), review([support_check("p1f1", ["s0001"])], additions=[addition]))
        result = await generate_compact_grounded_answer(question="How do scoring and partial completion work?",
            final_context=packed_context(), model=model)
        self.assertIn(fact["statement"] + " (pqac-beta)", result.answer)
        stage = result.audit["stages"]["grounded_focus"]
        self.assertEqual(stage["addition_application"][0]["fact_id"], "p1a1")
        self.assertEqual(stage["applied_corrections"]["final_fact_ids_by_part"]["p1"], ["p1f1", "p1a1"])
        self.assertEqual(len(model.calls), 2)

    async def test_invalid_review_preserves_both_returned_usages_without_retry(self):
        model = FakeModel(planned(outline_payload()), '{"fact_checks": [', "unused")
        result = await generate_compact_grounded_answer(question="What is checked?",
            final_context=packed_context(), model=model)
        self.assertEqual(result.status, "grounded_focus_failed")
        self.assertIsNone(result.answer)
        self.assertEqual(len(result.llm_results), 2)
        self.assertEqual(len(result.audit["model_usage"]), 2)
        self.assertEqual(model.outputs, ["unused"])
        self.assertNotIn("input", str(result.audit.get("validation_errors")))

    async def test_provider_failure_is_safe_and_does_not_add_another_call(self):
        model = FakeModel(planned(outline_payload()), RuntimeError("private-sentinel"))
        result = await generate_compact_grounded_answer(question="What is checked?",
            final_context=packed_context(), model=model)
        self.assertIsNone(result.answer)
        self.assertEqual(len(result.llm_results), 1)
        self.assertNotIn("private-sentinel", repr(result))

    async def test_cancellation_preserves_first_stage_and_propagates(self):
        started = asyncio.Event()
        class WaitingModel(FakeModel):
            async def call_single(self, **kwargs):
                if len(self.calls) == 1:
                    started.set()
                    await asyncio.Event().wait()
                return await super().call_single(**kwargs)
        model = WaitingModel(planned(outline_payload()))
        snapshots = []
        task = asyncio.create_task(generate_compact_grounded_answer(question="What is checked?",
            final_context=packed_context(), model=model,
            on_progress=lambda a,r: snapshots.append((copy.deepcopy(a),len(r)))))
        await asyncio.wait_for(started.wait(),3);task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertEqual(snapshots[-1][0]["status"], "grounded_focus_cancelled")
        self.assertEqual(snapshots[-1][1], 1)


class CompactAdditionValidationTest(unittest.TestCase):
    def test_supported_part_requires_explicit_direct_answer_before_review(self):
        data=json.loads(planned(outline_payload()))
        data["parts"][0]["direct_answer"]=None
        with self.assertRaisesRegex(ValueError,"compact_part_requires_direct_answer_or_gap"):
            normalize_plan(json.dumps(data),source_units(packed_context()))

    def test_support_guidance_does_not_reject_necessary_conditions(self):
        data=json.loads(planned(outline_payload()))
        fact=copy.deepcopy(data["parts"][0]["direct_answer"])
        data["parts"][0]["facts"]=[fact,fact,fact]
        normalized,_=normalize_plan(json.dumps(data),source_units(packed_context()))
        self.assertEqual(len(normalized.parts[0].facts),4)
        data["parts"][0]["answer_kind"]="mechanism"
        normalized,roles=normalize_plan(json.dumps(data),source_units(packed_context()))
        self.assertEqual(len(normalized.parts[0].facts),4)
        self.assertEqual(roles[0]["direct_answer_fact_id"],"p1f1")

    def test_known_scope_aliases_are_logged_and_unknown_scopes_still_fail(self):
        data=json.loads(planned(outline_payload()))
        data["parts"][0]["direct_answer"]["scope"]="model-response evaluation"
        raw=json.dumps(data); changes=[]
        normalized,_=normalize_plan(raw,source_units(packed_context()),normalizations=changes)
        self.assertEqual(normalized.parts[0].facts[0].scope,"response_evaluation")
        self.assertIn("model-response evaluation",raw)
        self.assertEqual(changes,[{"path":["parts",0,"direct_answer","scope"],
            "from":"model-response evaluation","to":"response_evaluation"}])
        data["parts"][0]["direct_answer"]["scope"]="made_up_scope"
        with self.assertRaises(ValueError): normalize_plan(json.dumps(data),source_units(packed_context()))

    def test_scope_alias_cannot_bypass_reference_construction_guard(self):
        data=json.loads(planned(outline_payload()))
        data["parts"][0]["direct_answer"].update(scope="model-response evaluation",sentence_ids=["s0001"])
        units=source_units(packed_context()); units[0]["source_scope"]="reference_construction"
        with self.assertRaisesRegex(ValueError,"reference_construction"):
            normalize_plan(json.dumps(data),units)

    def test_review_aliases_are_logged_without_extra_model_calls(self):
        units=source_units(packed_context())
        initial=validate_outline(json.dumps(outline_payload()),units)
        replacement={**initial.parts[0].facts[0].model_dump(),"scope":"metric definition"}
        correction={"fact_id":"p1f1","action":"replace","reason_type":"evidence",
            "reason":"Preserve operative conditions.","replacement":replacement,"gap":""}
        raw=review([support_check("p1f1",["s0001"],verdict="replace")],[correction])
        merged,provenance=merge_review(raw,initial,units)
        self.assertEqual(merged.parts[0].facts[0].scope,"metric_definition")
        self.assertEqual(provenance["scope_normalizations"],[{"path":["corrections",0,"replacement","scope"],
            "from":"metric definition","to":"metric_definition"}])
        for prompt in (OUTLINE_SYSTEM,FOCUS_SYSTEM):
            self.assertIn("data_or_reference_construction, response_evaluation, method, metric_definition,",prompt)

    def test_total_fact_limit_and_known_answer_kinds_still_apply(self):
        data=json.loads(planned(outline_payload()))
        data["parts"][0]["answer_kind"]="unrecognized"
        with self.assertRaisesRegex(ValueError,"compact_unknown_answer_kind"):
            normalize_plan(json.dumps(data),source_units(packed_context()))
        part=data["parts"][0];part["answer_kind"]="criterion"
        part["facts"]=[copy.deepcopy(part["direct_answer"]) for _ in range(8)]
        data["parts"]=[{**copy.deepcopy(part),"id":f"p{i}"} for i in (1,2,3)]
        with self.assertRaisesRegex(ValueError,"compact_outline_exceeds_24_facts"):
            normalize_plan(json.dumps(data),source_units(packed_context()))

    def test_unknown_duplicate_or_out_of_scope_additions_fail_closed(self):
        units = source_units(packed_context()); initial = validate_outline(json.dumps(outline_payload()),units)
        fact = {"statement": "A separate required condition is visible.","scope":"method","sentence_ids":["s0003"]}
        add = {"part_id":"p1","reason":"The question needs this condition.","fact":fact}
        bads = [[{**add,"part_id":"p9"}], [add,add], [{**add,"reason":" "}],
                [{**add,"fact":{**fact,"sentence_ids":["s9999"]}}],
                [{**add,"fact":initial.parts[0].facts[0].model_dump()}]]
        for additions in bads:
            with self.subTest(additions=additions), self.assertRaises(ValueError):
                merge_review(review([support_check("p1f1",["s0001","s0002","s0003"])],additions=additions),initial,units)
        initial.parts[0].facts[0].sentence_ids=["s0001","s0002"]
        units[2]["source_scope"]="reference_construction"
        with self.assertRaisesRegex(ValueError,"reference_construction"):
            merge_review(review([support_check("p1f1",["s0001","s0002"])],additions=[add]),initial,units)


if __name__ == "__main__": unittest.main()
