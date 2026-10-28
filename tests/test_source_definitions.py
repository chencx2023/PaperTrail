"""Offline tests of source-first definitions; these do not grade a live model."""
import copy
import json
import unittest
from backend.app.qa.grounded import (generate_grounded_answer, source_units, validate_planned_outline,
    apply_definitions, apply_focus_corrections)
from tests.test_grounded_generation import (FakeModel, packed_context, outline_payload, planned_outline,
    review_response, support_check)


def definitions(metric=True, reference=False):
    return {"parts": [{"part_id": "p1",
        "metrics": [{"name": "Completion fraction", "definition": "已通过的节点占全部指定节点的比例。",
                     "sentence_ids": ["s0003"]}] if metric else [],
        "reference_roles": [{"name": "Verified reference trace", "purpose": "供比较运行轨迹和分析过程，不是按轨迹相似度评分。",
                             "sentence_ids": ["s0003"]}] if reference else [], "gap": ""}]}


class SourceDefinitionsTest(unittest.TestCase):
    def test_append_keeps_original_and_primary_identity_and_binds_source(self):
        units = source_units(packed_context())
        original, plan = validate_planned_outline(planned_outline(json.dumps(outline_payload())), units)
        before = original.model_dump()
        merged, log, _ = apply_definitions(json.dumps(definitions()), original, units)
        self.assertEqual(original.model_dump(), before)
        self.assertEqual(merged.parts[0].facts[0].model_dump(), before["parts"][0]["facts"][0])
        self.assertEqual(log[0]["fact_id"], "p1f2")
        self.assertEqual(log[0]["answer_role"], "metric_definition")
        self.assertIn("比例", merged.parts[0].facts[1].statement)
        self.assertEqual(plan[0]["direct_answer_fact_id"], "p1f1")

    def test_parts_sources_embedded_ids_and_duplicates_are_rejected(self):
        units = source_units(packed_context())
        original, _ = validate_planned_outline(planned_outline(json.dumps(outline_payload())), units)
        bads = [definitions(metric=False, reference=False)]
        bad = definitions(); bad["parts"][0]["part_id"] = "p2"; bads.append(bad)
        bad = definitions(); bad["parts"][0]["metrics"][0]["sentence_ids"] = ["s9999"]; bads.append(bad)
        bad = definitions(); bad["parts"][0]["metrics"][0]["definition"] = "use s0003 as a citation"; bads.append(bad)
        bad = definitions(); bad["parts"][0]["metrics"] *= 2; bads.append(bad)
        for bad in bads:
            with self.subTest(payload=bad), self.assertRaises(ValueError):
                apply_definitions(json.dumps(bad), original, units)

    def test_gap_is_preserved_and_reference_scope_is_not_scoring(self):
        units = source_units(packed_context())
        original, _ = validate_planned_outline(planned_outline(json.dumps(outline_payload())), units)
        raw = definitions(metric=False, reference=True)
        raw["parts"][0]["gap"] = "没有针对指定日期的核验记录。"
        merged, log, _ = apply_definitions(json.dumps(raw), original, units)
        self.assertEqual(merged.parts[0].status, "partial")
        self.assertEqual(merged.parts[0].facts[1].scope, "data_or_reference_construction")
        self.assertEqual(log[0]["answer_role"], "reference_role")


    def test_empty_result_cannot_borrow_an_unrelated_old_gap(self):
        units = source_units(packed_context())
        data = outline_payload()
        data["parts"][0].update(status="partial", gap="Missing a controlled comparison.")
        original, _ = validate_planned_outline(planned_outline(json.dumps(data)), units)
        with self.assertRaisesRegex(ValueError, "empty_definition_result_for_evaluation_part"):
            apply_definitions(json.dumps(definitions(metric=False)), original, units)

    def test_combined_gaps_fail_closed_without_truncating_original(self):
        units = source_units(packed_context())
        data = outline_payload()
        data["parts"][0].update(status="partial", gap="a" * 600)
        original, _ = validate_planned_outline(planned_outline(json.dumps(data)), units)
        raw = definitions(metric=False); raw["parts"][0]["gap"] = "b" * 600
        with self.assertRaises(ValueError):
            apply_definitions(json.dumps(raw), original, units)
        self.assertEqual(original.parts[0].gap, "a" * 600)

    def test_non_evaluation_part_cannot_gain_unrequested_definitions_or_gap(self):
        units = source_units(packed_context())
        original, _ = validate_planned_outline(planned_outline(json.dumps(outline_payload())), units)
        for raw in (definitions(), definitions(metric=False, reference=True)):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError,
                    "definitions_must_only_target_evaluation_parts"):
                apply_definitions(json.dumps(raw), original, units, eligible_part_ids=set())
        raw = definitions(metric=False)
        raw["parts"][0]["gap"] = "Unasked definition gap."
        with self.assertRaisesRegex(ValueError, "definitions_must_only_target_evaluation_parts"):
            apply_definitions(json.dumps(raw), original, units, eligible_part_ids=set())
        raw["parts"][0]["gap"] = ""
        merged, additions, _ = apply_definitions(json.dumps(raw), original, units,
                                               eligible_part_ids=set())
        self.assertEqual(merged.model_dump(), original.model_dump())
        self.assertEqual(additions, [])

    def test_reference_construction_cannot_define_system_scoring_but_can_describe_reference(self):
        units = source_units(packed_context())
        original, _ = validate_planned_outline(planned_outline(json.dumps(outline_payload())), units)
        units[2]["source_scope"] = "reference_construction"
        with self.assertRaisesRegex(ValueError,
                "reference_construction_cannot_support_response_or_method_rule"):
            apply_definitions(json.dumps(definitions()), original, units)
        merged, _, _ = apply_definitions(json.dumps(definitions(metric=False, reference=True)),
                                         original, units)
        self.assertEqual(merged.parts[0].facts[-1].scope, "data_or_reference_construction")


class DefinitionGenerationTest(unittest.IsolatedAsyncioTestCase):
    async def test_missing_proportion_becomes_published_body_before_review(self):
        initial = planned_outline(json.dumps(outline_payload()))
        raw = json.dumps(definitions(), ensure_ascii=False)
        review = review_response([support_check("p1f1", ["s0001", "s0002", "s0003"]),
                                  support_check("p1f2", ["s0003"])])
        model = FakeModel(initial, raw, review)
        result = await generate_grounded_answer(question="如何判断任务表现？",
            final_context=packed_context(), model=model)
        self.assertEqual(result.status, "generated_not_independently_graded")
        self.assertIn("已通过的节点占全部指定节点的比例", result.answer)
        self.assertEqual(result.audit["stages"]["grounded_outline"]["raw"], initial)
        self.assertEqual(result.audit["stages"]["grounded_definitions"]["raw"], raw)
        self.assertEqual(result.audit["definition_addition_map"][0]["fact_id"], "p1f2")
        self.assertEqual(len(result.llm_results), 3)
        for call in model.calls:
            self.assertEqual(call["response_format"], {"type": "json_object"})
        extraction_payload = json.loads(model.calls[1]["messages"][1].content)
        self.assertEqual(extraction_payload["eligible_evaluation_part_ids"], ["p1"])
        payload = json.loads(model.calls[2]["messages"][1].content)
        self.assertEqual(payload["target_facts"][1]["answer_role"], "metric_definition")
        self.assertTrue(payload["target_facts"][1]["protected"])

    async def test_protected_definition_cannot_be_dropped_as_unasked(self):
        raw = planned_outline(json.dumps(outline_payload()))
        drop = {"fact_id": "p1f2", "action": "drop", "reason_type": "irrelevant_or_duplicate",
                "reason": "Offline invalid attempt to omit the definition.", "replacement": None, "gap": ""}
        review = review_response([support_check("p1f1", ["s0001", "s0002", "s0003"]),
            support_check("p1f2", ["s0003"], verdict="drop")], [drop])
        result = await generate_grounded_answer(question="如何判断任务表现？", final_context=packed_context(),
            model=FakeModel(raw, json.dumps(definitions()), review))
        self.assertIsNone(result.answer)
        self.assertEqual(result.audit["error_code"], "required_answer_fact_cannot_be_dropped")

    async def test_invalid_definition_raw_stops_without_repair_or_retry_and_keeps_usage(self):
        model = FakeModel(planned_outline(json.dumps(outline_payload())), '{"parts": []]]', "unused")
        result = await generate_grounded_answer(question="如何判断任务表现？", final_context=packed_context(), model=model)
        self.assertEqual(result.status, "grounded_definitions_failed")
        self.assertIsNone(result.answer)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(len(result.llm_results), 2)
        self.assertEqual(result.audit["error_code"], "json_invalid")
        self.assertTrue(result.audit["validation_errors"][0]["json_position"])
        self.assertEqual(result.audit["stages"]["grounded_definitions"]["raw"], '{"parts": []]]')
        self.assertNotIn("input", str(result.audit.get("validation_errors")))


if __name__ == "__main__":
    unittest.main()
