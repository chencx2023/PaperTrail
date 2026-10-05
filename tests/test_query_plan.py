import unittest

from backend.app.search.query_builder import (
    InvalidQueryPlan, build_query_plan, plan_from_data, query_plans,
)


EXPANSIONS = [
    {"label": "同义表达", "groups": [["coding agent", "software engineering agent"]], "exclude": []},
    {"label": "能力评估", "groups": [["coding agent"], ["benchmark", "evaluation"]], "exclude": ["medical coding"]},
]


class QueryPlanTest(unittest.TestCase):
    def test_keeps_baseline_and_compiles_synonyms_facets_and_exclusions(self):
        plan = build_query_plan("coding agent", EXPANSIONS)
        queries = query_plans(plan)
        self.assertEqual(len(queries), 3)
        self.assertEqual(queries[0].search_query, "(ti:coding OR abs:coding) AND (ti:agent OR abs:agent)")
        self.assertEqual(queries[1].search_query, '((ti:"coding agent" OR abs:"coding agent") OR (ti:"software engineering agent" OR abs:"software engineering agent"))')
        self.assertIn('(ti:benchmark OR abs:benchmark) OR (ti:evaluation OR abs:evaluation)', queries[2].search_query)
        self.assertIn('ANDNOT (ti:"medical coding" OR abs:"medical coding")', queries[2].search_query)
        self.assertEqual(plan_from_data(plan.as_data()), plan)

    def test_rejects_unbounded_or_injected_plans(self):
        for expansions in [[], EXPANSIONS * 2,
            [{"label": "bad", "groups": [["https://evil.test"]], "exclude": []}],
            [{"label": "bad", "groups": [["ti:agent"]], "exclude": []}],
            [{"label": "bad", "groups": [["agent OR memory"]], "exclude": []}],
            [{"label": "bad", "groups": [["agent"]] * 4, "exclude": []}],
            [{"label": "bad", "groups": [["agent"]], "exclude": [], "url": "evil"}],
        ]:
            with self.subTest(expansions=expansions), self.assertRaises(InvalidQueryPlan):
                build_query_plan("coding agent", expansions)

    def test_duplicate_query_and_version_mismatch_do_not_add_votes(self):
        with self.assertRaises(InvalidQueryPlan):
            build_query_plan("coding agent", [EXPANSIONS[0], EXPANSIONS[0]])
        with self.assertRaises(InvalidQueryPlan):
            plan_from_data({"version": 2, "baseline_query_text": "coding agent", "expansions": EXPANSIONS})

    def test_repeated_baseline_word_does_not_make_identical_expansion_distinct(self):
        with self.assertRaises(InvalidQueryPlan):
            build_query_plan("coding agent agent", [{"label": "重复", "groups": [["coding"], ["agent"]], "exclude": []}])


if __name__ == "__main__":
    unittest.main()
