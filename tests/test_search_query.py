import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.search.service import InvalidSearchQuery, build_arxiv_query


class SearchQueryTest(unittest.TestCase):
    def test_builds_and_of_title_or_abstract_clauses(self):
        plan = build_arxiv_query("LLM agent memory")

        self.assertEqual(plan.query_text, "LLM agent memory")
        self.assertEqual(plan.normalized_query_text, "LLM agent memory")
        self.assertEqual(plan.query_terms, ("LLM", "agent", "memory"))
        self.assertEqual(
            plan.search_query,
            "(ti:LLM OR abs:LLM) AND (ti:agent OR abs:agent) "
            "AND (ti:memory OR abs:memory)",
        )

    def test_normalizes_whitespace_without_removing_terms(self):
        plan = build_arxiv_query("  long-term   memory\tfor LLM agents  ")

        self.assertEqual(
            plan.query_terms,
            ("long-term", "memory", "for", "LLM", "agents"),
        )
        self.assertEqual(
            plan.normalized_query_text,
            "long-term memory for LLM agents",
        )
        self.assertEqual(
            plan.search_query,
            "(ti:long-term OR abs:long-term) AND "
            "(ti:memory OR abs:memory) AND (ti:for OR abs:for) AND "
            "(ti:LLM OR abs:LLM) AND (ti:agents OR abs:agents)",
        )

    def test_rejects_empty_query(self):
        with self.assertRaises(InvalidSearchQuery):
            build_arxiv_query(" \t\n ")

    def test_rejects_query_language_syntax(self):
        for query_text in (
            "LLM AND memory",
            "LLM or memory",
            '"long term" memory',
            "(LLM) memory",
            "ti:LLM memory",
            "大模型 agent",
        ):
            with self.subTest(query_text=query_text):
                with self.assertRaises(InvalidSearchQuery):
                    build_arxiv_query(query_text)

    def test_rejects_more_than_twelve_terms(self):
        query_text = " ".join(f"term{index}" for index in range(13))

        with self.assertRaises(InvalidSearchQuery):
            build_arxiv_query(query_text)


if __name__ == "__main__":
    unittest.main()
