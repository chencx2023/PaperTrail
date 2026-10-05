import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.search.arxiv import (
    ArxivAPIError,
    InvalidArxivFeed,
    parse_atom_feed,
)


FIXTURES = Path(__file__).parent / "fixtures" / "arxiv"


class ArxivParserTest(unittest.TestCase):
    def parse_fixture(self, filename: str):
        return parse_atom_feed((FIXTURES / filename).read_bytes())

    def test_parses_metadata_and_uses_highest_version_without_reordering(self):
        feed = self.parse_fixture("normal_with_versions.xml")

        self.assertEqual(feed.feed_id, "http://arxiv.org/api/papertrail-normal")
        self.assertEqual(feed.total_results, 3)
        self.assertEqual(feed.start_index, 0)
        self.assertEqual(feed.items_per_page, 3)
        self.assertEqual([record.source_rank for record in feed.records], [1, 2, 3])
        self.assertEqual(len(feed.candidates), 2)

        current = feed.candidates[0]
        self.assertEqual(current.arxiv_id, "2401.00001")
        self.assertEqual(current.versioned_id, "2401.00001v3")
        self.assertEqual(current.version, 3)
        self.assertEqual(current.source_rank, 3)
        self.assertEqual(current.display_rank, 1)
        self.assertEqual(current.title, "Memory for Agents: Revised Version")
        self.assertEqual(
            current.abstract,
            "The revised version studies persistent memory for language-model agents.",
        )
        self.assertEqual(
            current.authors,
            ("Alice Example", "Bob Example", "Carol Example"),
        )
        self.assertEqual(current.primary_category, "cs.AI")
        self.assertEqual(current.categories, ("cs.AI", "cs.CL"))
        self.assertEqual(current.doi, "10.1234/fixture.current")
        self.assertEqual(
            current.license_url,
            "https://creativecommons.org/licenses/by/4.0/",
        )
        self.assertEqual(
            [(item.versioned_id, item.version, item.source_rank) for item in current.versions],
            [("2401.00001v1", 1, 1), ("2401.00001v3", 3, 3)],
        )

        legacy = feed.candidates[1]
        self.assertEqual(legacy.arxiv_id, "cs/9901001")
        self.assertEqual(legacy.versioned_id, "cs/9901001v2")
        self.assertEqual((legacy.source_rank, legacy.display_rank), (2, 2))
        self.assertIsNone(legacy.doi)
        self.assertIsNone(legacy.license_url)

    def test_valid_empty_feed_is_the_only_zero_result_success(self):
        feed = self.parse_fixture("empty.xml")

        self.assertEqual(feed.total_results, 0)
        self.assertEqual(feed.records, ())
        self.assertEqual(feed.candidates, ())

    def test_feed_atom_link_without_rel_is_accepted(self):
        feed = self.parse_fixture("empty_link_without_rel.xml")

        self.assertEqual(
            feed.self_url,
            "https://arxiv.org/api/query?search_query=ti:no-such-paper&start=0&max_results=20",
        )

    def test_atom_error_entry_raises_api_error(self):
        with self.assertRaises(ArxivAPIError) as raised:
            self.parse_fixture("error.xml")

        self.assertEqual(
            raised.exception.error_id,
            "incorrect_id_format_for_1234.12345",
        )
        self.assertIn("incorrect id format", str(raised.exception))

    def test_malformed_xml_is_not_an_empty_result(self):
        with self.assertRaises(InvalidArxivFeed):
            self.parse_fixture("malformed.xml")

    def test_entry_missing_required_id_is_not_an_empty_result(self):
        with self.assertRaises(InvalidArxivFeed):
            self.parse_fixture("missing_id.xml")


if __name__ == "__main__":
    unittest.main()
