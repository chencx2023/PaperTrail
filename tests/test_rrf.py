from dataclasses import replace
from pathlib import Path
import unittest

from backend.app.search.arxiv import parse_atom_feed
from backend.app.search.ranking import fuse_rankings


FIXTURE = Path(__file__).parent / "fixtures/arxiv/normal_with_versions.xml"


class RrfTest(unittest.TestCase):
    def test_duplicate_versions_vote_once_and_rank_is_after_deduplication(self):
        feed = parse_atom_feed(FIXTURE.read_bytes())
        result = fuse_rankings([feed, feed])
        self.assertEqual(len(result.candidates), len(feed.candidates))
        by_id = {item["arxiv_id"]: item for item in result.ranking}
        for rank, candidate in enumerate(feed.candidates, 1):
            row = by_id[candidate.arxiv_id]
            self.assertEqual(len(row["matches"]), 2)
            self.assertAlmostEqual(row["score"], 2 / (60 + rank))
            self.assertEqual(row["matches"][0]["rank"], rank)
        self.assertEqual(len(result.records), len({record.versioned_id for record in feed.records}))

    def test_multiple_queries_promote_shared_work_and_preserve_every_candidate(self):
        feed = parse_atom_feed(FIXTURE.read_bytes())
        a, b = feed.candidates[:2]
        first = replace(feed, candidates=(a, b))
        second = replace(feed, candidates=(b,))
        result = fuse_rankings([first, second])
        self.assertEqual([c.arxiv_id for c in result.candidates], [b.arxiv_id, a.arxiv_id])
        self.assertEqual(result.merge_order, (a.arxiv_id, b.arxiv_id))
        self.assertEqual([c.display_rank for c in result.candidates], [1, 2])

    def test_equal_scores_use_stable_work_id_even_when_lists_swap(self):
        feed = parse_atom_feed(FIXTURE.read_bytes())
        a, b = feed.candidates[:2]
        one = replace(feed, candidates=(a,))
        two = replace(feed, candidates=(b,))
        expected = sorted([a.arxiv_id, b.arxiv_id])
        self.assertEqual([c.arxiv_id for c in fuse_rankings([one, two]).candidates], expected)
        self.assertEqual([c.arxiv_id for c in fuse_rankings([two, one]).candidates], expected)


if __name__ == "__main__":
    unittest.main()
