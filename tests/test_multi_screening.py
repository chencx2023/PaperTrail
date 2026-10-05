import json
from pathlib import Path
import tempfile
import unittest

import httpx

from backend.app.search import screening
from tests.test_abstract_screening import CONFIG, saved_search
from tests.test_screening_spans import response_for


def large_pool():
    search = saved_search()
    search["candidates"] = [
        {**search["candidates"][0], "versioned_id": f"2501.{i:05d}v1", "display_rank": i}
        for i in range(1, 36)]
    search["fusion"] = {"method": "rrf", "candidate_count": 35}
    return search


class MultiScreeningTest(unittest.IsolatedAsyncioTestCase):
    async def test_all_rrf_candidates_are_sent_and_scope_is_saved_replayed_and_verified(self):
        search = large_pool()
        submitted = []
        def response(request):
            papers = json.loads(request.content)["messages"][1]["content"]
            inputs = json.loads(papers)
            submitted.append(inputs)
            output = response_for(inputs)
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(output)}}]})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
                result = await screening.generate_screening(search, root, config=CONFIG, client=client)
            self.assertEqual([len(batch["papers"]) for batch in submitted], [10, 10, 10, 5])
            self.assertEqual([paper["paper_id"] for batch in submitted for paper in batch["papers"]],
                [p["versioned_id"] for p in search["candidates"]])
            self.assertEqual(result["candidate_count"], 35)
            self.assertEqual(result["screening_scope"]["source_candidate_count"], 35)
            self.assertEqual(result["screening_scope"]["paper_ids"], [p["versioned_id"] for p in search["candidates"]])
            self.assertEqual(len(search["candidates"]), 35)
            self.assertTrue(screening.read_screening(search, root)["cache_hit"])
            path = root / "abstract-screenings" / f"{result['input_hash']}.json"
            saved = json.loads(path.read_text(encoding="utf-8"))
            saved["screening_scope"]["source_candidate_count"] = 30
            path.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaises(screening.ScreeningError):
                screening.read_screening(search, root)

    async def test_single_query_cannot_silently_truncate_and_oversized_fusion_is_rejected(self):
        search = large_pool()
        search.pop("fusion")
        self.assertEqual(len(screening._input(search)[0]['papers']), 35)
        search = large_pool()
        search["candidates"] *= 2
        with self.assertRaises(screening.ScreeningError):
            screening._input(search)


if __name__ == "__main__":
    unittest.main()
