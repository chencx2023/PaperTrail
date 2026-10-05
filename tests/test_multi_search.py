import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from backend.app.search.client import ArxivClient, ArxivTransportError, TransportResponse
from backend.app.search.multi_query import run_multi_search
from backend.app.search.query_builder import build_query_plan
from backend.app.search.service import InvalidSearchSnapshot, publish_search_cache, replay_search_run
from backend.app.ingestion.selection import select_fixed_version
from tests.test_query_plan import EXPANSIONS

FIXTURES = Path(__file__).parent / "fixtures/arxiv"


class Responses:
    def __init__(self, statuses=(200, 200, 200), fixture="normal_with_versions.xml"):
        self.statuses = iter(statuses)
        self.calls = []
        self.body = (FIXTURES / fixture).read_bytes()

    def send(self, request):
        self.calls.append(request)
        status = next(self.statuses)
        if status is None:
            raise ArxivTransportError("Fixed transport failure without a response.")
        return TransportResponse(status, self.body if status == 200 else b"upstream failure", ())


class MultiSearchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = build_query_plan("coding agent", EXPANSIONS)

    def run_multi(self, transport, run_id="search-fixed"):
        return run_multi_search(self.plan, client=ArxivClient(transport=transport, max_attempts=1,
            minimum_interval_seconds=0), output_root=self.root, run_id=run_id)

    def test_three_bounded_calls_replay_without_requests_and_fixed_versions_survive(self):
        transport = Responses()
        run = self.run_multi(transport)
        self.assertEqual(run.status, "success")
        self.assertEqual(len(transport.calls), 3)
        self.assertTrue(all(dict(r.params)["max_results"] == "20" for r in transport.calls))
        replay = replay_search_run(run.run_dir)
        self.assertTrue(replay.candidates_match)
        self.assertEqual(replay.fusion["http_attempts"], 3)
        self.assertEqual(replay.fusion["successful_queries"], 3)
        for candidate in replay.feed.candidates:
            for version in candidate.versions:
                if version.version is not None:
                    source = select_fixed_version(run.run_dir, version.versioned_id)
                    self.assertEqual(source.snapshot_sha256, replay.snapshot_sha256)
        publish_search_cache(run.run_dir)
        other = self.run_multi(transport, "search-next")
        self.assertTrue(other.cache_hit)
        self.assertEqual(len(transport.calls), 3)

    def test_partial_failure_keeps_successful_pool_and_marks_missing_query(self):
        run = self.run_multi(Responses((200, 400, 200)))
        self.assertEqual(run.status, "success")
        fusion = replay_search_run(run.run_dir).fusion
        self.assertTrue(fusion["partial"])
        self.assertEqual(fusion["successful_queries"], 2)
        self.assertEqual(fusion["queries"][1]["status"], "failed")
        self.assertTrue(all(2 not in [m["query_index"] for m in r["matches"]] for r in fusion["ranking"]))

    def test_rate_limit_stops_remaining_network_queries_and_preserves_three_records(self):
        transport = Responses((429,))
        run = self.run_multi(transport)
        self.assertEqual(run.status, "failed")
        self.assertEqual(len(transport.calls), 1)
        manifest = json.loads((run.run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([q["error_kind"] for q in manifest["queries"]],
                         ["http_error", "cooldown", "cooldown"])
        self.assertEqual([q["http_attempts"] for q in manifest["queries"]], [1, 0, 0])
        self.assertTrue(all((self.root / q["run_id"] / "run.json").is_file()
                            for q in manifest["queries"]))

    def test_rate_limit_retains_completed_query_and_its_committed_cache(self):
        transport = Responses((200, 429))
        run = self.run_multi(transport)
        fusion = replay_search_run(run.run_dir).fusion
        self.assertTrue(fusion["partial"])
        self.assertEqual(fusion["successful_queries"], 1)
        self.assertEqual(fusion["http_attempts"], 2)
        self.assertEqual(len(transport.calls), 2)
        publish_search_cache(run.run_dir)
        retry_transport = Responses((429,))
        retry_run = self.run_multi(retry_transport, "search-retry")
        retry_fusion = replay_search_run(retry_run.run_dir).fusion
        self.assertEqual(retry_fusion["cache_hits"], 1)
        self.assertEqual(retry_fusion["successful_queries"], 1)
        self.assertEqual(len(retry_transport.calls), 1)

    def test_partial_failure_response_is_verified_even_though_not_used_as_candidates(self):
        for index, remove in enumerate((False, True)):
            run = self.run_multi(Responses((200, 400, 200)), f"search-partial-{index}")
            response = self.root / f"{run.run_id}-q2" / "response.atom"
            if remove:
                response.unlink()
            else:
                response.write_bytes(b"different failed response")
            with self.subTest(remove=remove), self.assertRaises(InvalidSearchSnapshot):
                replay_search_run(run.run_dir)

    def test_partial_transport_failure_without_response_remains_replayable(self):
        run = self.run_multi(Responses((200, None, 200)))
        fusion = replay_search_run(run.run_dir).fusion
        self.assertTrue(fusion["partial"])
        self.assertEqual(fusion["successful_queries"], 2)

    def test_all_failed_is_not_a_successful_empty_search_and_empty_does_not_expand(self):
        failed = self.run_multi(Responses((400, 400, 400)))
        self.assertEqual(failed.status, "failed")
        with self.assertRaises(InvalidSearchSnapshot):
            replay_search_run(failed.run_dir)
        empty_transport = Responses(fixture="empty.xml")
        empty = self.run_multi(empty_transport, "search-empty")
        self.assertEqual(len(empty_transport.calls), 3)
        self.assertEqual(replay_search_run(empty.run_dir).fusion["relaxation_requests"], 0)

    def test_tampered_source_or_fusion_manifest_cannot_be_used(self):
        run = self.run_multi(Responses())
        child = self.root / "search-fixed-q1" / "response.atom"
        child.write_bytes(child.read_bytes() + b" ")
        with self.assertRaises(InvalidSearchSnapshot):
            replay_search_run(run.run_dir)

    def test_original_order_must_still_match_the_verified_atom_source(self):
        run = self.run_multi(Responses())
        path = run.run_dir / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["queries"][0]["original_order"][0]["rank"] = 999
        path.write_text(json.dumps(manifest), encoding="utf-8")
        metadata_path = run.run_dir / "run.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["snapshot"].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), byte_count=path.stat().st_size)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(InvalidSearchSnapshot):
            replay_search_run(run.run_dir)

    def test_saved_pool_report_keeps_discovery_and_sorting_comparisons_separate(self):
        from scripts.replay_multi_search import comparison_report
        run = self.run_multi(Responses())
        result = comparison_report(run.run_dir)
        self.assertEqual(result["budget"]["http_attempts"], 3)
        self.assertIsNone(result["quality"])
        self.assertIsNone(result["budget"]["model_tokens"])
        self.assertEqual(set(result["rankings"]["simple_merge"]), set(result["rankings"]["rrf"]))
        self.assertEqual(result["source_snapshot_sha256"], replay_search_run(run.run_dir).snapshot_sha256)


if __name__ == "__main__":
    unittest.main()
