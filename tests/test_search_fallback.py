"""Provider substitution must preserve observed versions and offline replay."""

import json
from pathlib import Path
import tempfile
import unittest

import httpx
from fastapi.testclient import TestClient

from backend.app.api import create_app
from backend.app.db import open_database
from backend.app.collections.submission import SearchChoice, create_collection_with_import
from backend.app.ingestion.selection import select_fixed_version
from backend.app.jobs.worker import claim_next_search, execute_claimed_search
from backend.app.jobs.service import submit_search
from backend.app.search.client import ArxivClient, ArxivCooldownError, TransportResponse
from backend.app.search.failure import read_search_failure
from backend.app.search.fallback import OpenAlexArxivFallback, parse_abstract_page, run_resilient_search
from backend.app.search.service import InvalidSearchSnapshot, publish_search_cache, replay_search_run
from backend.app.search.query_builder import build_query_plan
from tests.test_query_plan import EXPANSIONS


PAPER = "2402.03480"
PAGE = b'''<html><head>
<meta name="citation_arxiv_id" content="2402.03480">
<meta name="citation_title" content="AI Infrastructure Survey">
<meta name="citation_author" content="Example, Author">
<meta name="citation_abstract" content="We survey infrastructure for scientific discovery.">
<meta name="citation_pdf_url" content="https://arxiv.org/pdf/2402.03480">
<meta name="citation_date" content="2024/02/05">
<meta name="citation_online_date" content="2024/04/01"></head>
<body><div class="submission-history"><strong>[v1]</strong><br><strong>[v3]</strong></div></body></html>'''


def discovery():
    return {"results": [{"id": "https://openalex.org/W1", "locations": [{
        "source": {"id": "https://openalex.org/S4306400194"},
        "landing_page_url": f"https://arxiv.org/abs/{PAPER}"}]}]}


class Pages:
    def __init__(self, body=PAGE, status=200):
        self.calls, self.body, self.status = [], body, status

    def send(self, request):
        self.calls.append(request)
        return TransportResponse(self.status, self.body, ())

    def close(self):
        pass


class Primary:
    def __init__(self, success=False):
        self.calls, self.success = 0, success

    def send(self, request):
        self.calls += 1
        if self.success:
            body = (Path(__file__).parent / "fixtures/arxiv/normal_with_versions.xml").read_bytes()
            return TransportResponse(200, body, ())
        return TransportResponse(429, b"Rate exceeded.", ())


class SearchFallbackTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.requests = []
        self.pages = Pages()

        def handle(request):
            self.requests.append(request)
            return httpx.Response(200, json=discovery())

        self.http = httpx.Client(transport=httpx.MockTransport(handle))
        self.addCleanup(self.http.close)
        self.fallback = OpenAlexArxivFallback(client=self.http, page_transport=self.pages)
        self.primary = Primary()
        self.client = ArxivClient(transport=self.primary, max_attempts=1)

    def run_search(self, run_id="search-one", **kwargs):
        return run_resilient_search("AI infrastructure survey", client=self.client,
            fallback=self.fallback, output_root=self.root, run_id=run_id, **kwargs)

    def test_fallback_retains_failure_replays_and_justifies_fixed_pdf(self):
        run = self.run_search()
        self.assertEqual(run.status, "success")
        replay = replay_search_run(run.run_dir)
        self.assertTrue(replay.candidates_match)
        self.assertIsNone(replay.fusion)
        self.assertEqual(replay.provenance["provider"], "openalex_arxiv")
        self.assertEqual(replay.provenance["primary_failure"]["kind"], "rate_limited")
        self.assertEqual(replay.provenance["primary_failure"]["http_attempts"], 1)
        self.assertEqual(select_fixed_version(run.run_dir, f"{PAPER}v3").download_url,
            f"https://arxiv.org/pdf/{PAPER}v3")
        self.assertTrue((self.root / "search-one-arxiv/response.atom").exists())
        self.assertEqual(self.primary.calls, 1)
        self.assertEqual(len(self.pages.calls), 1)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0].url.params["filter"], "locations.source.id:S4306400194")

    def test_primary_success_does_not_call_fallback(self):
        self.primary.success = True
        run = self.run_search()
        replay = replay_search_run(run.run_dir)
        self.assertTrue(replay.candidates_match)
        self.assertFalse(replay.provenance["fallback_used"])
        self.assertEqual(self.requests, [])
        self.assertEqual(self.pages.calls, [])

    def test_only_published_fallback_cache_skips_network(self):
        first = self.run_search()
        self.assertFalse(first.cache_hit)
        second = self.run_search("search-two")
        self.assertFalse(second.cache_hit)
        publish_search_cache(first.run_dir)
        third = self.run_search("search-three")
        self.assertTrue(third.cache_hit)
        self.assertTrue(replay_search_run(third.run_dir).candidates_match)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(self.pages.calls), 2)

    def test_bad_or_unknown_versions_never_become_v1(self):
        for body in (PAGE.replace(b"[v1]", b"?").replace(b"[v3]", b"?"),
                     PAGE.replace(b'content="2402.03480"', b'content="1706.03762"'),
                     PAGE.replace(b"https://arxiv.org/pdf/2402.03480", b"https://evil.example/pdf/2402.03480")):
            with self.assertRaises(ValueError):
                parse_abstract_page(body, PAPER, 1)

    def test_tampered_html_or_source_path_is_rejected(self):
        run = self.run_search()
        html = self.root / "search-one-openalex/page-1.html"
        html.write_bytes(PAGE + b"tampered")
        with self.assertRaises(InvalidSearchSnapshot):
            replay_search_run(run.run_dir)

    def test_multi_query_failure_falls_back_without_claiming_rrf(self):
        plan = build_query_plan("coding agent", EXPANSIONS)
        run = run_resilient_search("coding agent", client=self.client, fallback=self.fallback,
            output_root=self.root, run_id="search-multi", plan=plan)
        replay = replay_search_run(run.run_dir)
        self.assertTrue(replay.candidates_match)
        self.assertIsNone(replay.fusion)
        self.assertEqual(replay.provenance["original_plan"], plan.as_data())
        self.assertEqual(replay.provenance["primary_failure"]["http_attempts"], 1)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0].url.params["search"], "coding agent")

    def test_primary_failure_body_tampering_blocks_replay(self):
        run = self.run_search()
        (self.root / "search-one-arxiv/response.atom").write_bytes(b"changed")
        with self.assertRaises(InvalidSearchSnapshot):
            replay_search_run(run.run_dir)

    def test_discovery_or_candidate_tampering_blocks_replay(self):
        run = self.run_search()
        (self.root / "search-one-openalex/openalex.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(InvalidSearchSnapshot):
            replay_search_run(run.run_dir)

    def test_unverified_candidate_is_skipped_but_never_imported(self):
        self.pages.body = PAGE.replace(b"[v1]", b"?").replace(b"[v3]", b"?")
        run = self.run_search()
        self.assertEqual(run.status, "failed")
        self.assertEqual(json.loads((run.run_dir / "candidates.json").read_text()), [])

    def test_both_failures_remain_readable_without_retry(self):
        self.pages.status = 429
        run = self.run_search()
        self.assertEqual(run.status, "failed")
        failure = read_search_failure(run.run_dir, expected_run_id=run.run_id,
            query_text="AI infrastructure survey", query_plan=None)
        self.assertIn("备用检索", failure["message"])
        self.assertEqual(failure["http_attempts"], 1)
        self.assertEqual(failure["fallback_requests"], 1)
        self.assertEqual(len(self.pages.calls), 1)

    def test_page_cooldown_does_not_claim_an_http_request_was_sent(self):
        def blocked(request):
            raise ArxivCooldownError("Pages endpoint is cooling down.")
        self.pages.send = blocked
        run = self.run_search()
        self.assertEqual(run.status, "failed")
        manifest = json.loads((self.root / "search-one-openalex/manifest.json").read_text())
        self.assertEqual(manifest["page_requests"], 0)

    def test_worker_api_read_and_import_share_wrapper_snapshot(self):
        engine = open_database(self.root / "test.sqlite")
        self.addCleanup(engine.dispose)
        data = self.root / "data"
        with TestClient(create_app(self.root / "test.sqlite", data)) as api:
            response = api.post("/api/searches", json={"question": "AI 基础设施", "query_text": "AI infrastructure survey"},
                headers={"Idempotency-Key": "fallback-test"})
            self.assertEqual(response.status_code, 202, response.text)
            claimed = claim_next_search(engine, worker_id="test")
            executed = execute_claimed_search(engine, claimed, data_root=data,
                client=self.client, fallback=self.fallback)
            self.assertEqual(executed.status, "succeeded")
            state = api.get(f"/api/searches/{response.json()['search_id']}")
            self.assertEqual(state.status_code, 200, state.text)
            self.assertEqual(state.json()["candidates"][0]["versioned_id"], f"{PAPER}v3")
            self.assertTrue(state.json()["provenance"]["fallback_used"])
            selected = select_fixed_version(data / "runs" / f"search-{claimed.attempt_id}", f"{PAPER}v3")
            self.assertEqual(selected.snapshot_sha256, state.json()["snapshot_sha256"])
            collection = create_collection_with_import(engine, data_root=data,
                choices=(SearchChoice(response.json()["search_id"], f"{PAPER}v3"),),
                idempotency_key="import-fallback")
            self.assertTrue(collection.job_id)

    def test_multi_query_api_exposes_actual_source_and_original_plan(self):
        engine = open_database(self.root / "multi.sqlite")
        self.addCleanup(engine.dispose)
        data = self.root / "multi-data"
        plan = build_query_plan("coding agent", EXPANSIONS)
        submitted = submit_search(engine, question="代码 Agent", query_text="coding agent",
            query_plan=plan.as_data(), idempotency_key="multi-fallback")
        claimed = claim_next_search(engine, worker_id="multi-test")
        execute_claimed_search(engine, claimed, data_root=data, client=self.client, fallback=self.fallback)
        with TestClient(create_app(self.root / "multi.sqlite", data)) as api:
            response = api.get(f"/api/searches/{submitted.search_id}")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["query_plan"], plan.as_data())
            self.assertIsNone(response.json()["fusion"])
            self.assertTrue(response.json()["provenance"]["fallback_used"])


if __name__ == "__main__":
    unittest.main()
