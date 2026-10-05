import hashlib
from contextlib import redirect_stdout
from io import StringIO
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.search.client import (
    ArxivClient,
    ArxivSearchError,
    ArxivTransportError,
    HttpxArxivTransport,
    TransportResponse,
    build_arxiv_request,
)
from backend.app.search.service import build_arxiv_query, run_search
from scripts.arxiv_search import main as search_main


FIXTURES = Path(__file__).parent / "fixtures" / "arxiv"
FIXED_TIME = datetime(2026, 9, 22, 8, 30, tzinfo=timezone.utc)


class SequenceTransport:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def send(self, request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class MonotonicSequence:
    def __init__(self, *values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class AdvancingClock:
    def __init__(self):
        self.seconds = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.seconds

    def utc(self):
        return FIXED_TIME + timedelta(seconds=self.seconds)

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.seconds += seconds


def response(
    status_code: int,
    filename: str | None = None,
    extra_headers: tuple[tuple[str, str], ...] = (),
) -> TransportResponse:
    body = (FIXTURES / filename).read_bytes() if filename else b"temporary failure"
    return TransportResponse(
        status_code=status_code,
        body=body,
        headers=(("Content-Type", "application/atom+xml"), *extra_headers),
    )


class ArxivClientTest(unittest.TestCase):
    def test_429_without_valid_retry_after_stops_and_blocks_following_queries(self):
        for headers in ((), (("Retry-After", "invalid"),)):
            for max_attempts in (1, 3):
                with self.subTest(headers=headers, max_attempts=max_attempts):
                    clock = AdvancingClock()
                    transport = SequenceTransport(
                        response(429, extra_headers=headers), response(200, "empty.xml")
                    )
                    client = ArxivClient(transport=transport, max_attempts=max_attempts,
                        clock=clock.utc, monotonic=clock.monotonic, sleeper=clock.sleep)
                    with self.assertRaises(ArxivSearchError) as first:
                        client.fetch(build_arxiv_query("knowledge graph"))
                    self.assertEqual(first.exception.status_code, 429)
                    self.assertEqual(len(first.exception.attempts), 1)
                    with self.assertRaises(ArxivSearchError) as second:
                        client.fetch(build_arxiv_query("semantic graph"))
                    self.assertEqual(second.exception.kind, "cooldown")
                    self.assertEqual(second.exception.attempts, ())
                    self.assertEqual(len(transport.requests), 1)
                    self.assertEqual(clock.sleeps, [])
                    clock.seconds += 120
                    self.assertEqual(client.fetch(build_arxiv_query("knowledge graph")).feed.candidates, ())
                    self.assertEqual(len(transport.requests), 2)

    def test_transient_retries_back_off_from_three_to_six_seconds(self):
        clock = AdvancingClock()

        class TimedTransport(SequenceTransport):
            def send(self, request):
                send_times.append(clock.seconds)
                return super().send(request)

        send_times = []
        transport = TimedTransport(
            ArxivTransportError("timed out"),
            response(503),
            response(200, "empty.xml"),
        )
        client = ArxivClient(
            transport=transport,
            clock=clock.utc,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
        )

        client.fetch(build_arxiv_query("LLM agent memory"))

        self.assertEqual(send_times, [0.0, 3.0, 9.0])
        self.assertEqual(clock.sleeps, [3.0, 6.0])

    def test_retry_after_seconds_delays_the_next_attempt(self):
        clock = AdvancingClock()
        send_times = []

        class TimedTransport(SequenceTransport):
            def send(self, request):
                send_times.append(clock.seconds)
                return super().send(request)

        client = ArxivClient(
            transport=TimedTransport(
                response(429, extra_headers=(("rEtRy-AfTeR", "8"),)),
                response(200, "empty.xml"),
            ),
            clock=clock.utc,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
        )

        client.fetch(build_arxiv_query("LLM agent memory"))

        self.assertEqual(send_times, [0.0, 8.0])

    def test_retry_after_http_date_delays_the_next_attempt(self):
        clock = AdvancingClock()
        send_times = []
        retry_at = format_datetime(FIXED_TIME + timedelta(seconds=10), usegmt=True)

        class TimedTransport(SequenceTransport):
            def send(self, request):
                send_times.append(clock.seconds)
                return super().send(request)

        client = ArxivClient(
            transport=TimedTransport(
                response(503, extra_headers=(("Retry-After", retry_at),)),
                response(200, "empty.xml"),
            ),
            clock=clock.utc,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
        )

        client.fetch(build_arxiv_query("LLM agent memory"))

        self.assertEqual(send_times, [0.0, 10.0])

    def test_long_retry_after_stops_automatic_retry_and_keeps_response(self):
        clock = AdvancingClock()
        transport = SequenceTransport(
            response(429, extra_headers=(("Retry-After", "120"),)),
            response(200, "empty.xml"),
        )
        client = ArxivClient(
            transport=transport,
            clock=clock.utc,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
        )

        with self.assertRaises(ArxivSearchError) as raised:
            client.fetch(build_arxiv_query("LLM agent memory"))

        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(len(raised.exception.attempts), 1)
        self.assertIn("Retry-After", str(raised.exception))
        self.assertEqual(len(transport.requests), 1)
        self.assertEqual(clock.sleeps, [])

    def test_httpx_transport_sends_params_and_returns_raw_response(self):
        body = (FIXTURES / "empty.xml").read_bytes()

        def handle(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.host, "export.arxiv.org")
            self.assertEqual(request.url.params["max_results"], "20")
            self.assertEqual(request.url.params["sortBy"], "relevance")
            self.assertEqual(request.headers["User-Agent"], "PaperTrail/0.1")
            return httpx.Response(
                200,
                content=body,
                headers={"Content-Type": "application/atom+xml"},
            )

        transport = HttpxArxivTransport(
            http_transport=httpx.MockTransport(handle), request_gate=None
        )
        actual = transport.send(build_arxiv_request(build_arxiv_query("LLM agent memory")))

        self.assertEqual(actual.status_code, 200)
        self.assertEqual(actual.body, body)
        self.assertIn(("content-type", "application/atom+xml"), actual.headers)

    def test_request_has_fixed_search_contract(self):
        request = build_arxiv_request(build_arxiv_query("LLM agent memory"))

        self.assertEqual(request.endpoint, "https://export.arxiv.org/api/query")
        self.assertEqual(
            request.params,
            (
                (
                    "search_query",
                    "(ti:LLM OR abs:LLM) AND (ti:agent OR abs:agent) "
                    "AND (ti:memory OR abs:memory)",
                ),
                ("start", "0"),
                ("max_results", "20"),
                ("sortBy", "relevance"),
                ("sortOrder", "descending"),
            ),
        )
        self.assertEqual(request.timeout_seconds, 20.0)
        self.assertEqual(request.headers, (("User-Agent", "PaperTrail/0.1"),))

    def test_transient_failures_keep_attempt_history_and_recover(self):
        transport = SequenceTransport(
            ArxivTransportError("timed out"),
            response(503),
            response(200, "normal_with_versions.xml"),
        )
        clock = AdvancingClock()
        client = ArxivClient(
            transport=transport,
            clock=clock.utc,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
        )

        result = client.fetch(build_arxiv_query("LLM agent memory"))

        self.assertEqual([item.outcome for item in result.attempts], [
            "transport_error", "http_error", "success"
        ])
        self.assertEqual([item.status_code for item in result.attempts], [None, 503, 200])
        self.assertEqual(clock.sleeps, [3.0, 6.0])
        self.assertEqual(len(transport.requests), 3)
        self.assertEqual(len(result.feed.candidates), 2)

    def test_nonretryable_http_and_atom_failures_stop_after_one_attempt(self):
        cases = (
            (response(400), "http_error"),
            (response(200, "error.xml"), "api_error"),
            (response(200, "malformed.xml"), "invalid_feed"),
        )
        for first_response, expected_kind in cases:
            with self.subTest(expected_kind=expected_kind):
                transport = SequenceTransport(
                    first_response,
                    response(200, "normal_with_versions.xml"),
                )
                client = ArxivClient(
                    transport=transport,
                    clock=lambda: FIXED_TIME,
                    monotonic=MonotonicSequence(0.0),
                    sleeper=lambda seconds: None,
                )

                with self.assertRaises(ArxivSearchError) as raised:
                    client.fetch(build_arxiv_query("LLM agent memory"))

                self.assertEqual(raised.exception.kind, expected_kind)
                self.assertEqual(len(raised.exception.attempts), 1)
                self.assertEqual(len(transport.requests), 1)
                self.assertIsNotNone(raised.exception.response_body)

    def test_successful_run_saves_request_raw_response_and_candidates(self):
        body = (FIXTURES / "normal_with_versions.xml").read_bytes()
        client = ArxivClient(
            transport=SequenceTransport(response(200, "normal_with_versions.xml")),
            clock=lambda: FIXED_TIME,
            monotonic=MonotonicSequence(0.0),
            sleeper=lambda seconds: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            run = run_search(
                "LLM agent memory",
                client=client,
                output_root=Path(directory),
                run_id="search-fixture",
                clock=lambda: FIXED_TIME,
            )

            self.assertEqual(run.status, "success")
            self.assertEqual((run.run_dir / "response.atom").read_bytes(), body)
            request_data = json.loads((run.run_dir / "request.json").read_text(encoding="utf-8"))
            run_data = json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))
            candidates = json.loads((run.run_dir / "candidates.json").read_text(encoding="utf-8"))
            self.assertEqual(request_data["query_terms"], ["LLM", "agent", "memory"])
            self.assertEqual(request_data["query_contract_version"], 1)
            self.assertEqual(run_data["created_at"], "2026-09-22T08:30:00Z")
            self.assertEqual(run_data["fetched_at"], "2026-09-22T08:30:00Z")
            self.assertEqual(run_data["parser_version"], 2)
            self.assertEqual(run_data["snapshot"]["source"], "network")
            self.assertEqual(run_data["snapshot"]["sha256"], hashlib.sha256(body).hexdigest())
            self.assertEqual(run_data["attempts"][0]["outcome"], "success")
            self.assertEqual(candidates[0]["versioned_id"], "2401.00001v3")

    def test_failed_parse_saves_raw_response_and_failed_status(self):
        client = ArxivClient(
            transport=SequenceTransport(response(200, "malformed.xml")),
            clock=lambda: FIXED_TIME,
            monotonic=MonotonicSequence(0.0),
            sleeper=lambda seconds: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            run = run_search(
                "LLM agent memory",
                client=client,
                output_root=Path(directory),
                run_id="search-failed-fixture",
                clock=lambda: FIXED_TIME,
            )

            self.assertEqual(run.status, "failed")
            self.assertEqual(run.error_kind, "invalid_feed")
            self.assertTrue((run.run_dir / "response.atom").exists())
            run_data = json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_data["status"], "failed")
            self.assertEqual(run_data["error"]["kind"], "invalid_feed")
            self.assertNotEqual(run_data["candidate_count"], 0)

    def test_cli_preview_does_not_create_a_run(self):
        with tempfile.TemporaryDirectory() as directory:
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = search_main([
                    "--query", "LLM agent memory",
                    "--output-root", directory,
                ])

            self.assertEqual(exit_code, 0)
            self.assertIn("(ti:LLM OR abs:LLM)", stdout.getvalue())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_live_cli_prints_candidates_and_saved_run(self):
        client = ArxivClient(
            transport=SequenceTransport(response(200, "normal_with_versions.xml")),
            clock=lambda: FIXED_TIME,
            monotonic=MonotonicSequence(0.0),
            sleeper=lambda seconds: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = search_main([
                    "--query", "LLM agent memory",
                    "--output-root", directory,
                    "--live",
                ], client=client)

            self.assertEqual(exit_code, 0)
            self.assertIn("Memory for Agents: Revised Version", stdout.getvalue())
            self.assertIn("2401.00001v3", stdout.getvalue())
            self.assertEqual(len(list(Path(directory).iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
