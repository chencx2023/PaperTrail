import json
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.search.client import ArxivClient, TransportResponse
from backend.app.search.service import (
    InvalidSearchSnapshot,
    replay_search_run,
    run_search,
)
from scripts.arxiv_search import main as search_main


FIXTURES = Path(__file__).parent / "fixtures" / "arxiv"
FETCHED_AT = datetime(2026, 9, 22, 8, 30, tzinfo=timezone.utc)


class SequenceTransport:
    def __init__(self, *responses: TransportResponse) -> None:
        self.responses = list(responses)
        self.requests = []

    def send(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


class NetworkMustNotRun:
    def fetch(self, plan):
        raise AssertionError("cache hit attempted a network request")


def fixture_response() -> TransportResponse:
    return TransportResponse(
        status_code=200,
        body=(FIXTURES / "normal_with_versions.xml").read_bytes(),
        headers=(("Content-Type", "application/atom+xml"),),
    )


def fixture_client(at: datetime) -> tuple[ArxivClient, SequenceTransport]:
    transport = SequenceTransport(fixture_response())
    client = ArxivClient(
        transport=transport,
        clock=lambda: at,
        monotonic=lambda: 0.0,
        sleeper=lambda seconds: None,
    )
    return client, transport


class SearchCacheTest(unittest.TestCase):
    def create_source_run(self, output_root: Path):
        client, transport = fixture_client(FETCHED_AT)
        run = run_search(
            "LLM agent memory",
            client=client,
            output_root=output_root,
            run_id="search-source",
            clock=lambda: FETCHED_AT,
        )
        self.assertEqual(len(transport.requests), 1)
        return run

    def test_fresh_cache_hit_creates_a_self_contained_run_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            source = self.create_source_run(output_root)
            created_at = FETCHED_AT + timedelta(hours=2)

            cached = run_search(
                "LLM agent memory",
                client=NetworkMustNotRun(),
                output_root=output_root,
                run_id="search-cache-hit",
                clock=lambda: created_at,
            )

            self.assertTrue(cached.cache_hit)
            self.assertEqual(cached.snapshot_source, "cache")
            self.assertEqual(cached.created_at, "2026-09-22T10:30:00Z")
            self.assertEqual(cached.fetched_at, "2026-09-22T08:30:00Z")
            self.assertEqual(cached.feed.candidates, source.feed.candidates)
            self.assertEqual(
                (cached.run_dir / "response.atom").read_bytes(),
                (source.run_dir / "response.atom").read_bytes(),
            )

            run_data = json.loads(
                (cached.run_dir / "run.json").read_text(encoding="utf-8")
            )
            request_data = json.loads(
                (cached.run_dir / "request.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_data["attempts"], [])
            self.assertIsNone(run_data["http_status"])
            self.assertEqual(run_data["snapshot"]["source"], "cache")
            self.assertEqual(run_data["snapshot"]["origin_run_id"], source.run_id)
            self.assertEqual(run_data["snapshot"]["origin_source"], "network")
            self.assertEqual(run_data["cache"]["age_seconds"], 7200)
            self.assertEqual(len(request_data["cache_key"]), 64)

    def test_cache_expires_at_twenty_four_hours(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            self.create_source_run(output_root)
            refreshed_at = FETCHED_AT + timedelta(hours=24)
            client, transport = fixture_client(refreshed_at)

            refreshed = run_search(
                "LLM agent memory",
                client=client,
                output_root=output_root,
                run_id="search-refreshed",
                clock=lambda: refreshed_at,
            )

            self.assertFalse(refreshed.cache_hit)
            self.assertEqual(refreshed.snapshot_source, "network")
            self.assertEqual(len(transport.requests), 1)
            self.assertEqual(refreshed.fetched_at, "2026-09-23T08:30:00Z")

    def test_tampered_cached_snapshot_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            source = self.create_source_run(output_root)
            (source.run_dir / "response.atom").write_bytes(b"tampered")
            refreshed_at = FETCHED_AT + timedelta(hours=1)
            client, transport = fixture_client(refreshed_at)

            refreshed = run_search(
                "LLM agent memory",
                client=client,
                output_root=output_root,
                run_id="search-after-tamper",
                clock=lambda: refreshed_at,
            )

            self.assertFalse(refreshed.cache_hit)
            self.assertEqual(len(transport.requests), 1)

    def test_offline_replay_rebuilds_and_compares_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.create_source_run(Path(directory))

            replay = replay_search_run(source.run_dir)

            self.assertTrue(replay.candidates_match)
            self.assertEqual(replay.source_run_id, source.run_id)
            self.assertEqual(len(replay.feed.candidates), 2)
            self.assertEqual(replay.saved_parser_version, 2)
            self.assertEqual(replay.current_parser_version, 2)

            (source.run_dir / "response.atom").write_bytes(b"tampered")
            with self.assertRaises(InvalidSearchSnapshot):
                replay_search_run(source.run_dir)

    def test_offline_replay_rejects_missing_request_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.create_source_run(Path(directory))
            (source.run_dir / "request.json").unlink()

            with self.assertRaises(InvalidSearchSnapshot):
                replay_search_run(source.run_dir)

    def test_cli_replay_is_offline_and_does_not_create_a_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            source = self.create_source_run(output_root)
            before = set(output_root.iterdir())
            stdout = StringIO()

            with redirect_stdout(stdout):
                exit_code = search_main(["--replay", str(source.run_dir)])

            self.assertEqual(exit_code, 0)
            self.assertIn("[REPLAY] 2 candidates", stdout.getvalue())
            self.assertIn("[MATCH] Saved candidates are identical", stdout.getvalue())
            self.assertEqual(set(output_root.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
