import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sys
import tempfile
from threading import Thread
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from backend.app.search.client import (
    ArxivClient,
    ArxivRequest,
    ArxivSearchError,
    HttpxArxivTransport,
    build_arxiv_request,
)
from backend.app.search.service import build_arxiv_query


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMPTY_ATOM = (PROJECT_ROOT / "tests" / "fixtures" / "arxiv" / "empty.xml").read_bytes()


def wait_for_marker(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 5
    while not path.is_file() and time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"child exited: {process.stderr.read()}")
        time.sleep(0.02)
    if not path.is_file():
        raise AssertionError(f"child did not write {path.name}")


class ArxivRequestGateTest(unittest.TestCase):
    def test_429_without_retry_after_defers_another_transport_before_network(self):
        from backend.app.search.request_gate import ArxivRequestGate

        calls = []
        def limited(_request):
            calls.append("sent")
            return httpx.Response(429, content=b"Rate exceeded.")

        with tempfile.TemporaryDirectory() as directory:
            gate_path = Path(directory) / "arxiv-api.lock"
            first = HttpxArxivTransport(http_transport=httpx.MockTransport(limited),
                request_gate=ArxivRequestGate(gate_path, minimum_interval_seconds=0))
            second = HttpxArxivTransport(http_transport=httpx.MockTransport(limited),
                request_gate=ArxivRequestGate(gate_path, minimum_interval_seconds=0))
            try:
                request = build_arxiv_request(build_arxiv_query("knowledge graph"))
                self.assertEqual(first.send(request).status_code, 429)
                with patch("backend.app.search.request_gate.time.sleep",
                    side_effect=AssertionError("cooldown must not keep the worker waiting")):
                    with self.assertRaises(ArxivSearchError) as raised:
                        ArxivClient(transport=second).fetch(build_arxiv_query("semantic graph"))
                self.assertEqual(raised.exception.kind, "cooldown")
                self.assertEqual(raised.exception.attempts, ())
            finally:
                first.close()
                second.close()
        self.assertEqual(calls, ["sent"])

    def test_persisted_deadline_uses_wall_time_after_monotonic_clock_resets(self):
        from backend.app.search.request_gate import ArxivRequestGate

        class RestartClock:
            wall = 1_700_000_000.0
            uptime = 10_000.0

            def time(self):
                return self.wall

            def monotonic(self):
                return self.uptime

        clock = RestartClock()
        with tempfile.TemporaryDirectory() as directory:
            gate = ArxivRequestGate(Path(directory) / "arxiv-api.lock")
            with patch("backend.app.search.request_gate.time", clock):
                gate._write_deadline(clock.monotonic() + 120)
                clock.wall += 10
                clock.uptime = 5
                self.assertAlmostEqual(gate._next_allowed(), 115)

    def test_preboot_v2_monotonic_deadline_does_not_block_new_boot(self):
        from backend.app.search.request_gate import ArxivRequestGate

        class RestartClock:
            def time(self):
                return 1_700_000_000.0

            def monotonic(self):
                return 5_000.0

        with tempfile.TemporaryDirectory() as directory:
            gate = ArxivRequestGate(Path(directory) / "arxiv-api.lock")
            gate.state_path.write_text(
                json.dumps({"version": 2, "next_allowed_monotonic": 23_322.14}),
                encoding="ascii",
            )
            # Even a state file written immediately before reboot is stale.
            old_time = 1_699_994_999.0
            os.utime(gate.state_path, (old_time, old_time))
            with patch("backend.app.search.request_gate.time", RestartClock()):
                self.assertIsNone(gate._next_allowed())

    def test_current_boot_v2_deadline_remains_active(self):
        from backend.app.search.request_gate import ArxivRequestGate

        class CurrentClock:
            def time(self):
                return 1_700_000_000.0

            def monotonic(self):
                return 5_000.0

        with tempfile.TemporaryDirectory() as directory:
            gate = ArxivRequestGate(Path(directory) / "arxiv-api.lock")
            gate.state_path.write_text(
                json.dumps({"version": 2, "next_allowed_monotonic": 5_120.0}),
                encoding="ascii",
            )
            same_boot_time = 1_699_995_010.0
            os.utime(gate.state_path, (same_boot_time, same_boot_time))
            with patch("backend.app.search.request_gate.time", CurrentClock()):
                self.assertEqual(gate._next_allowed(), 5_120.0)

    def test_seed_diagnostic_records_retry_after_in_shared_gate(self):
        from backend.app.search.request_gate import ArxivCooldownActive, ArxivRequestGate
        from scripts.diagnose_search_seeds import _send_intersection_query

        calls = []

        def limited(_request: httpx.Request) -> httpx.Response:
            calls.append("sent")
            return httpx.Response(429, headers={"Retry-After": "120"})

        with tempfile.TemporaryDirectory() as directory:
            gate = ArxivRequestGate(
                Path(directory) / "arxiv-api.lock", minimum_interval_seconds=0.05
            )
            with httpx.Client(transport=httpx.MockTransport(limited)) as client:
                response, _started_at = _send_intersection_query(
                    client, gate, "all:memory", ["2401.00001"]
                )
                self.assertEqual(response.status_code, 429)
                with self.assertRaises(ArxivCooldownActive):
                    _send_intersection_query(
                        client, gate, "all:memory", ["2401.00001"]
                    )
        self.assertEqual(calls, ["sent"])

    def test_long_shared_cooldown_fails_next_job_before_network(self):
        from backend.app.search.request_gate import ArxivRequestGate

        calls = []

        def handle(_request: httpx.Request) -> httpx.Response:
            calls.append("sent")
            return httpx.Response(200, content=EMPTY_ATOM)

        with tempfile.TemporaryDirectory() as directory:
            gate_path = Path(directory) / "arxiv-api.lock"
            with ArxivRequestGate(gate_path, minimum_interval_seconds=0.05).slot() as slot:
                slot.defer_for(120)
            transport = HttpxArxivTransport(
                http_transport=httpx.MockTransport(handle),
                request_gate=ArxivRequestGate(gate_path, minimum_interval_seconds=0.05),
            )
            try:
                with patch(
                    "backend.app.search.request_gate.time.sleep",
                    side_effect=AssertionError("long cooldown must not block a worker"),
                ):
                    with self.assertRaises(ArxivSearchError) as raised:
                        ArxivClient(
                            transport=transport,
                            minimum_interval_seconds=0,
                        ).fetch(build_arxiv_query("LLM agent memory"))
            finally:
                transport.close()

        self.assertEqual(raised.exception.kind, "cooldown")
        self.assertEqual(raised.exception.attempts, ())
        self.assertEqual(calls, [])

    def test_retry_after_defers_other_transport_using_the_same_gate_file(self):
        from backend.app.search.request_gate import ArxivRequestGate

        starts = []

        def limited(_request: httpx.Request) -> httpx.Response:
            starts.append(time.monotonic())
            return httpx.Response(429, headers={"Retry-After": "1"})

        def succeeding(_request: httpx.Request) -> httpx.Response:
            starts.append(time.monotonic())
            return httpx.Response(200, content=EMPTY_ATOM)

        with tempfile.TemporaryDirectory() as directory:
            gate_path = Path(directory) / "arxiv-api.lock"
            first = HttpxArxivTransport(
                http_transport=httpx.MockTransport(limited),
                request_gate=ArxivRequestGate(gate_path, minimum_interval_seconds=0.05),
            )
            second = HttpxArxivTransport(
                http_transport=httpx.MockTransport(succeeding),
                request_gate=ArxivRequestGate(gate_path, minimum_interval_seconds=0.05),
            )
            request = build_arxiv_request(build_arxiv_query("LLM agent memory"))
            try:
                self.assertEqual(first.send(request).status_code, 429)
                self.assertEqual(second.send(request).status_code, 200)
            finally:
                first.close()
                second.close()

        self.assertEqual(len(starts), 2)
        self.assertGreaterEqual(starts[1] - starts[0], 0.9)

    def test_live_transport_does_not_keep_an_idle_connection_open(self):
        from backend.app.search.request_gate import ArxivRequestGate

        client_ports = []

        class LocalHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                client_ports.append(self.client_address[1])
                self.send_response(200)
                self.send_header("Content-Length", str(len(EMPTY_ATOM)))
                self.end_headers()
                self.wfile.write(EMPTY_ATOM)

            def log_message(self, _format, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), LocalHandler)
        server.daemon_threads = True
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = ArxivRequest(
                endpoint=f"http://127.0.0.1:{server.server_address[1]}/query",
                params=(),
                headers=(),
                timeout_seconds=2.0,
            )
            with tempfile.TemporaryDirectory() as directory:
                with patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost"}):
                    transport = HttpxArxivTransport(
                        request_gate=ArxivRequestGate(
                            Path(directory) / "arxiv-api.lock",
                            minimum_interval_seconds=0,
                        )
                    )
                    try:
                        transport.send(request)
                        transport.send(request)
                    finally:
                        transport.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertEqual(len(client_ports), 2)
        self.assertEqual(len(set(client_ports)), 2)

    def test_redirect_does_not_trigger_an_unmetered_second_request(self):
        from backend.app.search.request_gate import ArxivRequestGate

        calls = []

        def handle(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(
                    302,
                    headers={"Location": "https://export.arxiv.org/api/query?start=1"},
                )
            return httpx.Response(200, content=EMPTY_ATOM)

        with tempfile.TemporaryDirectory() as directory:
            transport = HttpxArxivTransport(
                http_transport=httpx.MockTransport(handle),
                request_gate=ArxivRequestGate(
                    Path(directory) / "arxiv-api.lock",
                    minimum_interval_seconds=0.1,
                ),
            )
            try:
                response = transport.send(
                    build_arxiv_request(build_arxiv_query("LLM agent memory"))
                )
            finally:
                transport.close()

        self.assertEqual(response.status_code, 302)
        self.assertEqual(len(calls), 1)

    def test_other_process_waits_for_request_completion_and_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate_path = root / "arxiv-api.lock"
            ready_path = root / "ready"
            ended_path = root / "ended"
            attempting_path = root / "attempting"
            entered_path = root / "entered"
            holder_code = (
                "from pathlib import Path\n"
                "import sys, time\n"
                "from backend.app.search.request_gate import ArxivRequestGate\n"
                "with ArxivRequestGate(Path(sys.argv[1]), minimum_interval_seconds=0.2).slot():\n"
                "    Path(sys.argv[2]).write_text(str(time.monotonic()))\n"
                "    sys.stdin.read(1)\n"
                "    Path(sys.argv[3]).write_text(str(time.monotonic()))\n"
            )
            contender_code = (
                "from pathlib import Path\n"
                "import sys, time\n"
                "from backend.app.search.request_gate import ArxivRequestGate\n"
                "Path(sys.argv[2]).write_text('attempting')\n"
                "with ArxivRequestGate(Path(sys.argv[1]), minimum_interval_seconds=0.2).slot():\n"
                "    Path(sys.argv[3]).write_text(str(time.monotonic()))\n"
            )
            holder = subprocess.Popen(
                [sys.executable, "-c", holder_code, str(gate_path), str(ready_path), str(ended_path)],
                cwd=PROJECT_ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            contender = None
            try:
                wait_for_marker(ready_path, holder)
                contender = subprocess.Popen(
                    [sys.executable, "-c", contender_code, str(gate_path), str(attempting_path), str(entered_path)],
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                wait_for_marker(attempting_path, contender)
                time.sleep(0.05)
                self.assertFalse(entered_path.exists(), "requests overlapped")

                holder.stdin.write("x")
                holder.stdin.flush()
                holder.communicate(timeout=5)
                contender.communicate(timeout=5)
                self.assertEqual(contender.returncode, 0)
                entered = float(entered_path.read_text())
                ended = float(ended_path.read_text())
                self.assertGreaterEqual(entered - ended, 0.18)
            finally:
                for process in (holder, contender):
                    if process is not None:
                        if process.poll() is None:
                            process.kill()
                        process.communicate(timeout=5)

    def test_transport_retries_use_a_new_shared_slot(self):
        try:
            from backend.app.search.request_gate import ArxivRequestGate
        except ModuleNotFoundError:
            self.fail("shared arXiv request gate is missing")

        starts = []

        def handle(_request: httpx.Request) -> httpx.Response:
            starts.append(time.monotonic())
            if len(starts) == 1:
                return httpx.Response(503, content=b"temporary failure")
            return httpx.Response(200, content=EMPTY_ATOM)

        with tempfile.TemporaryDirectory() as directory:
            transport = HttpxArxivTransport(
                http_transport=httpx.MockTransport(handle),
                request_gate=ArxivRequestGate(
                    Path(directory) / "arxiv-api.lock",
                    minimum_interval_seconds=0.15,
                ),
            )
            try:
                result = ArxivClient(
                    transport=transport,
                    minimum_interval_seconds=0,
                ).fetch(build_arxiv_query("LLM agent memory"))
            finally:
                transport.close()

        self.assertEqual([attempt.status_code for attempt in result.attempts], [503, 200])
        self.assertEqual(len(starts), 2)
        self.assertGreaterEqual(starts[1] - starts[0], 0.14)

    def test_gate_rechecks_deadline_when_sleep_wakes_early(self):
        from backend.app.search.request_gate import ArxivRequestGate

        class EarlyClock:
            def __init__(self):
                self.now = 100.0
                self.sleeps = 0

            def time(self):
                return 1_700_000_000.0 + self.now

            def monotonic(self):
                return self.now

            def sleep(self, seconds):
                self.sleeps += 1
                self.now += seconds - 0.02 if self.sleeps == 1 else seconds

        clock = EarlyClock()
        with tempfile.TemporaryDirectory() as directory:
            gate = ArxivRequestGate(
                Path(directory) / "arxiv-api.lock", minimum_interval_seconds=0.15
            )
            with patch("backend.app.search.request_gate.time", clock):
                with gate.slot():
                    pass
                with gate.slot():
                    entered_at = clock.now

        self.assertGreaterEqual(entered_at, 100.15)
        self.assertEqual(clock.sleeps, 2)


if __name__ == "__main__":
    unittest.main()
