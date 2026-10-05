"""Runtime probes never create a worker or claim queued work."""

import socket
import subprocess
import sys
import time
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.api import create_app
from backend.app.db import open_database
from backend.app.jobs.process_lock import worker_singleton, worker_status
from backend.app.jobs.service import submit_search
from scripts.run_papertrail import child_commands, preflight


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = self.root / "runtime.sqlite"
        self.engine = open_database(self.database)

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def test_missing_and_idle_lock_are_stopped_without_creating_files(self):
        lock = self.database.with_name(self.database.name + ".worker.lock")
        self.assertEqual(worker_status(self.database), "stopped")
        self.assertFalse(lock.exists())
        with worker_singleton(self.database):
            self.assertEqual(worker_status(self.database), "running")
        self.assertTrue(lock.exists())
        self.assertEqual(worker_status(self.database), "stopped")

    def test_status_reports_queued_work_without_claiming_it(self):
        queued = submit_search(self.engine, question="记忆评估", query_text="agent memory", idempotency_key="runtime")
        with TestClient(create_app(self.database, self.root)) as client:
            result = client.get("/api/runtime")
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()["worker_status"], "stopped")
            self.assertEqual(result.json()["queued_count"], 1)
            job = client.get(f"/api/jobs/{queued.job_id}").json()
            self.assertEqual(job["status"], "queued")
            self.assertIsNone(job["attempt"])

    def test_launcher_passes_same_absolute_paths_and_requires_live(self):
        commands = child_commands(self.database, self.root, 9876, offline=False)
        for command in commands:
            self.assertEqual(command[command.index("--database") + 1], str(self.database.resolve()))
            self.assertEqual(command[command.index("--data-root") + 1], str(self.root.resolve()))
        self.assertIn("--live", commands[1])

    def test_preflight_rejects_busy_port_and_existing_worker(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaisesRegex(ValueError, "port"):
                preflight(self.database, listener.getsockname()[1])
        with worker_singleton(self.database):
            with self.assertRaisesRegex(ValueError, "worker"):
                preflight(self.database, 0)

    def test_supervisor_stop_file_releases_port_and_worker_lock(self):
        project = Path(__file__).resolve().parents[1]
        scratch = project / ".tooling"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch, prefix="runtime-test-") as temp:
            root = Path(temp)
            database = root / "db.sqlite"
            engine = open_database(database)
            engine.dispose()
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            signal = root / "stop"
            with (root / "launcher.log").open("w", encoding="utf-8") as output:
                process = subprocess.Popen([sys.executable, str(project / "scripts/run_papertrail.py"), "--database", str(database), "--data-root", str(root), "--port", str(port), "--offline", "--stop-file", str(signal)], cwd=project, stdout=output, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 30
                    while worker_status(database) != "running":
                        self.assertIsNone(process.poll(), (root / "launcher.log").read_text(encoding="utf-8"))
                        if time.monotonic() > deadline:
                            self.fail("Fixture worker did not become ready")
                        time.sleep(.1)
                    signal.touch()
                    self.assertEqual(process.wait(timeout=15), 0)
                    self.assertEqual(worker_status(database), "stopped")
                    with socket.socket() as listener:
                        listener.bind(("127.0.0.1", port))
                finally:
                    if process.poll() is None:
                        signal.touch()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()


if __name__ == "__main__":
    unittest.main()
