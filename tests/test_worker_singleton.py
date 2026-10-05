import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.search_worker import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WorkerSingletonTest(unittest.TestCase):
    def test_second_worker_is_busy_while_first_worker_polls(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "papertrail.sqlite"
            first_worker = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(PROJECT_ROOT / "scripts" / "search_worker.py"),
                    "--database",
                    str(database_path),
                    "--poll-seconds",
                    "0.05",
                    "--live",
                ],
                cwd=PROJECT_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 5
                while not database_path.is_file() and time.monotonic() < deadline:
                    if first_worker.poll() is not None:
                        self.fail(f"first worker exited: {first_worker.stderr.read()}")
                    time.sleep(0.02)
                self.assertTrue(database_path.is_file(), "first worker did not start")

                output = StringIO()
                with redirect_stdout(output):
                    denied = main(
                        ["--database", str(database_path), "--once", "--live"],
                        client=object(),
                    )
                self.assertEqual(denied, 2)
                self.assertIn("[BUSY]", output.getvalue())
                self.assertIsNone(first_worker.poll())
            finally:
                if first_worker.poll() is None:
                    first_worker.kill()
                first_worker.communicate(timeout=5)

    def test_other_process_blocks_worker_and_crash_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "papertrail.sqlite"
            ready_path = root / "holder-ready"
            holder_code = (
                "from pathlib import Path\n"
                "import sys\n"
                "from backend.app.jobs.process_lock import worker_singleton\n"
                "with worker_singleton(Path(sys.argv[1])) as acquired:\n"
                "    if not acquired:\n"
                "        sys.exit(2)\n"
                "    Path(sys.argv[2]).write_text('ready')\n"
                "    sys.stdin.read(1)\n"
            )
            holder = subprocess.Popen(
                [sys.executable, "-c", holder_code, str(database_path), str(ready_path)],
                cwd=PROJECT_ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 5
                while not ready_path.is_file() and time.monotonic() < deadline:
                    if holder.poll() is not None:
                        self.fail(f"lock holder exited: {holder.stderr.read()}")
                    time.sleep(0.02)
                self.assertTrue(ready_path.is_file(), "lock holder did not start")

                output = StringIO()
                with redirect_stdout(output):
                    denied = main(
                        ["--database", str(database_path), "--once", "--live"],
                        client=object(),
                    )
                self.assertEqual(denied, 2)
                self.assertIn("[BUSY]", output.getvalue())
                self.assertFalse(database_path.exists())

                holder.kill()
                holder.communicate(timeout=5)
                output = StringIO()
                with redirect_stdout(output):
                    admitted = main(
                        ["--database", str(database_path), "--once", "--live"],
                        client=object(),
                    )
                self.assertEqual(admitted, 0)
                self.assertIn("[IDLE]", output.getvalue())
            finally:
                if holder.poll() is None:
                    holder.kill()
                holder.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
