import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.db import open_database
from scripts.api_server import main


class ApiServerTest(unittest.TestCase):
    def test_missing_database_does_not_start_or_create_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "missing" / "papertrail.sqlite"
            starts = []

            def runner(_app, *, host, port):
                starts.append((host, port))

            exit_code = main(
                ["--database", str(database_path), "--data-root", str(root / "data")],
                runner=runner,
            )

            self.assertEqual(exit_code, 2)
            self.assertEqual(starts, [])
            self.assertFalse(database_path.exists())
            self.assertFalse(database_path.parent.exists())

    def test_existing_database_starts_a_healthy_local_only_app(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "papertrail.sqlite"
            engine = open_database(database_path)
            engine.dispose()
            starts = []

            def runner(app, *, host, port):
                with TestClient(app) as client:
                    starts.append((host, port, client.get("/api/health").status_code))

            exit_code = main(
                ["--database", str(database_path), "--port", "8765"], runner=runner
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(starts, [("127.0.0.1", 8765, 200)])

    def test_explicit_init_creates_schema_before_local_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "new.sqlite"
            statuses = []

            def runner(app, *, host, port):
                with TestClient(app) as client:
                    statuses.append(client.get("/api/health").status_code)

            exit_code = main(
                ["--database", str(database_path), "--init-db"], runner=runner
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(database_path.is_file())
            self.assertEqual(statuses, [200])


if __name__ == "__main__":
    unittest.main()
