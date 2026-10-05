import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.api import create_app
from backend.app.db import open_database


class ApiHealthTest(unittest.TestCase):
    def test_initialized_database_is_healthy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "papertrail.sqlite"
            engine = open_database(database_path)
            engine.dispose()

            with TestClient(create_app(database_path, root / "data")) as client:
                response = client.get("/api/health")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "ok"})

    def test_missing_database_is_unavailable_without_creating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "missing" / "papertrail.sqlite"

            with TestClient(create_app(database_path, root / "data")) as client:
                response = client.get("/api/health")

            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["code"], "DATABASE_UNAVAILABLE")
            self.assertFalse(database_path.exists())
            self.assertFalse(database_path.parent.exists())

    def test_existing_database_without_schema_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "empty.sqlite"
            sqlite3.connect(database_path).close()

            with TestClient(create_app(database_path, root / "data")) as client:
                response = client.get("/api/health")

            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["code"], "DATABASE_UNAVAILABLE")
            self.assertEqual(response.json()["resource_id"], None)


if __name__ == "__main__":
    unittest.main()
