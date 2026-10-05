"""Manual T04 commands preview external work before downloading or embedding."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backend.app.db import open_database
from backend.app.collections.service import SelectedSource, create_collection, set_collection_selection
from backend.app.models import CollectionImportRun
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from tests.test_ingestion_selection import saved_search


class CollectionCliTest(unittest.TestCase):
    def test_previews_do_not_create_a_missing_database(self):
        from collection_cli import main

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "missing.sqlite"
            for command in (
                ["import", "--collection-id", "unknown", "--selection-revision", "1"],
                ["index", "--revision-id", "unknown"],
            ):
                with self.subTest(command=command[0]), redirect_stdout(StringIO()):
                    code = main(["--database", str(database), *command])
                    self.assertEqual(code, 1)
                    self.assertFalse(database.exists())

    def test_import_preview_creates_no_report_or_pdf_request(self):
        from collection_cli import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            run_dir = saved_search(data_root)
            database = root / "papertrail.sqlite"
            engine = open_database(database)
            try:
                collection = create_collection(engine)
                set_collection_selection(
                    engine, collection.id, expected_revision=0, data_root=data_root,
                    selections=(SelectedSource(run_dir, "2401.00001v1"),),
                )
                output = StringIO()
                with redirect_stdout(output):
                    code = main([
                        "--database", str(database), "--data-root", str(data_root),
                        "import", "--collection-id", collection.id,
                        "--selection-revision", "1",
                    ])
                self.assertEqual(code, 0)
                self.assertIn("PREVIEW", output.getvalue())
                with Session(engine) as session:
                    self.assertEqual(
                        session.scalar(select(func.count()).select_from(CollectionImportRun)), 0
                    )
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
