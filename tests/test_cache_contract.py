import hashlib
import json
import sys
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paperqa_agent import cached_papers
from paperqa.types import Doc, Text


class CacheContractTest(unittest.TestCase):
    def test_changed_pdf_or_embedding_config_cannot_reuse_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "data/papers"
            folder.mkdir(parents=True)
            pdf = folder / "fixture.pdf"
            pdf.write_bytes(b"Synthetic cache identity fixture, not a parseable PDF")
            config = {"EMBEDDING_MODEL": "test", "EMBEDDING_BASE_URL": "https://example.invalid",
                      "EMBEDDING_DIMENSIONS": 2}
            spec = {"schema": 1, "filename": pdf.name, "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                    "paperqa": version("paper-qa"), "lmi": version("fhlmi"),
                    "embedding_model": "test", "embedding_base_url": config["EMBEDDING_BASE_URL"],
                    "dimensions": 2, "chunk_chars": 3000, "overlap": 200, "multimodal": False}
            doc = Doc(docname="fixture", citation="fixture", dockey="fixture")
            text = Text(name="fixture", text="text", doc=doc, embedding=[1.0, 0.0])
            key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
            cache = root / "data/indexes/smoke" / (key + ".json")
            cache.parent.mkdir(parents=True)
            cache.write_text(json.dumps({"spec": spec, "doc": doc.model_dump(mode="json"),
                                         "texts": [text.model_dump(mode="json")]}), encoding="utf-8")
            self.assertEqual(len(cached_papers(config, root)), 1)
            for change in ({"EMBEDDING_DIMENSIONS": 3}, {"EMBEDDING_MODEL": "other"},
                           {"EMBEDDING_BASE_URL": "https://other.invalid"}):
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "No matching smoke cache"):
                    cached_papers({**config, **change}, root)
            pdf.write_bytes(b"Changed PDF content")
            with self.assertRaisesRegex(ValueError, "No matching smoke cache"):
                cached_papers(config, root)
