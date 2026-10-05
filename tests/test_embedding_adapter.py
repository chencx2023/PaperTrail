"""Exercise the real LMI/LiteLLM/OpenAI adapter with an offline HTTP transport."""

import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paperqa_smoke import build_models


class EmbeddingAdapterTest(unittest.IsolatedAsyncioTestCase):
    async def test_dimensions_and_batching_reach_dashscope(self):
        import httpx
        from openai import AsyncOpenAI

        for dimensions in (1024, 256):
            with self.subTest(dimensions=dimensions):
                config = {
                    "DEEPSEEK_MODEL": "deepseek-flash",
                    "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
                    "DEEPSEEK_API_KEY": "offline-test-key",
                    "EMBEDDING_MODEL": "text-embedding-v4",
                    "EMBEDDING_BASE_URL": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                    "EMBEDDING_API_KEY": "offline-test-key",
                    "EMBEDDING_DIMENSIONS": dimensions,
                    "EMBEDDING_BATCH_SIZE": 10,
                }
                settings, _, _, _ = build_models(config)
                # Exercise the factory used by native indexing/tools, not an
                # embedding object manually passed by the smoke script.
                with patch("litellm.get_model_cost_map", side_effect=AssertionError("Unexpected dimension inference")):
                    embedding = settings.get_embedding_model()
                self.assertEqual(embedding.config["kwargs"]["api_base"], config["EMBEDDING_BASE_URL"])
                self.assertEqual(embedding.config["kwargs"]["api_key"], config["EMBEDDING_API_KEY"])
                requests = []

                def handle(request):
                    self.assertEqual(str(request.url), config["EMBEDDING_BASE_URL"] + "/embeddings")
                    body = json.loads(request.content)
                    requests.append(body)
                    self.assertEqual(body["model"], "text-embedding-v4")
                    self.assertEqual(body["dimensions"], dimensions)
                    self.assertEqual(body["encoding_format"], "float")
                    self.assertLessEqual(len(body["input"]), 10)
                    return httpx.Response(200, json={
                        "object": "list", "model": "text-embedding-v4",
                        "data": [{"object": "embedding", "index": i,
                                  "embedding": [0.1] * dimensions}
                                 for i in range(len(body["input"]))],
                        "usage": {"prompt_tokens": len(body["input"]),
                                  "total_tokens": len(body["input"])},
                    })

                async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
                    async with AsyncOpenAI(
                        api_key="offline-test-key", base_url=config["EMBEDDING_BASE_URL"],
                        http_client=http, max_retries=0,
                    ) as client:
                        embedding.config["kwargs"]["client"] = client
                        vectors = await embedding.embed_documents(["test paper"] * 11)
                self.assertEqual([len(r["input"]) for r in requests], [10, 1])
                self.assertEqual(len(vectors), 11)
                self.assertTrue(all(len(v) == dimensions for v in vectors))


if __name__ == "__main__":
    unittest.main()
