"""Native ToolSelector construction must work with the installed LMI version."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paperqa_smoke import build_models


class AgentModelConfigTest(unittest.TestCase):
    def test_native_selector_constructs(self):
        config = {
            "DEEPSEEK_MODEL": "deepseek-flash",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            "DEEPSEEK_API_KEY": "offline-test-key",
            "EMBEDDING_MODEL": "text-embedding-v4",
            "EMBEDDING_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "EMBEDDING_API_KEY": "offline-test-key",
            "EMBEDDING_DIMENSIONS": 1024, "EMBEDDING_BATCH_SIZE": 10,
        }
        settings, _, _, _ = build_models(config)
        from aviary.core import ToolSelector
        self.assertIsInstance(settings.make_aviary_tool_selector("ToolSelector"), ToolSelector)


class LanguageModelRequestTest(unittest.IsolatedAsyncioTestCase):
    async def test_all_language_roles_send_configured_credentials_and_thinking_mode(self):
        import httpx
        from aviary.core import Message
        from openai import AsyncOpenAI

        config = {
            "DEEPSEEK_MODEL": "deepseek-flash",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            "DEEPSEEK_API_KEY": "offline-deepseek-key",
            "EMBEDDING_MODEL": "text-embedding-v4",
            "EMBEDDING_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "EMBEDDING_API_KEY": "offline-embedding-key",
            "EMBEDDING_DIMENSIONS": 1024, "EMBEDDING_BATCH_SIZE": 10,
        }
        settings, answer, summary, _ = build_models(config)
        for model in (answer, summary, settings.get_agent_llm()):
            spec = model.llm_config.models[0]
            self.assertEqual(spec.api_base, config["DEEPSEEK_BASE_URL"])
            self.assertEqual(spec.api_key.get_secret_value(), config["DEEPSEEK_API_KEY"])
        requests = []

        def handle(request):
            self.assertEqual(str(request.url), "https://api.deepseek.com/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer offline-deepseek-key")
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(body["model"], "deepseek-flash")
            self.assertEqual(body["thinking"], {"type": "disabled"})
            self.assertEqual(body["max_tokens"], 6000 if len(requests) == 1 else 1800)
            message = {"role": "assistant", "content": "offline answer"}
            finish = "stop"
            if body.get("tools"):
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call-offline", "type": "function", "function": {
                        "name": "probe", "arguments": "{}",
                    },
                }]}
                finish = "tool_calls"
            return httpx.Response(200, json={
                "id": "offline-response", "object": "chat.completion", "created": 1,
                "model": "deepseek-flash",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            async with AsyncOpenAI(api_key=config["DEEPSEEK_API_KEY"],
                                   base_url=config["DEEPSEEK_BASE_URL"],
                                   http_client=http, max_retries=0) as client:
                for model in (answer, summary):
                    result = await model.call_single(messages=[Message(content="probe")], client=client)
                    self.assertEqual(result.text, "offline answer")
                router = settings.get_agent_llm().get_router()
                response = await router.acompletion(
                    model=settings.agent.agent_llm, client=client,
                    messages=[{"role": "user", "content": "Use probe."}],
                    tools=[{"type": "function", "function": {
                        "name": "probe", "description": "Offline probe.",
                        "parameters": {"type": "object", "properties": {}},
                    }}], tool_choice="required",
                )
                self.assertEqual(response.choices[0].message.tool_calls[0].function.name, "probe")
        self.assertEqual(len(requests), 3)


if __name__ == "__main__":
    unittest.main()
