"""Shared model settings for document QA and native PaperQA agents.

Run this file without flags for an offline construction check. No model API calls.
Do not serialize these settings to logs: they contain API credentials.
"""

import os
from copy import deepcopy

try:
    from scripts.check_models import ROOT, load_config
except ModuleNotFoundError:  # Direct execution: python scripts/model_config.py
    from check_models import ROOT, load_config

os.environ["PQA_HOME"] = str(ROOT / "data" / "runtime")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
from lmi import LiteLLMEmbeddingModel, LiteLLMModel
from paperqa import Settings
from pydantic import PrivateAttr

from backend.app.qa.context import original_context_serializer
from backend.app.qa.prompts import ANSWER_PROMPT


class AgentModel(LiteLLMModel):
    """Bridge PaperQA's legacy ToolSelector router API to installed LMI 1.0.x."""

    _agent_router: object = PrivateAttr(default=None)

    def get_router(self):
        if self._agent_router is None:
            self._agent_router = litellm.Router(
                model_list=self.config["model_list"],
                **self.config.get("router_kwargs", {}),
            )
        return self._agent_router


class PaperTrailSettings(Settings):
    """Preserve configuration when PaperQA creates its own model instances."""

    def get_embedding_model(self):
        config = deepcopy(self.embedding_config or {})
        dimensions = config.pop("dimensions")
        return LiteLLMEmbeddingModel(name=self.embedding, ndim=dimensions, config=config)

    def get_agent_llm(self):
        return AgentModel(name=self.agent.agent_llm, config=deepcopy(self.agent.agent_llm_config))


def build_settings(config):
    llm = "openai/" + config["DEEPSEEK_MODEL"]
    llm_config = {
        "model_list": [{"model_name": llm, "litellm_params": {
            "model": llm, "api_base": config["DEEPSEEK_BASE_URL"],
            "api_key": config["DEEPSEEK_API_KEY"], "max_tokens": 1800,
            "timeout": 90, "max_retries": 0,
            "extra_body": {"thinking": {"type": "disabled"}},
        }}],
        "router_kwargs": {"num_retries": 0},
    }
    answer_config = deepcopy(llm_config)
    answer_config["model_list"][0]["litellm_params"]["max_tokens"] = 6000
    return PaperTrailSettings(
        llm=llm, llm_config=answer_config,
        summary_llm=llm, summary_llm_config=deepcopy(llm_config),
        agent={"agent_llm": llm, "agent_llm_config": deepcopy(llm_config)},
        embedding="openai/" + config["EMBEDDING_MODEL"],
        embedding_config={
            "dimensions": config["EMBEDDING_DIMENSIONS"],
            "batch_size": config["EMBEDDING_BATCH_SIZE"],
            "kwargs": {
                "api_base": config["EMBEDDING_BASE_URL"],
                "api_key": config["EMBEDDING_API_KEY"],
                "timeout": 90, "num_retries": 0, "encoding_format": "float",
                "allowed_openai_params": ["dimensions"],
            },
        },
        parsing={"use_doc_details": False, "multimodal": False,
                 "reader_config": {"chunk_chars": 3000, "overlap": 200}},
        # 0 is our explicit no-piece-cap sentinel, consumed by the custom serializer.
        answer={"evidence_k": 10, "answer_max_sources": 0,
                "max_concurrent_requests": 1, "answer_length": "as much detail as the question and evidence require"},
        prompts={"qa": ANSWER_PROMPT},
        custom_context_serializer=original_context_serializer,
        verbosity=0,
    )


def build_models(config):
    settings = build_settings(config)
    return settings, settings.get_llm(), settings.get_summary_llm(), settings.get_embedding_model()


if __name__ == "__main__":
    settings, answer, summary, embedding = build_models(load_config("all"))
    selector = settings.make_aviary_tool_selector("ToolSelector")
    print(f"[READY] Answer: {answer.name}")
    print(f"[READY] Summary: {summary.name}")
    print(f"[READY] Agent: {settings.agent.agent_llm} ({type(selector).__name__})")
    print(f"[READY] Embedding: {embedding.name}; dimensions={embedding.ndim}; batch_size={embedding.config['batch_size']}")
    print("[LOCAL ONLY] Four model roles constructed. No model API calls made.")
