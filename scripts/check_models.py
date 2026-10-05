"""PaperTrail API smoke checks. No network access unless --live is supplied.

Run from the project: uv run python scripts/check_models.py [--live]
This checks provider APIs, not PaperQA's adapter or retrieval quality.
"""

import argparse
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]


class CheckError(Exception):
    """A diagnostic safe to show without exposing credentials or response bodies."""


def load_config(only):
    # Explicit project .env takes precedence over shell variables.
    config = {**os.environ, **dotenv_values(ROOT / ".env")}
    prefixes = ["DEEPSEEK", "EMBEDDING"] if only == "all" else [only.upper()]
    for prefix in prefixes:
        for suffix in ("API_KEY", "BASE_URL", "MODEL"):
            key = f"{prefix}_{suffix}"
            config[key] = (config.get(key) or "").strip()
            if not config[key]:
                raise CheckError(f"Missing {key}; fill in the project .env file.")
        url = urlsplit(config[f"{prefix}_BASE_URL"])
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise CheckError(f"{prefix}_BASE_URL must be an HTTPS base URL without credentials or query parameters.")
    if "DEEPSEEK" in prefixes and config.get("DEEPSEEK_THINKING") != "disabled":
        raise CheckError("This probe requires DEEPSEEK_THINKING=disabled.")
    if "EMBEDDING" in prefixes:
        if config.get("EMBEDDING_BACKEND") != "api":
            raise CheckError("This probe requires EMBEDDING_BACKEND=api.")
        try:
            config["EMBEDDING_DIMENSIONS"] = int(config.get("EMBEDDING_DIMENSIONS", "1024"))
            batch_size = int(config.get("EMBEDDING_BATCH_SIZE", "10"))
        except (ValueError, TypeError):
            raise CheckError("Embedding dimensions and batch size must be integers.") from None
        if config["EMBEDDING_DIMENSIONS"] <= 0 or not 1 <= batch_size <= 10:
            raise CheckError("Embedding dimensions must be positive; batch size must be 1..10.")
        config["EMBEDDING_BATCH_SIZE"] = batch_size
    return config


def post_json(base_url, api_key, route, payload):
    """One request, a 45-second socket timeout, and no automatic retries."""
    request = Request(
        base_url.rstrip("/") + route,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=45) as response:
            return json.load(response)
    except HTTPError as exc:
        hints = {
            400: "Check model ID, request parameters and input limits.",
            401: "Check the API key and its region.",
            402: "Check account balance and service activation.",
            403: "Check account permissions, model access and region.",
            404: "Check the base URL and model ID.",
            429: "Check quota or rate limits; retry later manually.",
        }
        raise CheckError(f"HTTP {exc.code}: {hints.get(exc.code, 'Check provider service status.')}" ) from None
    except (URLError, TimeoutError, OSError):
        raise CheckError("Connection failed or timed out; check network, proxy and endpoint.") from None
    except (ValueError, UnicodeError):
        raise CheckError("The endpoint did not return valid JSON.") from None


def usage_of(response):
    usage = response.get("usage") or {}
    return {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if type(usage.get(key)) is int}


def chat(config, messages, **kwargs):
    return post_json(config["DEEPSEEK_BASE_URL"], config["DEEPSEEK_API_KEY"], "/chat/completions", {
        "model": config["DEEPSEEK_MODEL"], "messages": messages,
        "thinking": {"type": "disabled"}, "max_tokens": 256, **kwargs,
    })


def check_chat(config):
    response = chat(config, [{"role": "user", "content": "Reply with a short greeting."}])
    content = response["choices"][0]["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise CheckError("Chat returned no non-empty text.")
    return {"usage": usage_of(response)}


def check_tools(config):
    # Only this harmless local tool can run. No eval, shell or file access.
    tool = {"type": "function", "function": {
        "name": "get_probe_code", "description": "Return the current random probe code.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }}
    messages = [{"role": "user", "content":
                 "Call get_probe_code, then reply with only the code returned by the tool."}]
    first = chat(config, messages, tools=[tool], tool_choice={
        "type": "function", "function": {"name": "get_probe_code"},
    })
    message = first["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != "get_probe_code":
        raise CheckError("Expected exactly one get_probe_code tool call.")
    call = calls[0]
    if call.get("type") != "function" or not call.get("id"):
        raise CheckError("Tool call is missing its type or ID.")
    if json.loads(call["function"]["arguments"]) != {}:
        raise CheckError("Tool arguments must be an empty JSON object.")
    code = uuid.uuid4().hex  # Generated after the request: the model cannot guess it.
    messages.extend([
        {"role": "assistant", "content": message.get("content"), "tool_calls": calls},
        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"code": code})},
    ])
    second = chat(config, messages, tools=[tool], tool_choice="none")
    final_message = second["choices"][0]["message"]
    if final_message.get("tool_calls") or (final_message.get("content") or "").strip() != code:
        raise CheckError("Tool result was not returned exactly; inspect tool-following compatibility.")
    return {"usage": [usage_of(first), usage_of(second)]}


def check_embeddings(config):
    texts = ["How do agents remember previous conversations?",
             "An agent stores and retrieves memories from previous conversations.",
             "Tomatoes grow best with sunlight and regular watering."]
    dimensions = config["EMBEDDING_DIMENSIONS"]
    vectors, usages = [], []
    batch_size = config["EMBEDDING_BATCH_SIZE"]
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = post_json(config["EMBEDDING_BASE_URL"], config["EMBEDDING_API_KEY"], "/embeddings", {
            "model": config["EMBEDDING_MODEL"], "input": batch,
            "dimensions": dimensions, "encoding_format": "float",
        })
        data = sorted(response["data"], key=lambda item: item["index"])
        if [item["index"] for item in data] != list(range(len(batch))):
            raise CheckError("Embedding response has missing, duplicate or invalid indices.")
        for item in data:
            vector = item["embedding"]
            if not isinstance(vector, list) or len(vector) != dimensions:
                raise CheckError("Embedding vector dimension differs from configuration.")
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in vector):
                raise CheckError("Embedding contains non-numeric or non-finite values.")
            norm = math.sqrt(sum(x * x for x in vector))
            if not math.isfinite(norm) or norm == 0:
                raise CheckError("Embedding has a zero or invalid norm.")
            vectors.append([x / norm for x in vector])
        usages.append(usage_of(response))
    scores = [sum(a * b for a, b in zip(vectors[0], v)) for v in vectors[1:]]
    return {"vectors": len(vectors), "dimensions": dimensions, "usage": usages,
            "related_cosine": round(scores[0], 4), "unrelated_cosine": round(scores[1], 4),
            "note": "Similarity scores are diagnostics, not a retrieval-quality evaluation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Send billable API requests.")
    parser.add_argument("--only", choices=("all", "deepseek", "embedding"), default="all")
    args = parser.parse_args()
    try:
        config = load_config(args.only)
    except (CheckError, ValueError) as exc:
        print(f"[FAIL] Configuration: {exc}")
        return 1
    if not args.live:
        print("[PASS] Local configuration only. No API request sent; keys are not authenticated.")
        print("Add --live to run API checks. Optional: --only deepseek or --only embedding.")
        return 0

    checks = []
    if args.only in ("all", "deepseek"):
        checks.extend([("chat", check_chat), ("tools", check_tools)])
    if args.only in ("all", "embedding"):
        checks.append(("embedding", check_embeddings))
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "scope": "provider_api_only",
              "models": {p.lower(): config[f"{p}_MODEL"] for p in ("DEEPSEEK", "EMBEDDING")
                         if args.only == "all" or args.only == p.lower()}, "checks": []}
    for name, check in checks:
        started = time.monotonic()
        try:
            result = {"name": name, "status": "pass", **check(config)}
        except CheckError as exc:
            result = {"name": name, "status": "fail", "error": str(exc)}
        except Exception:
            # Never dump arbitrary HTTP bodies, exception text or credentials.
            result = {"name": name, "status": "fail", "error": "Unexpected response schema or local error."}
        result["seconds"] = round(time.monotonic() - started, 2)
        report["checks"].append(result)
        print(f"[{result['status'].upper()}] {name}: {result.get('error', 'OK')}", flush=True)
        if result["status"] == "fail":
            break  # Avoid further paid requests after a failure.
    report["not_run"] = [name for name, _ in checks[len(report["checks"]):]]
    directory = ROOT / "data" / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"model-check-{uuid.uuid4().hex[:12]}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {path}")
    return int(any(item["status"] == "fail" for item in report["checks"]))


if __name__ == "__main__":
    raise SystemExit(main())
