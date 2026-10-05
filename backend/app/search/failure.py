"""Safe explanations of saved failed searches, without replaying HTTP requests."""

import hashlib
import json
from pathlib import Path

from backend.app.search.query_builder import plan_from_data, query_plans
from backend.app.search.service import (
    InvalidSearchSnapshot, _read_json_object, _verified_snapshot_bytes, build_arxiv_query,
)


def _failed_source(root: Path, expected_id: str, expected_query: str) -> dict:
    request = _read_json_object(root / "request.json", "failed request")
    metadata = _read_json_object(root / "run.json", "failed run")
    if (request.get("mode") is not None or request["search_query"] != expected_query
            or metadata["run_id"] != expected_id or metadata["status"] != "failed"
            or not isinstance(metadata.get("error"), dict)
            or not isinstance(metadata.get("attempts"), list)
            or not all(isinstance(item, dict) for item in metadata["attempts"])):
        raise InvalidSearchSnapshot("Failed source does not match its saved search.")
    if metadata.get("snapshot") is not None:
        _verified_snapshot_bytes(root, metadata)
    return metadata


def read_search_failure(run_dir: Path, *, expected_run_id: str,
                        query_text: str, query_plan: dict | None) -> dict | None:
    """Return only typed causes and counts from matching, intact local records."""
    try:
        root = run_dir.resolve()
        metadata = _read_json_object(root / "run.json", "failed run")
        request = _read_json_object(root / "request.json", "failed request")
        if (root.name != expected_run_id or metadata["run_id"] != expected_run_id
                or metadata["status"] != "failed"):
            raise InvalidSearchSnapshot("Failed run identity changed.")
        if query_plan is None:
            sources = [_failed_source(root, expected_run_id,
                                      build_arxiv_query(query_text).search_query)]
        else:
            plan = plan_from_data(query_plan)
            if request.get("mode") != "multi_query" or request["plan"] != plan.as_data():
                raise InvalidSearchSnapshot("Failed run plan changed.")
            body, _digest = _verified_snapshot_bytes(root, metadata)
            manifest = json.loads(body)
            expected_queries = query_plans(plan)
            if manifest["plan"] != plan.as_data() or len(manifest["queries"]) != len(expected_queries):
                raise InvalidSearchSnapshot("Failed source count or plan changed.")
            sources = []
            for index, (entry, query) in enumerate(zip(manifest["queries"], expected_queries, strict=True), 1):
                child_id = f"{expected_run_id}-q{index}"
                child = (root.parent / child_id).resolve()
                if (entry["index"] != index or entry["run_id"] != child_id
                        or entry["status"] != "failed" or not child.is_relative_to(root.parent)
                        or hashlib.sha256((child / "run.json").read_bytes()).hexdigest() != entry["run_sha256"]
                        or hashlib.sha256((child / "request.json").read_bytes()).hexdigest() != entry["request_sha256"]):
                    raise InvalidSearchSnapshot("Failed source identity or hash changed.")
                source = _failed_source(child, child_id, query.search_query)
                if (entry["http_attempts"] != len(source["attempts"])
                        or entry["error_kind"] != source["error"]["kind"]):
                    raise InvalidSearchSnapshot("Failed source provenance changed.")
                sources.append(source)
        attempts = [attempt for source in sources for attempt in source["attempts"]]
        kinds = {source["error"]["kind"] for source in sources}
        if any(attempt.get("status_code") == 429 for attempt in attempts):
            kind = "rate_limited"
            message = "arXiv 暂时限制请求（HTTP 429），请稍后重试。频繁提交不会加快恢复。"
        elif "cooldown" in kinds:
            kind = "cooldown"
            message = "arXiv 请求仍处于冷却期，本次未继续发送请求，请稍后重试。"
        elif "transport_error" in kinds:
            kind = "network"
            message = "连接 arXiv 失败或请求超时，请稍后重试。"
        else:
            kind = "upstream"
            message = "arXiv 未返回可用的搜索结果，请稍后重试。"
        return {"kind": kind, "message": message, "http_attempts": len(attempts)}
    except (OSError, ValueError, KeyError, TypeError):
        # Keep the failed Job readable when its artifact is missing or inconsistent.
        return None
