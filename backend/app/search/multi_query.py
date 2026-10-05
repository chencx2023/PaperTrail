"""Saved multi-query sources and replayable fusion; no synthetic arXiv records."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from backend.app.search.arxiv import ARXIV_PARSER_VERSION, ParsedArxivFeed
from backend.app.search.query_builder import MultiQueryPlan, plan_from_data, query_plans
from backend.app.search.ranking import RRF_K, fuse_rankings
from backend.app.search.service import (
    InvalidSearchSnapshot, PersistedSearchRun, ReplayedSearchRun,
    _candidate_data, _canonical_json, _read_json_object, _read_json_value,
    _verified_snapshot_bytes, _write_json, publish_search_cache, replay_search_run, run_search,
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feed(feeds, run_id):
    fused = fuse_rankings(feeds)
    valid = [f for f in feeds if f is not None]
    feed = ParsedArxivFeed(
        feed_title="Saved multi-query pool", feed_id=f"papertrail:{run_id}",
        feed_updated_at=max((f.feed_updated_at for f in valid), default=""), self_url="",
        total_results=len(fused.candidates), start_index=0, items_per_page=len(fused.candidates),
        records=fused.records, candidates=fused.candidates,
    )
    return feed, fused


def _fusion(manifest, fused, generation):
    queries = manifest["queries"]
    return {
        "method": "rrf", "k": RRF_K, "plan": manifest["plan"],
        "queries": queries, "successful_queries": sum(q["status"] == "success" for q in queries),
        "partial": any(q["status"] != "success" for q in queries),
        "candidate_count": len(fused.candidates), "ranking": list(fused.ranking),
        "merge_order": list(fused.merge_order),
        "http_attempts": sum(q["http_attempts"] for q in queries),
        "cache_hits": sum(q["cache_hit"] for q in queries),
        "elapsed_ms": manifest["elapsed_ms"], "relaxation_requests": 0,
        "generation": generation,
    }


def run_multi_search(plan: MultiQueryPlan, *, client, output_root: Path,
                     run_id: str, generation: dict | None = None) -> PersistedSearchRun:
    plan = plan_from_data(plan.as_data())
    if not re.fullmatch(r"[A-Za-z0-9-]{1,80}", run_id):
        raise ValueError("Invalid multi-query run ID.")
    root = Path(output_root)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    _write_json(run_dir / "request.json", {"schema_version": 1, "mode": "multi_query",
                "plan": plan.as_data(), "generation": generation})
    started = time.perf_counter()
    feeds, children, fetched = [], [], []
    for index, query in enumerate(query_plans(plan), 1):
        query_started = time.perf_counter()
        child = run_search(query.query_text, client=client, output_root=root,
                           run_id=f"{run_id}-q{index}", query_plan=query,
                           cache_requires_commit_marker=True)
        if child.feed is not None and len(child.feed.records) > 20:
            raise InvalidSearchSnapshot("An arXiv response exceeded the per-query 20-record limit.")
        metadata = _read_json_object(child.run_dir / "run.json", "query run")
        feeds.append(child.feed)
        if child.status == "success" and child.fetched_at:
            fetched.append(child.fetched_at)
        children.append({
            "index": index, "label": "原查询" if index == 1 else plan.expansions[index - 2].label,
            "query_text": query.query_text, "search_query": query.search_query,
            "run_id": child.run_id, "status": child.status,
            "run_sha256": _digest(child.run_dir / "run.json"),
            "request_sha256": _digest(child.run_dir / "request.json"),
            "candidate_count": len(child.feed.candidates) if child.feed else None,
            "cache_hit": child.cache_hit, "http_attempts": len(metadata["attempts"]),
            "elapsed_ms": round((time.perf_counter() - query_started) * 1000),
            "fetched_at": child.fetched_at, "error_kind": child.error_kind,
            "original_order": [{"arxiv_id": c.arxiv_id, "versioned_id": c.versioned_id, "rank": rank}
                               for rank, c in enumerate(child.feed.candidates, 1)] if child.feed else [],
        })
    feed, fused = _feed(feeds, run_id)
    manifest = {"schema_version": 1, "plan": plan.as_data(), "queries": children,
                "rrf_k": RRF_K, "relaxation_requests": 0,
                "elapsed_ms": round((time.perf_counter() - started) * 1000)}
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "candidates.json", _candidate_data(feed))
    _write_json(run_dir / "ranking.json", {"ranking": list(fused.ranking), "merge_order": list(fused.merge_order)})
    succeeded = any(child["status"] == "success" for child in children)
    status = "success" if succeeded else "failed"
    error = None if succeeded else {"kind": "all_queries_failed", "message": "所有查询均未完成，请检查失败记录后显式重试。"}
    fetched_at = max(fetched, default=None)
    _write_json(run_dir / "run.json", {
        "schema_version": 2, "parser_version": ARXIV_PARSER_VERSION, "run_id": run_id,
        "status": status, "created_at": created_at, "fetched_at": fetched_at,
        "candidate_count": len(feed.candidates), "error": error,
        "snapshot": {"path": "manifest.json", "sha256": _digest(run_dir / "manifest.json"),
                     "byte_count": (run_dir / "manifest.json").stat().st_size, "source": "multi_query"},
    })
    return PersistedSearchRun(run_id, run_dir, status, created_at, fetched_at,
        feed if succeeded else None, error["kind"] if error else None, error["message"] if error else None,
        all(q["cache_hit"] for q in children), "multi_query")


def replay_multi_search(run_dir: Path) -> ReplayedSearchRun:
    """Verify each observed Atom source and rebuild ranks, without I/O requests."""
    try:
        root = run_dir.resolve()
        request = _read_json_object(root / "request.json", "multi-query request")
        plan = plan_from_data(request["plan"])
        metadata = _read_json_object(root / "run.json", "multi-query run")
        if metadata["status"] != "success" or metadata["run_id"] != root.name:
            raise InvalidSearchSnapshot("Only the owning successful multi-query run may be replayed.")
        body, digest = _verified_snapshot_bytes(root, metadata)
        manifest = json.loads(body)
        if (manifest["plan"] != plan.as_data() or manifest["rrf_k"] != RRF_K
                or manifest["relaxation_requests"] != 0
                or len(manifest["queries"]) != len(query_plans(plan))):
            raise InvalidSearchSnapshot("The saved multi-query plan or source count changed.")
        feeds = []
        for index, (entry, query) in enumerate(zip(manifest["queries"], query_plans(plan), strict=True), 1):
            expected_id = f"{root.name}-q{index}"
            child = (root.parent / expected_id).resolve()
            if (entry["index"] != index or entry["run_id"] != expected_id
                    or not child.is_relative_to(root.parent)
                    or _digest(child / "run.json") != entry["run_sha256"]
                    or _digest(child / "request.json") != entry["request_sha256"]):
                raise InvalidSearchSnapshot("A multi-query source identity or hash changed.")
            child_request = _read_json_object(child / "request.json", "query request")
            child_run = _read_json_object(child / "run.json", "query run")
            if (child_request.get("mode") is not None or child_request["search_query"] != query.search_query
                    or child_request["params"]["max_results"] != "20"
                    or entry["search_query"] != query.search_query
                    or child_run["run_id"] != expected_id or child_run["status"] != entry["status"]):
                raise InvalidSearchSnapshot("A saved source no longer matches its planned query.")
            if (entry["label"] != ("原查询" if index == 1 else plan.expansions[index - 2].label)
                    or entry["query_text"] != query.query_text
                    or entry["fetched_at"] != child_run["fetched_at"]
                    or entry["http_attempts"] != len(child_run["attempts"])
                    or entry["cache_hit"] != child_run["cache"]["hit"]
                    or type(entry["elapsed_ms"]) is not int or entry["elapsed_ms"] < 0):
                raise InvalidSearchSnapshot("Saved source provenance differs from its verified run.")
            if entry["status"] == "success":
                replay = replay_search_run(child)
                if not replay.candidates_match or len(replay.feed.records) > 20:
                    raise InvalidSearchSnapshot("Saved source candidates differ from their Atom snapshot.")
                original = [{"arxiv_id": c.arxiv_id, "versioned_id": c.versioned_id, "rank": rank}
                            for rank, c in enumerate(replay.feed.candidates, 1)]
                if entry["original_order"] != original or entry["candidate_count"] != len(replay.feed.candidates):
                    raise InvalidSearchSnapshot("Saved native ranks differ from their Atom source.")
                feeds.append(replay.feed)
            elif entry["status"] == "failed":
                if child_run.get("snapshot") is not None:
                    _verified_snapshot_bytes(child, child_run)
                if entry["original_order"] or entry["candidate_count"] is not None:
                    raise InvalidSearchSnapshot("A failed query cannot claim native candidates.")
                feeds.append(None)
            else:
                raise InvalidSearchSnapshot("Unknown saved query status.")
        feed, fused = _feed(feeds, root.name)
        if not any(f is not None for f in feeds) or metadata["candidate_count"] != len(feed.candidates):
            raise InvalidSearchSnapshot("The fusion success or candidate count differs from its sources.")
        saved_ranking = _read_json_value(root / "ranking.json", "fusion ranking")
        expected_ranking = {"ranking": list(fused.ranking), "merge_order": list(fused.merge_order)}
        if _canonical_json(saved_ranking) != _canonical_json(expected_ranking):
            raise InvalidSearchSnapshot("The saved fusion ranks differ from their sources.")
        matches = _canonical_json(_read_json_value(root / "candidates.json", "fused candidates")) == _canonical_json(_candidate_data(feed))
        return ReplayedSearchRun(root.name, root, feed, matches, metadata["parser_version"],
                                 ARXIV_PARSER_VERSION, digest, _fusion(manifest, fused, request.get("generation")))
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, InvalidSearchSnapshot):
            raise
        raise InvalidSearchSnapshot("The saved multi-query result is incomplete or invalid.") from error


def publish_multi_cache(run_dir: Path):
    replay = replay_multi_search(run_dir)
    if not replay.candidates_match:
        raise InvalidSearchSnapshot("The saved fusion candidates changed.")
    for query in replay.fusion["queries"]:
        if query["status"] == "success":
            publish_search_cache(run_dir.parent / query["run_id"])
