"""Application-level query construction and persisted search runs."""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable
from uuid import uuid4

from backend.app.search.arxiv import (
    ARXIV_PARSER_VERSION,
    ArxivAPIError,
    InvalidArxivFeed,
    ParsedArxivFeed,
    parse_atom_feed,
)
from backend.app.search.client import ArxivClient, ArxivSearchError, build_arxiv_request


MAX_QUERY_TERMS = 12
QUERY_CONTRACT_VERSION = 1
SEARCH_CACHE_VERSION = 1
DEFAULT_CACHE_TTL = timedelta(hours=24)
_QUERY_TERM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+\-]*\Z")
_RESERVED_OPERATORS = {"and", "or", "andnot"}


@dataclass(frozen=True, slots=True)
class SearchQueryPlan:
    query_text: str
    normalized_query_text: str
    query_terms: tuple[str, ...]
    search_query: str
    contract_version: int = QUERY_CONTRACT_VERSION


class InvalidSearchQuery(ValueError):
    """Raised when user input cannot be represented by the T02 query contract."""


class InvalidSearchSnapshot(ValueError):
    """Raised when a saved search run cannot be replayed safely."""


@dataclass(frozen=True, slots=True)
class PersistedSearchRun:
    run_id: str
    run_dir: Path
    status: str
    created_at: str
    fetched_at: str | None
    feed: ParsedArxivFeed | None
    error_kind: str | None
    error_message: str | None
    cache_hit: bool
    snapshot_source: str | None


@dataclass(frozen=True, slots=True)
class ReplayedSearchRun:
    source_run_id: str
    source_run_dir: Path
    feed: ParsedArxivFeed
    candidates_match: bool
    saved_parser_version: int | None
    current_parser_version: int
    snapshot_sha256: str
    fusion: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    source_run_id: str
    body: bytes
    fetched_at: str
    fetched_time: datetime
    feed: ParsedArxivFeed
    origin_run_id: str
    origin_source: str
    age_seconds: int


def build_arxiv_query(query_text: str) -> SearchQueryPlan:
    terms = tuple(query_text.split())
    if not terms:
        raise InvalidSearchQuery("Enter at least one English keyword.")
    if len(terms) > MAX_QUERY_TERMS:
        raise InvalidSearchQuery(
            f"Use at most {MAX_QUERY_TERMS} space-separated keywords."
        )

    for term in terms:
        if term.casefold() in _RESERVED_OPERATORS or not _QUERY_TERM.fullmatch(term):
            raise InvalidSearchQuery(
                "Use simple English keywords without Boolean operators, quotes, "
                "parentheses, field prefixes, or non-ASCII characters."
            )

    search_query = " AND ".join(
        f"(ti:{term} OR abs:{term})" for term in terms
    )
    return SearchQueryPlan(
        query_text=query_text,
        normalized_query_text=" ".join(terms),
        query_terms=terms,
        search_query=search_query,
    )


def run_search(
    query_text: str,
    *,
    client: ArxivClient,
    output_root: Path,
    run_id: str | None = None,
    clock: Callable[[], datetime] | None = None,
    use_cache: bool = True,
    cache_ttl: timedelta = DEFAULT_CACHE_TTL,
    cache_requires_commit_marker: bool = False,
    query_plan: SearchQueryPlan | None = None,
) -> PersistedSearchRun:
    """Resolve one search from a fresh cache entry or the arXiv network API."""

    if cache_ttl < timedelta(0):
        raise ValueError("cache_ttl cannot be negative.")

    current_time = clock or (lambda: datetime.now(timezone.utc))
    created_time = _as_utc(current_time())
    created_at = _format_time(created_time)
    plan = query_plan or build_arxiv_query(query_text)
    request = build_arxiv_request(plan)
    request_data = _request_data(plan, request)
    cache_key = _cache_key(request_data)
    request_data["cache_key"] = cache_key
    if cache_requires_commit_marker:
        request_data["cache_requires_commit_marker"] = True

    cache_entry = None
    if use_cache and cache_ttl > timedelta(0):
        cache_entry = _find_fresh_cache(
            output_root,
            cache_key=cache_key,
            now=created_time,
            cache_ttl=cache_ttl,
        )

    resolved_run_id = run_id or f"search-{uuid4().hex[:12]}"
    run_dir = output_root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "request.json", request_data)

    if cache_entry is not None:
        snapshot = _save_snapshot(
            run_dir,
            cache_entry.body,
            source="cache",
            origin_run_id=cache_entry.origin_run_id,
            origin_source=cache_entry.origin_source,
        )
        _write_json(
            run_dir / "candidates.json",
            _candidate_data(cache_entry.feed),
        )
        run_data = {
            "schema_version": 1,
            "parser_version": ARXIV_PARSER_VERSION,
            "run_id": resolved_run_id,
            "status": "success",
            "created_at": created_at,
            "fetched_at": cache_entry.fetched_at,
            "http_status": None,
            "candidate_count": len(cache_entry.feed.candidates),
            "raw_record_count": len(cache_entry.feed.records),
            "attempts": [],
            "response_headers": [],
            "snapshot": snapshot,
            "cache": {
                "hit": True,
                "ttl_seconds": int(cache_ttl.total_seconds()),
                "age_seconds": cache_entry.age_seconds,
                "source_run_id": cache_entry.source_run_id,
            },
            "feed": _feed_data(cache_entry.feed),
            "error": None,
        }
        _write_json(run_dir / "run.json", run_data)
        return PersistedSearchRun(
            run_id=resolved_run_id,
            run_dir=run_dir,
            status="success",
            created_at=created_at,
            fetched_at=cache_entry.fetched_at,
            feed=cache_entry.feed,
            error_kind=None,
            error_message=None,
            cache_hit=True,
            snapshot_source="cache",
        )

    try:
        result = client.fetch(plan)
    except ArxivSearchError as error:
        snapshot = _save_snapshot(run_dir, error.response_body, source="network")
        run_data = {
            "schema_version": 1,
            "parser_version": ARXIV_PARSER_VERSION,
            "run_id": resolved_run_id,
            "status": "failed",
            "created_at": created_at,
            "fetched_at": error.fetched_at,
            "http_status": error.status_code,
            "candidate_count": None,
            "raw_record_count": None,
            "attempts": [asdict(attempt) for attempt in error.attempts],
            "response_headers": _headers_to_json(error.response_headers),
            "snapshot": snapshot,
            "cache": {
                "hit": False,
                "ttl_seconds": int(cache_ttl.total_seconds()),
            },
            "error": {"kind": error.kind, "message": str(error)},
        }
        _write_json(run_dir / "run.json", run_data)
        return PersistedSearchRun(
            run_id=resolved_run_id,
            run_dir=run_dir,
            status="failed",
            created_at=created_at,
            fetched_at=error.fetched_at,
            feed=None,
            error_kind=error.kind,
            error_message=str(error),
            cache_hit=False,
            snapshot_source=snapshot["source"] if snapshot else None,
        )

    snapshot = _save_snapshot(run_dir, result.response_body, source="network")
    _write_json(run_dir / "candidates.json", _candidate_data(result.feed))
    run_data = {
        "schema_version": 1,
        "parser_version": ARXIV_PARSER_VERSION,
        "run_id": resolved_run_id,
        "status": "success",
        "created_at": created_at,
        "fetched_at": result.fetched_at,
        "http_status": result.attempts[-1].status_code,
        "candidate_count": len(result.feed.candidates),
        "raw_record_count": len(result.feed.records),
        "attempts": [asdict(attempt) for attempt in result.attempts],
        "response_headers": _headers_to_json(result.response_headers),
        "snapshot": snapshot,
        "cache": {
            "hit": False,
            "ttl_seconds": int(cache_ttl.total_seconds()),
        },
        "feed": _feed_data(result.feed),
        "error": None,
    }
    _write_json(run_dir / "run.json", run_data)
    return PersistedSearchRun(
        run_id=resolved_run_id,
        run_dir=run_dir,
        status="success",
        created_at=created_at,
        fetched_at=result.fetched_at,
        feed=result.feed,
        error_kind=None,
        error_message=None,
        cache_hit=False,
        snapshot_source="network",
    )


def publish_search_cache(run_dir: Path) -> None:
    """Make a successful worker run cacheable after its Job commit succeeds."""

    request_data = _read_json_object(run_dir / "request.json", "saved request")
    if request_data.get("mode") in ("resilient_search", "openalex_arxiv"):
        from backend.app.search.fallback import publish_fallback_cache
        publish_fallback_cache(run_dir)
        return
    if request_data.get("mode") == "multi_query":
        from backend.app.search.multi_query import publish_multi_cache
        publish_multi_cache(run_dir)
        return
    if request_data.get("cache_requires_commit_marker") is not True:
        raise InvalidSearchSnapshot("This search run does not require cache publication.")
    run_data = _read_json_object(run_dir / "run.json", "saved run")
    if run_data.get("status") != "success":
        raise InvalidSearchSnapshot("Only a successful run can enter the cache.")
    run_id = run_data.get("run_id")
    snapshot = run_data.get("snapshot")
    if not isinstance(run_id, str) or not isinstance(snapshot, dict):
        raise InvalidSearchSnapshot("The run has no cache publication identity.")
    snapshot_sha256 = snapshot.get("sha256")
    if not isinstance(snapshot_sha256, str):
        raise InvalidSearchSnapshot("The run has no snapshot digest.")

    marker = run_dir / "cache-committed.json"
    temporary = run_dir / f".cache-committed-{uuid4().hex}.tmp"
    try:
        _write_json(
            temporary,
            {"run_id": run_id, "snapshot_sha256": snapshot_sha256},
        )
        temporary.replace(marker)
    finally:
        temporary.unlink(missing_ok=True)


def replay_search_run(run_dir: Path) -> ReplayedSearchRun:
    """Rebuild one saved candidate list without network access or new files."""

    source_run_dir = run_dir.resolve()
    request_data = _read_json_object(
        source_run_dir / "request.json",
        "saved request",
    )
    if request_data.get("mode") in ("resilient_search", "openalex_arxiv"):
        from backend.app.search.fallback import replay_fallback_search
        return replay_fallback_search(source_run_dir)
    if request_data.get("mode") == "multi_query":
        from backend.app.search.multi_query import replay_multi_search
        return replay_multi_search(source_run_dir)
    calculated_cache_key = _cache_key(request_data)
    stored_cache_key = request_data.get("cache_key")
    if stored_cache_key is not None and stored_cache_key != calculated_cache_key:
        raise InvalidSearchSnapshot("The saved request cache key does not match.")

    run_data = _read_json_object(source_run_dir / "run.json", "run metadata")
    if run_data.get("status") != "success":
        raise InvalidSearchSnapshot("Only successful search runs can be replayed.")

    source_run_id = run_data.get("run_id")
    if not isinstance(source_run_id, str) or not source_run_id:
        raise InvalidSearchSnapshot("The saved run is missing its run ID.")

    body, digest = _verified_snapshot_bytes(source_run_dir, run_data)
    try:
        feed = parse_atom_feed(body)
    except (ArxivAPIError, InvalidArxivFeed) as error:
        raise InvalidSearchSnapshot(
            f"The saved Atom snapshot cannot be parsed: {error}"
        ) from error

    saved_candidates = _read_json_value(
        source_run_dir / "candidates.json",
        "saved candidates",
    )
    candidates_match = _canonical_json(saved_candidates) == _canonical_json(
        _candidate_data(feed)
    )
    saved_parser_version = run_data.get("parser_version")
    if not isinstance(saved_parser_version, int):
        saved_parser_version = None

    return ReplayedSearchRun(
        source_run_id=source_run_id,
        source_run_dir=source_run_dir,
        feed=feed,
        candidates_match=candidates_match,
        saved_parser_version=saved_parser_version,
        current_parser_version=ARXIV_PARSER_VERSION,
        snapshot_sha256=digest,
    )


def _request_data(plan: SearchQueryPlan, request) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "query_contract_version": plan.contract_version,
        "cache_contract_version": SEARCH_CACHE_VERSION,
        "query_text": plan.query_text,
        "normalized_query_text": plan.normalized_query_text,
        "query_terms": list(plan.query_terms),
        "search_query": plan.search_query,
        "endpoint": request.endpoint,
        "params": dict(request.params),
        "headers": dict(request.headers),
        "timeout_seconds": request.timeout_seconds,
    }


def _cache_key(request_data: dict[str, Any]) -> str:
    query_contract_version = request_data.get("query_contract_version")
    normalized_query_text = request_data.get("normalized_query_text")
    endpoint = request_data.get("endpoint")
    params = request_data.get("params")
    if (
        not isinstance(query_contract_version, int)
        or not isinstance(normalized_query_text, str)
        or not isinstance(endpoint, str)
        or not isinstance(params, dict)
    ):
        raise InvalidSearchSnapshot("The saved request cache identity is incomplete.")

    identity = {
        "cache_contract_version": SEARCH_CACHE_VERSION,
        "query_contract_version": query_contract_version,
        "normalized_query_text": normalized_query_text,
        "endpoint": endpoint,
        "params": params,
    }
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def _find_fresh_cache(
    output_root: Path,
    *,
    cache_key: str,
    now: datetime,
    cache_ttl: timedelta,
) -> _CacheEntry | None:
    if not output_root.exists():
        return None

    matches: list[_CacheEntry] = []
    for candidate_dir in output_root.glob("search-*"):
        if not candidate_dir.is_dir():
            continue
        try:
            request_data = _read_json_object(
                candidate_dir / "request.json",
                "cached request",
            )
            if _cache_key(request_data) != cache_key:
                continue
            stored_key = request_data.get("cache_key")
            if stored_key is not None and stored_key != cache_key:
                continue

            run_data = _read_json_object(
                candidate_dir / "run.json",
                "cached run metadata",
            )
            if run_data.get("status") != "success":
                continue
            requires_marker = request_data.get("cache_requires_commit_marker")
            if requires_marker is True:
                marker = _read_json_object(
                    candidate_dir / "cache-committed.json", "cache commit marker"
                )
                snapshot = run_data.get("snapshot")
                if (
                    not isinstance(snapshot, dict)
                    or marker.get("run_id") != run_data.get("run_id")
                    or marker.get("snapshot_sha256") != snapshot.get("sha256")
                ):
                    continue
            elif requires_marker not in (None, False):
                continue
            fetched_at = run_data.get("fetched_at")
            fetched_time = _parse_time(fetched_at)
            age = now - fetched_time
            if age < timedelta(0) or age >= cache_ttl:
                continue

            body, _ = _verified_snapshot_bytes(candidate_dir.resolve(), run_data)
            feed = parse_atom_feed(body)
            source_run_id = run_data.get("run_id")
            if not isinstance(source_run_id, str) or not source_run_id:
                continue
            snapshot = run_data.get("snapshot")
            if not isinstance(snapshot, dict):
                continue
            origin_run_id = snapshot.get("origin_run_id", source_run_id)
            origin_source = snapshot.get("origin_source", snapshot.get("source"))
            if not isinstance(origin_run_id, str) or not isinstance(origin_source, str):
                continue

            matches.append(
                _CacheEntry(
                    source_run_id=source_run_id,
                    body=body,
                    fetched_at=_format_time(fetched_time),
                    fetched_time=fetched_time,
                    feed=feed,
                    origin_run_id=origin_run_id,
                    origin_source=origin_source,
                    age_seconds=int(age.total_seconds()),
                )
            )
        except (
            ArxivAPIError,
            InvalidArxivFeed,
            InvalidSearchSnapshot,
            OSError,
            TypeError,
            ValueError,
        ):
            continue

    if not matches:
        return None
    return max(matches, key=lambda item: item.fetched_time)


def _verified_snapshot_bytes(
    run_dir: Path,
    run_data: dict[str, Any],
) -> tuple[bytes, str]:
    snapshot = run_data.get("snapshot")
    if not isinstance(snapshot, dict):
        raise InvalidSearchSnapshot("The saved run has no Atom snapshot metadata.")

    relative_path = snapshot.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise InvalidSearchSnapshot("The snapshot path is missing.")
    snapshot_path = (run_dir / relative_path).resolve()
    if not snapshot_path.is_relative_to(run_dir):
        raise InvalidSearchSnapshot("The snapshot path leaves the run directory.")

    try:
        body = snapshot_path.read_bytes()
    except OSError as error:
        raise InvalidSearchSnapshot("The saved Atom snapshot cannot be read.") from error

    digest = hashlib.sha256(body).hexdigest()
    if digest != snapshot.get("sha256"):
        raise InvalidSearchSnapshot("The saved Atom snapshot SHA-256 does not match.")
    if len(body) != snapshot.get("byte_count"):
        raise InvalidSearchSnapshot("The saved Atom snapshot byte count does not match.")
    return body, digest


def _save_snapshot(
    run_dir: Path,
    body: bytes | None,
    *,
    source: str,
    origin_run_id: str | None = None,
    origin_source: str | None = None,
) -> dict[str, Any] | None:
    if body is None:
        return None
    filename = "response.atom"
    (run_dir / filename).write_bytes(body)
    snapshot: dict[str, Any] = {
        "source": source,
        "path": filename,
        "sha256": hashlib.sha256(body).hexdigest(),
        "byte_count": len(body),
    }
    if origin_run_id is not None:
        snapshot["origin_run_id"] = origin_run_id
    if origin_source is not None:
        snapshot["origin_source"] = origin_source
    return snapshot


def _candidate_data(feed: ParsedArxivFeed) -> list[dict[str, Any]]:
    return json.loads(
        json.dumps(
            [asdict(candidate) for candidate in feed.candidates],
            ensure_ascii=False,
        )
    )


def _feed_data(feed: ParsedArxivFeed) -> dict[str, Any]:
    return {
        "title": feed.feed_title,
        "id": feed.feed_id,
        "updated_at": feed.feed_updated_at,
        "self_url": feed.self_url,
        "total_results": feed.total_results,
        "start_index": feed.start_index,
        "items_per_page": feed.items_per_page,
    }


def _headers_to_json(headers: tuple[tuple[str, str], ...]) -> list[dict[str, str]]:
    return [{"name": name, "value": value} for name, value in headers]


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    value = _read_json_value(path, description)
    if not isinstance(value, dict):
        raise InvalidSearchSnapshot(f"The {description} is not a JSON object.")
    return value


def _read_json_value(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidSearchSnapshot(f"The {description} cannot be read.") from error


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise InvalidSearchSnapshot("The saved fetched_at value is missing.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise InvalidSearchSnapshot("The saved fetched_at value is invalid.") from error
    return _as_utc(parsed)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Clock values must include a timezone.")
    return value.astimezone(timezone.utc)


def _format_time(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="seconds").replace("+00:00", "Z")
