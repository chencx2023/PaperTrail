"""Fallback discovery with raw OpenAlex responses and verified arXiv versions.

OpenAlex supplies discovery ranks only. Candidate metadata and numeric versions
come from saved arXiv abstract pages; no Atom feed or RRF result is fabricated.
"""

from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import httpx

from backend.app.search.arxiv import (
    ARXIV_PARSER_VERSION, ArxivRecord, InvalidArxivFeed, ParsedArxivFeed,
    _deduplicate, _parse_identifier,
)
from backend.app.search.client import (
    ArxivCooldownError, ArxivRequest, ArxivTransportError, HttpxArxivTransport,
)
from backend.app.search.request_gate import ArxivRequestGate, DEFAULT_GATE_PATH
from backend.app.search.service import (
    InvalidSearchSnapshot, PersistedSearchRun, ReplayedSearchRun,
    _candidate_data, _canonical_json, _read_json_object, _read_json_value,
    _verified_snapshot_bytes, _write_json, build_arxiv_query,
    publish_search_cache, replay_search_run, run_search,
)


OPENALEX_ENDPOINT = "https://api.openalex.org/works"
ARXIV_SOURCE = "S4306400194"
FALLBACK_LIMIT = 8
FALLBACK_CONTRACT = 1
# API and HTML have distinct server cooldowns, but share a single traffic gate.
ARXIV_TRAFFIC_GATE = ArxivRequestGate(DEFAULT_GATE_PATH.with_name("arxiv-traffic.lock"))
ARXIV_PAGES_GATE = ArxivRequestGate(DEFAULT_GATE_PATH.with_name("arxiv-pages.lock"))


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(root, name, source):
    return {"path": name, "sha256": _digest(root / name),
            "byte_count": (root / name).stat().st_size, "source": source}


class _AbstractPage(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta = {}
        self.history_depth = 0
        self.history = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "meta" and values.get("name") and values.get("content"):
            self.meta.setdefault(values["name"], []).append(values["content"])
        if tag == "div":
            if self.history_depth:
                self.history_depth += 1
            elif "submission-history" in values.get("class", "").split():
                self.history_depth = 1

    def handle_endtag(self, tag):
        if tag == "div" and self.history_depth:
            self.history_depth -= 1

    def handle_data(self, data):
        if self.history_depth:
            self.history.append(data)


def parse_abstract_page(body: bytes, arxiv_id: str, rank: int) -> ArxivRecord:
    """Reject missing history, unrelated pages, and unconfirmed version numbers."""
    page = _AbstractPage()
    page.feed(body.decode("utf-8"))
    meta = page.meta

    def required(key):
        value = " ".join(meta.get(key, [""])[0].split())
        if not value:
            raise ValueError(f"arXiv page has no {key}.")
        return value

    identifier = required("citation_arxiv_id")
    base, page_version, _ = _parse_identifier(f"https://arxiv.org/abs/{identifier}")
    if base != arxiv_id:
        raise ValueError("The abstract page identifies a different paper.")
    versions = [int(v) for v in re.findall(r"\[v([1-9]\d*)\]", " ".join(page.history))]
    if not versions:
        raise ValueError("No numeric version was observed in submission history.")
    version = max(versions)
    if page_version is not None and page_version != version:
        raise ValueError("The page version and submission history disagree.")
    pdf = urlsplit(required("citation_pdf_url"))
    if (pdf.scheme not in ("https", "http") or pdf.netloc != "arxiv.org"
            or pdf.path not in (f"/pdf/{arxiv_id}", f"/pdf/{arxiv_id}v{version}")
            or pdf.query or pdf.fragment):
        raise ValueError("The page PDF link does not identify the verified paper.")
    authors = tuple(" ".join(a.split()) for a in meta.get("citation_author", []) if a.strip())
    if not authors:
        raise ValueError("The abstract page has no authors.")
    published = required("citation_date").replace("/", "-")
    updated = meta.get("citation_online_date", [published])[0].replace("/", "-")
    # Validate calendar dates without inventing time-of-day information.
    datetime.strptime(published, "%Y-%m-%d")
    datetime.strptime(updated, "%Y-%m-%d")
    versioned = f"{arxiv_id}v{version}"
    return ArxivRecord(rank, arxiv_id, versioned, version,
        f"https://arxiv.org/abs/{versioned}", f"https://arxiv.org/abs/{versioned}",
        f"https://arxiv.org/pdf/{versioned}", required("citation_title"),
        required("citation_abstract"), authors, published, updated,
        None, (), None, None, None, None)


def _discovered_ids(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("OpenAlex did not return a works result list.")
    ids = []
    for work in payload["results"]:
        if not isinstance(work, dict):
            raise ValueError("Invalid OpenAlex work.")
        locations = work.get("locations", [])
        if not isinstance(locations, list):
            raise ValueError("Invalid OpenAlex locations.")
        for location in locations:
            if not isinstance(location, dict):
                continue
            source = location.get("source") or {}
            if not isinstance(source, dict) or source.get("id") != f"https://openalex.org/{ARXIV_SOURCE}":
                continue
            link = location.get("landing_page_url")
            if not isinstance(link, str):
                continue
            parsed = urlsplit(link)
            if parsed.scheme not in ("http", "https") or parsed.netloc != "arxiv.org" or parsed.query or parsed.fragment:
                continue
            try:
                base, _, _ = _parse_identifier(link)
            except InvalidArxivFeed:
                continue
            if base not in ids:
                ids.append(base)
            break
        if len(ids) >= FALLBACK_LIMIT:
            break
    return ids


def _provider_request(query_text):
    plan = build_arxiv_query(query_text)
    return {"mode": "openalex_arxiv", "contract": FALLBACK_CONTRACT,
        "query_text": plan.normalized_query_text, "endpoint": OPENALEX_ENDPOINT,
        "params": {"search": plan.normalized_query_text,
            "filter": f"locations.source.id:{ARXIV_SOURCE}", "per_page": FALLBACK_LIMIT,
            "select": "id,display_name,locations"}, "cache_requires_commit_marker": True}


class OpenAlexArxivFallback:
    def __init__(self, *, client=None, page_transport=None):
        self.client = client or httpx.Client(follow_redirects=False, timeout=20)
        self.pages = page_transport or HttpxArxivTransport(
            request_gate=ARXIV_PAGES_GATE, traffic_gate=ARXIV_TRAFFIC_GATE)

    def close(self):
        self.client.close()
        self.pages.close()

    def run(self, query_text, *, output_root: Path, run_id: str):
        root = Path(output_root) / run_id
        root.mkdir(parents=True, exist_ok=False)
        request = _provider_request(query_text)
        _write_json(root / "request.json", request)
        created_at = _now()
        cached = _cached_provider(output_root, request)
        if cached:
            cached_root, old_run, manifest = cached
            for name in ["openalex.json", "manifest.json"] + [s["path"] for s in manifest["pages"] if s.get("path")]:
                (root / name).write_bytes((cached_root / name).read_bytes())
            feed = _rebuild_provider(root, manifest)
            return _finish(root, created_at, old_run["fetched_at"], feed, "openalex_arxiv",
                cache_hit=True, cache_origin=cached_root.name)
        manifest = {"request": request, "pages": [], "openalex_status": None,
                    "openalex_requests": 0, "page_requests": 0}
        records = []
        fetched_at = None
        failure = None
        deadline = time.monotonic() + 100
        try:
            manifest["openalex_requests"] = 1
            response = self.client.get(OPENALEX_ENDPOINT, params=request["params"],
                headers={"User-Agent": "PaperTrail/0.1"}, timeout=20)
            fetched_at = _now()
            (root / "openalex.json").write_bytes(response.content)
            manifest["openalex_sha256"] = _digest(root / "openalex.json")
            manifest["openalex_status"] = response.status_code
            response.raise_for_status()
            ids = _discovered_ids(response.json())
            for rank, arxiv_id in enumerate(ids, 1):
                entry = {"arxiv_id": arxiv_id, "rank": rank, "path": None,
                         "status_code": None, "error": None}
                manifest["pages"].append(entry)
                if time.monotonic() >= deadline:
                    entry["error"] = "Fallback time limit reached; not requested."
                    continue
                try:
                    manifest["page_requests"] += 1
                    result = self.pages.send(ArxivRequest(f"https://arxiv.org/abs/{arxiv_id}", (),
                        (("User-Agent", "PaperTrail/0.1"),), min(15, max(1, deadline - time.monotonic()))))
                    entry["status_code"] = result.status_code
                    entry["path"] = f"page-{rank}.html"
                    (root / entry["path"]).write_bytes(result.body)
                    entry["sha256"] = _digest(root / entry["path"])
                    if result.status_code != 200:
                        raise ValueError(f"arXiv abstract page HTTP {result.status_code}.")
                    records.append(parse_abstract_page(result.body, arxiv_id, rank))
                except (ArxivTransportError, ValueError) as error:
                    entry["error"] = str(error)
                    if isinstance(error, ArxivCooldownError):
                        manifest["page_requests"] -= 1
                        break
                    # A server-side limit is a stop signal, not an invitation to probe.
                    if entry["status_code"] in (403, 429):
                        break
            if not records:
                failure = "备用检索没有取得可核实版本的论文，请查看检索记录。"
        except (httpx.HTTPError, ValueError) as error:
            failure = f"备用检索未完成：{type(error).__name__}: {error}"
        _write_json(root / "manifest.json", manifest)
        feed = _provider_feed(tuple(records), fetched_at or created_at) if records else None
        return _finish(root, created_at, fetched_at, feed, "openalex_arxiv", error=failure)


def _provider_feed(records, fetched_at):
    return ParsedArxivFeed("OpenAlex discovery; arXiv metadata verified", OPENALEX_ENDPOINT,
        fetched_at, OPENALEX_ENDPOINT, len(records), 0, len(records), records, _deduplicate(records))


def _finish(root, created_at, fetched_at, feed, source, *, error=None, cache_hit=False, cache_origin=None):
    success = feed is not None
    _write_json(root / "candidates.json", _candidate_data(feed) if success else [])
    _write_json(root / "run.json", {"run_id": root.name,
        "status": "success" if success else "failed", "created_at": created_at,
        "fetched_at": fetched_at, "parser_version": ARXIV_PARSER_VERSION,
        "snapshot": _snapshot(root, "manifest.json", source),
        "candidate_count": len(feed.candidates) if success else 0,
        "cache": {"hit": cache_hit, "origin_run_id": cache_origin},
        "error": None if success else {"kind": "search_unavailable", "message": error}})
    return PersistedSearchRun(root.name, root, "success" if success else "failed",
        created_at, fetched_at, feed, None if success else "search_unavailable",
        None if success else error, cache_hit, source)


def _cached_provider(output_root, request):
    candidates = []
    for root in Path(output_root).glob("search-*-openalex"):
        try:
            if _read_json_object(root / "request.json", "cache request") != request:
                continue
            run = _read_json_object(root / "run.json", "cache run")
            marker = _read_json_object(root / "cache-committed.json", "cache publication")
            fetched = datetime.fromisoformat(run["fetched_at"].replace("Z", "+00:00"))
            age = datetime.now(timezone.utc) - fetched
            if not timedelta(0) <= age < timedelta(hours=24):
                continue
            if marker != {"run_id": root.name, "snapshot_sha256": run["snapshot"]["sha256"]}:
                continue
            replay = replay_fallback_search(root)
            if not replay.candidates_match:
                continue
            body, _ = _verified_snapshot_bytes(root.resolve(), run)
            candidates.append((fetched, root, run, json.loads(body)))
        except (InvalidSearchSnapshot, OSError, ValueError, KeyError, TypeError):
            continue
    if not candidates:
        return None
    _, root, run, manifest = max(candidates, key=lambda item: item[0])
    return root, run, manifest


def _rebuild_provider(root, manifest):
    request = _read_json_object(root / "request.json", "provider request")
    if request != _provider_request(request["query_text"]) or manifest["request"] != request:
        raise InvalidSearchSnapshot("The fallback query or source contract changed.")
    if manifest["openalex_status"] != 200 or _digest(root / "openalex.json") != manifest["openalex_sha256"]:
        raise InvalidSearchSnapshot("The OpenAlex discovery snapshot changed.")
    ids = _discovered_ids(json.loads((root / "openalex.json").read_bytes()))
    pages = manifest["pages"]
    if not isinstance(pages, list) or len(pages) > len(ids):
        raise InvalidSearchSnapshot("Invalid verified page count.")
    records = []
    for rank, page in enumerate(pages, 1):
        if page["arxiv_id"] != ids[rank - 1] or page["rank"] != rank:
            raise InvalidSearchSnapshot("The verified page does not match discovery order.")
        if page.get("path"):
            if page["path"] != f"page-{rank}.html" or _digest(root / page["path"]) != page["sha256"]:
                raise InvalidSearchSnapshot("The arXiv abstract snapshot changed.")
            if page["status_code"] == 200 and page["error"] is None:
                records.append(parse_abstract_page((root / page["path"]).read_bytes(), page["arxiv_id"], rank))
    if not records:
        raise InvalidSearchSnapshot("No verified explicit versions are available.")
    run = _read_json_object(root / "run.json", "provider run") if (root / "run.json").exists() else {}
    return _provider_feed(tuple(records), run.get("fetched_at") or _now())


def run_resilient_search(query_text, *, client, fallback, output_root, run_id, plan=None, generation=None):
    from backend.app.search.multi_query import run_multi_search
    from backend.app.search.query_builder import plan_from_data
    if not re.fullmatch(r"[A-Za-z0-9-]{1,64}", run_id):
        raise ValueError("Invalid search run ID.")
    query_text = build_arxiv_query(query_text).normalized_query_text
    if plan is not None:
        plan = plan_from_data(plan.as_data())
        if plan.baseline_query_text != query_text:
            raise ValueError("The multi-query baseline changed.")
    root = Path(output_root) / run_id
    root.mkdir(parents=True, exist_ok=False)
    request = {"mode": "resilient_search", "contract": FALLBACK_CONTRACT,
               "query_text": query_text, "plan": plan.as_data() if plan else None}
    _write_json(root / "request.json", request)
    created_at = _now()
    native = (run_multi_search(plan, client=client, output_root=output_root,
                run_id=f"{run_id}-arxiv", generation=generation) if plan else
        run_search(query_text, client=client, output_root=output_root,
            run_id=f"{run_id}-arxiv", cache_requires_commit_marker=True))
    children = [_child_identity(native)]
    selected = native
    provider = "arxiv"
    if native.status != "success":
        selected = fallback.run(query_text, output_root=output_root, run_id=f"{run_id}-openalex")
        children.append(_child_identity(selected))
        provider = "openalex_arxiv"
    _write_json(root / "manifest.json", {"request": request, "children": children,
        "provider": provider, "selected_run_id": selected.run_id})
    return _finish(root, created_at, selected.fetched_at, selected.feed, "resilient_search",
        error=selected.error_message, cache_hit=selected.cache_hit)


def _child_identity(run):
    return {"run_id": run.run_id, "status": run.status,
        "request_sha256": _digest(run.run_dir / "request.json"),
        "run_sha256": _digest(run.run_dir / "run.json")}


def _verified_wrapper(root, metadata, request):
    body, digest = _verified_snapshot_bytes(root, metadata)
    manifest = json.loads(body)
    if request.get("contract") != FALLBACK_CONTRACT or manifest["request"] != request:
        raise InvalidSearchSnapshot("The saved search contract changed.")
    build_arxiv_query(request["query_text"])
    children = manifest["children"]
    expected_count = 1 if manifest["provider"] == "arxiv" else 2
    if len(children) != expected_count or manifest["provider"] not in ("arxiv", "openalex_arxiv"):
        raise InvalidSearchSnapshot("The fallback source list changed.")
    for index, entry in enumerate(children):
        expected = f"{root.name}-{'arxiv' if index == 0 else 'openalex'}"
        child = (root.parent / expected).resolve()
        if (entry["run_id"] != expected or not child.is_relative_to(root.parent)
                or _digest(child / "request.json") != entry["request_sha256"]
                or _digest(child / "run.json") != entry["run_sha256"]):
            raise InvalidSearchSnapshot("A search source identity or hash changed.")
        child_request = _read_json_object(child / "request.json", "child request")
        child_run = _read_json_object(child / "run.json", "child run")
        if child_run["run_id"] != expected or child_run["status"] != entry["status"]:
            raise InvalidSearchSnapshot("A search source status changed.")
        if index == 0:
            if request["plan"] is None:
                if child_request.get("mode") is not None or child_request["normalized_query_text"] != request["query_text"]:
                    raise InvalidSearchSnapshot("The primary query changed.")
            elif child_request.get("mode") != "multi_query" or child_request["plan"] != request["plan"]:
                raise InvalidSearchSnapshot("The primary multi-query plan changed.")
        elif child_request != _provider_request(request["query_text"]) or children[0]["status"] != "failed":
            raise InvalidSearchSnapshot("The fallback query or activation changed.")
    if manifest["selected_run_id"] != children[-1]["run_id"]:
        raise InvalidSearchSnapshot("The selected source changed.")
    return manifest, digest


def replay_fallback_search(run_dir):
    root = Path(run_dir).resolve()
    try:
        request = _read_json_object(root / "request.json", "search request")
        metadata = _read_json_object(root / "run.json", "search run")
        if metadata["run_id"] != root.name or metadata["status"] != "success":
            raise InvalidSearchSnapshot("Only an owning successful run can be replayed.")
        if request["mode"] == "resilient_search":
            manifest, digest = _verified_wrapper(root, metadata, request)
            selected = replay_search_run(root.parent / manifest["selected_run_id"])
            if not selected.candidates_match:
                raise InvalidSearchSnapshot("Source candidates changed.")
            provenance = {"provider": manifest["provider"], "original_plan": request["plan"],
                "fallback_used": manifest["provider"] == "openalex_arxiv",
                "primary_failure": None}
            if provenance["fallback_used"]:
                from backend.app.search.failure import read_search_failure
                provenance["primary_failure"] = read_search_failure(root.parent / f"{root.name}-arxiv",
                    expected_run_id=f"{root.name}-arxiv", query_text=request["query_text"], query_plan=request["plan"])
                if provenance["primary_failure"] is None:
                    raise InvalidSearchSnapshot("The primary failure record changed.")
            feed, fusion = selected.feed, selected.fusion
        elif request["mode"] == "openalex_arxiv":
            body, digest = _verified_snapshot_bytes(root, metadata)
            feed = _rebuild_provider(root, json.loads(body))
            fusion, provenance = None, {"provider": "openalex_arxiv", "fallback_used": True}
        else:
            raise InvalidSearchSnapshot("Unknown fallback search mode.")
        matches = _canonical_json(_read_json_value(root / "candidates.json", "candidates")) == _canonical_json(_candidate_data(feed))
        if metadata["candidate_count"] != len(feed.candidates):
            raise InvalidSearchSnapshot("The verified candidate count changed.")
        return ReplayedSearchRun(root.name, root, feed, matches, metadata["parser_version"],
            ARXIV_PARSER_VERSION, digest, fusion, provenance)
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, InvalidSearchSnapshot):
            raise
        raise InvalidSearchSnapshot("The saved fallback search is incomplete or invalid.") from error


def publish_fallback_cache(run_dir):
    root = Path(run_dir).resolve()
    replay = replay_fallback_search(root)
    if not replay.candidates_match:
        raise InvalidSearchSnapshot("Saved candidates changed before publication.")
    request = _read_json_object(root / "request.json", "request")
    if request["mode"] == "resilient_search":
        run = _read_json_object(root / "run.json", "run")
        manifest, _ = _verified_wrapper(root, run, request)
        publish_search_cache(root.parent / manifest["selected_run_id"])
    _write_json(root / "cache-committed.json", {"run_id": root.name, "snapshot_sha256": replay.snapshot_sha256})
