import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.search.arxiv import ArxivAPIError, InvalidArxivFeed, parse_atom_feed
from backend.app.search.client import ARXIV_ENDPOINT, retry_after_seconds
from backend.app.search.evaluation import InvalidSearchEvalCase, load_search_cases
from backend.app.search.request_gate import (
    DEFAULT_ARXIV_REQUEST_GATE,
    ArxivCooldownActive,
    ArxivRequestGate,
)
from backend.app.search.service import build_arxiv_query, replay_search_run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether missing frozen seeds fail the query terms or only "
            "fall below the saved top 20."
        )
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "eval" / "search_cases.jsonl",
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cases = load_search_cases(args.cases)
        report = _read_json(args.report_dir / "report.json")
    except (InvalidSearchEvalCase, OSError, ValueError) as error:
        print(f"[FAIL] inputs: {error}")
        return 1

    print(
        "[METHOD] arXiv search_query + id_list intersection over each case's "
        "frozen seeds."
    )
    if not args.live:
        print(f"[PREVIEW] {len(cases)} diagnostic requests; add --live to run.")
        return 0

    output_path = args.report_dir / "seed-diagnosis.json"
    atom_dir = args.report_dir / "seed-diagnosis-atoms"
    if output_path.exists() or atom_dir.exists():
        print("[FAIL] diagnostic output already exists.")
        return 1
    atom_dir.mkdir(parents=True)

    report_cases = {
        item["case_id"]: item
        for item in report.get("cases", [])
        if item.get("status") == "success"
    }
    records = []
    with httpx.Client(
        follow_redirects=False,
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
    ) as client:
        for case in cases:
            report_case = report_cases.get(case.case_id)
            if report_case is None:
                print(f"[FAIL] no successful report case for {case.case_id}")
                return 1
            replay = replay_search_run(Path(report_case["run_dir"]))
            top20_ids = {candidate.arxiv_id for candidate in replay.feed.candidates}
            known_ids = [paper.arxiv_id for paper in case.known_relevant]
            plan = build_arxiv_query(case.query)

            try:
                response, request_started_at = _send_intersection_query(
                    client, DEFAULT_ARXIV_REQUEST_GATE, plan.search_query, known_ids
                )
            except (ArxivCooldownActive, httpx.RequestError) as error:
                print(f"[FAIL] {case.case_id}: {error}")
                return 1
            atom_path = atom_dir / f"{case.case_id}.atom"
            atom_path.write_bytes(response.content)
            if not 200 <= response.status_code < 300:
                print(f"[FAIL] {case.case_id}: HTTP {response.status_code}")
                return 1
            try:
                feed = parse_atom_feed(response.content)
            except (ArxivAPIError, InvalidArxivFeed) as error:
                print(f"[FAIL] {case.case_id}: {error}")
                return 1

            eligible_ids = {candidate.arxiv_id for candidate in feed.candidates}
            unexpected = eligible_ids - set(known_ids)
            if unexpected:
                print(f"[FAIL] {case.case_id}: unexpected IDs {sorted(unexpected)}")
                return 1
            hit_ids = [paper_id for paper_id in known_ids if paper_id in top20_ids]
            ranking_miss_ids = [
                paper_id
                for paper_id in known_ids
                if paper_id in eligible_ids and paper_id not in top20_ids
            ]
            query_miss_ids = [
                paper_id for paper_id in known_ids if paper_id not in eligible_ids
            ]
            record = {
                "case_id": case.case_id,
                "query": case.query,
                "run_id": report_case["run_id"],
                "request_started_at": request_started_at,
                "http_status": response.status_code,
                "atom_snapshot": {
                    "path": str(atom_path.relative_to(args.report_dir)),
                    "sha256": hashlib.sha256(response.content).hexdigest(),
                    "byte_count": len(response.content),
                },
                "known_count": len(known_ids),
                "top20_hit_ids": hit_ids,
                "ranking_miss_ids": ranking_miss_ids,
                "query_term_miss_ids": query_miss_ids,
            }
            records.append(record)
            print(
                f"[DIAGNOSIS] {case.case_id}: hit={len(hit_ids)}, "
                f"below_top20={len(ranking_miss_ids)}, "
                f"query_miss={len(query_miss_ids)}"
            )

    result = {
        "schema_version": 1,
        "method": (
            "For each case, send the frozen search_query together with only its "
            "frozen arXiv id_list. Per arXiv API semantics, the response is the "
            "intersection. Eligible seeds absent from the saved top 20 are ranking "
            "misses; seeds absent from the intersection are query-term misses."
        ),
        "case_count": len(records),
        "aggregate": {
            "known_count": sum(item["known_count"] for item in records),
            "top20_hit_count": sum(len(item["top20_hit_ids"]) for item in records),
            "ranking_miss_count": sum(
                len(item["ranking_miss_ids"]) for item in records
            ),
            "query_term_miss_count": sum(
                len(item["query_term_miss_ids"]) for item in records
            ),
        },
        "cases": records,
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Results: {output_path}")
    return 0


def _send_intersection_query(
    client: httpx.Client,
    gate: ArxivRequestGate,
    search_query: str,
    known_ids: list[str],
) -> tuple[httpx.Response, str]:
    with gate.slot() as slot:
        request_started_at = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")
        response = client.get(
            ARXIV_ENDPOINT,
            params={
                "search_query": search_query,
                "id_list": ",".join(known_ids),
                "start": "0",
                "max_results": str(len(known_ids)),
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            headers={"User-Agent": "PaperTrail/0.1"},
            timeout=20.0,
        )
        retry_after = retry_after_seconds(
            tuple(response.headers.multi_items()), datetime.now(timezone.utc)
        )
        if retry_after is not None:
            slot.defer_for(retry_after)
    return response, request_started_at


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
