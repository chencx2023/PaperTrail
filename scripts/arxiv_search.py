import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.search.client import ArxivClient, build_arxiv_request
from backend.app.search.service import (
    InvalidSearchSnapshot,
    build_arxiv_query,
    replay_search_run,
    run_search,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or run PaperTrail's single-query arXiv search."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--query",
        help='Space-separated English keywords, for example "LLM agent memory".',
    )
    source.add_argument(
        "--replay",
        type=Path,
        help="Offline replay of an existing successful search run directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "runs",
        help="Directory in which the search run folder is created.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Resolve the query from the 24-hour cache or arXiv API.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass the 24-hour cache; requires --query and --live.",
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    client: ArxivClient | None = None,
) -> int:
    args = parse_args(argv)
    if args.replay is not None:
        if args.live or args.refresh:
            print("[FAIL] --replay cannot be combined with --live or --refresh.")
            return 2
        try:
            replay = replay_search_run(args.replay)
        except InvalidSearchSnapshot as error:
            print(f"[FAIL] replay: {error}")
            return 1
        print(
            f"[REPLAY] {len(replay.feed.candidates)} candidates from "
            f"{replay.source_run_id}"
        )
        print(
            f"[PARSER] saved={replay.saved_parser_version}; "
            f"current={replay.current_parser_version}"
        )
        if replay.candidates_match:
            print("[MATCH] Saved candidates are identical to the offline rebuild.")
            return 0
        print("[MISMATCH] Saved candidates differ from the offline rebuild.")
        return 1

    if args.refresh and not args.live:
        print("[FAIL] --refresh requires --live.")
        return 2

    plan = build_arxiv_query(args.query)
    request = build_arxiv_request(plan)

    print(f"[QUERY] {plan.normalized_query_text}")
    print(f"[ARXIV] {plan.search_query}")
    print(f"[PARAMS] {dict(request.params)}")
    if not args.live:
        print("[PREVIEW] No network request was made. Add --live to execute it.")
        return 0

    run = run_search(
        args.query,
        client=client or ArxivClient(),
        output_root=args.output_root,
        use_cache=not args.refresh,
    )
    if run.status != "success":
        print(f"[FAIL] {run.error_kind}: {run.error_message}")
        print(f"Results: {run.run_dir}")
        return 1

    candidates = run.feed.candidates if run.feed is not None else ()
    if run.cache_hit:
        print(f"[SOURCE] cache; original fetched_at={run.fetched_at}")
    else:
        print(f"[SOURCE] {run.snapshot_source}")
    print(f"[RESULTS] {len(candidates)} candidates")
    for candidate in candidates:
        print(
            f"{candidate.display_rank:>2}. {candidate.title} "
            f"[{candidate.versioned_id}]"
        )
    print(f"Results: {run.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
