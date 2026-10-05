import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.search.client import ArxivClient
from backend.app.search.evaluation import (
    InvalidSearchEvalCase,
    load_search_cases,
    run_frozen_search_evaluation,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PaperTrail's pre-frozen T02 arXiv development cases."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "eval" / "search_cases.jsonl",
        help="Frozen JSONL search cases.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "runs",
        help="Directory for individual search runs.",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=PROJECT_ROOT / "eval" / "reports",
        help="Directory for the evaluation report folder.",
    )
    parser.add_argument(
        "--report-id",
        help="Optional deterministic report directory name.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run cases sequentially against the 24-hour cache or arXiv API.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass the search cache; requires --live.",
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    client: ArxivClient | None = None,
) -> int:
    args = parse_args(argv)
    if args.refresh and not args.live:
        print("[FAIL] --refresh requires --live.")
        return 2

    try:
        cases = load_search_cases(args.cases)
    except InvalidSearchEvalCase as error:
        print(f"[FAIL] cases: {error}")
        return 1

    print(f"[FROZEN] {len(cases)} development cases from {args.cases}")
    for case in cases:
        print(
            f"- {case.case_id}: {case.query} "
            f"({len(case.known_relevant)} known papers)"
        )
    if not args.live:
        print("[PREVIEW] No search was run. Add --live to execute all cases.")
        return 0

    try:
        report, report_dir = run_frozen_search_evaluation(
            cases_path=args.cases,
            output_root=args.output_root,
            report_root=args.report_root,
            client=client or ArxivClient(),
            report_id=args.report_id,
            use_cache=not args.refresh,
        )
    except (InvalidSearchEvalCase, OSError) as error:
        print(f"[FAIL] evaluation: {error}")
        return 1

    for case in report["cases"]:
        if case["status"] == "success":
            print(
                f"[RESULT] {case['case_id']}: "
                f"{case['hit_count']}/{case['known_count']} known papers; "
                f"Recall@20={case['known_relevant_recall_at_20']:.3f}; "
                f"source={case['source']}"
            )
        else:
            print(
                f"[FAIL] {case['case_id']}: "
                f"{case.get('error', {}).get('message')}"
            )
    aggregate = report.get("aggregate")
    if aggregate is not None:
        print(
            "[AGGREGATE] "
            f"macro={aggregate['known_relevant_recall_at_20_macro']:.3f}; "
            f"micro={aggregate['known_relevant_recall_at_20_micro']:.3f}"
        )
    print(f"Report: {report_dir}")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
