import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.search.evaluation import (
    InvalidSearchEvalCase,
    apply_precision_judgments,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate manual top-10 relevance labels and update a T02 report."
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = apply_precision_judgments(
            report_dir=args.report_dir,
            judgments_path=args.judgments,
        )
    except (InvalidSearchEvalCase, OSError) as error:
        print(f"[FAIL] precision: {error}")
        return 1

    precision = report["manual_precision"]
    for case in precision["case_scores"]:
        print(
            f"[PRECISION] {case['case_id']}: "
            f"{case['relevant_count']}/{case['judged_count']}; "
            f"P@10={case['precision_at_10']:.3f}"
        )
    returned = precision["returned_precision_at_10"]
    returned_text = f"{returned:.3f}" if returned is not None else "N/A"
    print(
        f"[AGGREGATE] P@10={precision['precision_at_10']:.3f}; "
        f"returned={returned_text}"
    )
    print(f"Report: {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
