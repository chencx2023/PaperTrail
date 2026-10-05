"""Compare saved native/merge/RRF orders without search or model requests."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.search.service import InvalidSearchSnapshot, replay_search_run


def comparison_report(run_dir: Path) -> dict:
    replay = replay_search_run(Path(run_dir))
    fusion = replay.fusion
    if fusion is None or not replay.candidates_match:
        raise InvalidSearchSnapshot("A verified multi-query search is required.")
    baseline = [row["arxiv_id"] for row in fusion["queries"][0]["original_order"]]
    generation = fusion.get("generation") or {}
    return {
        "source_run_id": replay.source_run_id, "source_snapshot_sha256": replay.snapshot_sha256,
        "rankings": {"single_query": baseline, "simple_merge": fusion["merge_order"],
                     "rrf": [candidate.arxiv_id for candidate in replay.feed.candidates]},
        "candidate_counts": {"single_query": len(baseline), "multi_query": len(replay.feed.candidates)},
        "budget": {"query_count": len(fusion["queries"]), "http_attempts": fusion["http_attempts"],
                   "cache_hits": fusion["cache_hits"], "search_elapsed_ms": fusion["elapsed_ms"],
                   "model_tokens": generation.get("usage", {}).get("total_tokens"),
                   "model_elapsed_ms": generation.get("elapsed_ms"), "cost_usd": generation.get("cost_usd"),
                   "model_usage_scope": "expansion_generation_only"},
        "quality": None, "partial": fusion["partial"],
        "comparisons": {"single_vs_multi": "整个发现方案，包括额外候选与请求；不能全部归因于 RRF。",
                        "merge_vs_rrf": "相同保存候选池上的排序比较；需人工相关性标注才能判断提升。"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = comparison_report(args.run_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
