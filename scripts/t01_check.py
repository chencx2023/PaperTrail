"""T01 checks: offline by default; --live explicitly runs up to six paid agent probes."""
import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.pqa.adapter import PaperQAAdapter


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def review_record(bundle):
    return {"run_id": bundle.run_id, "job_status": bundle.job_status,
            "answer_outcome": bundle.answer_outcome,
            "citation_occurrences": len(bundle.citations),
            "unmapped_citations": sum(c.mapping_status != "mapped" for c in bundle.citations),
            "evidence_count": len(bundle.evidence), "warnings": bundle.warnings,
            "human_review": {"key_facts": "pending", "citation_support": "pending", "language": "pending"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--case", default="all", help="all or one ID in eval/t01_cases.json")
    parser.add_argument("--run", type=Path, help="Existing native agent result to export/reload offline")
    args = parser.parse_args()
    cases = json.loads((ROOT / "eval/t01_cases.json").read_text(encoding="utf-8"))["cases"]
    if args.case != "all":
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            parser.error("Unknown case ID.")
    if args.live and args.run:
        parser.error("Use --run for offline export or --live for new probes, not both.")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    # No credentials or model requests are required by the offline test suite.
    test = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, env=env)
    if test.returncode:
        return test.returncode
    directory = ROOT / "data/runs" / ("t01-" + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True)
    report = {"mode": "live" if args.live else "offline", "offline_tests": "passed",
              "t01_acceptance": "pending_real_language_checks_and_review", "runs": []}
    report_path = directory / "report.json"
    save(report_path, report)
    print(f"T01 report: {directory}", flush=True)
    if not args.live:
        candidates = sorted((ROOT / "data/runs").glob("agent-*/session.json"), key=lambda p: p.stat().st_mtime)
        existing = args.run if args.run else (candidates[-1].parent if candidates else None)
        if existing:
            bundle = PaperQAAdapter.export_run(existing, ROOT / "data/indexes/smoke")
            report["runs"].append(review_record(bundle))
            save(report_path, report)
            print(f"[RELOADED] {existing / 'answer_bundle.json'}")
        print("[OFFLINE PASS] No model API calls. Real bilingual probes remain pending.")
        return 0
    from check_models import load_config
    from paperqa_agent import cached_papers
    papers = cached_papers(load_config("all"))
    corpus = "\n".join(t.text for p in papers for t in p["texts"]).casefold()
    if "papertrail-t01-private-benchmark" in corpus:
        raise ValueError("Missing-evidence fixture unexpectedly exists in the corpus.")
    print(f"[LIVE] {len(cases)} paid agent runs; sequential, 8 steps and 300 seconds per run.", flush=True)
    try:
        for case in cases:
            run_id = "agent-" + case["id"] + "-" + uuid.uuid4().hex[:8]
            entry = {"case_id": case["id"], "pair": case["pair"], "run_id": run_id, "status": "running"}
            report["runs"].append(entry)
            save(report_path, report)
            completed = subprocess.run([sys.executable, "scripts/paperqa_agent.py", "--live",
                                        "--question", case["question"], "--run-id", run_id], cwd=ROOT, env=env)
            entry["exit_code"] = completed.returncode
            output = ROOT / "data/runs" / run_id
            if (output / "run.json").exists():
                bundle = PaperQAAdapter.export_run(output)
                entry.update(review_record(bundle))
                entry["status"] = "finished"
            else:
                entry["status"] = "failed_before_run"
            save(report_path, report)
            if entry.get("job_status") != "succeeded" or entry.get("answer_outcome") == "truncated":
                print("[STOP] Failed or truncated probe; remaining paid runs were not started.")
                return 2
        report["t01_acceptance"] = ("pending_human_language_and_evidence_review" if len(cases) == 6
                                    else "partial_cases_run_remaining_pairs_pending")
        print("[RUNS SAVED] Review the three language pairs and missing-evidence answers before marking T01 accepted.")
        return 0
    finally:
        save(report_path, report)


if __name__ == "__main__":
    raise SystemExit(main())
