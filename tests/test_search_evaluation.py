import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.search.client import ArxivClient, TransportResponse
from backend.app.search.evaluation import (
    InvalidSearchEvalCase,
    apply_precision_judgments,
    aggregate_search_scores,
    load_search_cases,
    run_frozen_search_evaluation,
    score_search_run,
)
from backend.app.search.service import run_search


FIXTURES = Path(__file__).parent / "fixtures" / "arxiv"
AT = datetime(2026, 9, 22, 14, 30, tzinfo=timezone.utc)


class OneResponseTransport:
    def __init__(self) -> None:
        self.requests = []

    def send(self, request):
        self.requests.append(request)
        return TransportResponse(
            status_code=200,
            body=(FIXTURES / "normal_with_versions.xml").read_bytes(),
            headers=(("Content-Type", "application/atom+xml"),),
        )


class SnapshotCheckingTransport(OneResponseTransport):
    def __init__(self, snapshot_path: Path, expected_bytes: bytes) -> None:
        super().__init__()
        self.snapshot_path = snapshot_path
        self.expected_bytes = expected_bytes

    def send(self, request):
        if self.snapshot_path.read_bytes() != self.expected_bytes:
            raise AssertionError("case snapshot was not frozen before the request")
        return super().send(request)


def valid_case(**updates):
    value = {
        "schema_version": 1,
        "case_id": "dev_fixture",
        "split": "development",
        "status": "frozen",
        "frozen_at": "2026-09-22T22:01:40+08:00",
        "question_zh": "哪些论文研究 Agent 记忆？",
        "query": "LLM agent memory",
        "relevance_definition": "主要贡献研究语言模型 Agent 的记忆机制。",
        "known_relevant": [
            {
                "arxiv_id": "2401.00001",
                "title": "Memory for Agents",
                "source_url": "https://arxiv.org/abs/2401.00001",
            },
            {
                "arxiv_id": "cs/9901001",
                "title": "Legacy Identifier Paper",
                "source_url": "https://arxiv.org/abs/cs/9901001",
            },
            {
                "arxiv_id": "9999.99999",
                "title": "Known Missing Paper",
                "source_url": "https://arxiv.org/abs/9999.99999",
            },
        ],
        "selection_note": "Selected before running the query.",
    }
    value.update(updates)
    return value


def write_cases(path: Path, *cases) -> None:
    path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )


class SearchCaseLoadingTest(unittest.TestCase):
    def test_loads_a_frozen_case_with_three_to_five_unique_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            write_cases(path, valid_case())

            cases = load_search_cases(path)

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].case_id, "dev_fixture")
            self.assertEqual(cases[0].known_relevant[1].arxiv_id, "cs/9901001")

    def test_rejects_duplicate_case_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            write_cases(path, valid_case(), valid_case())

            with self.assertRaisesRegex(InvalidSearchEvalCase, "Duplicate case_id"):
                load_search_cases(path)

    def test_rejects_too_few_or_versioned_seed_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            too_few = valid_case(known_relevant=valid_case()["known_relevant"][:2])
            write_cases(path, too_few)
            with self.assertRaisesRegex(InvalidSearchEvalCase, "3 to 5"):
                load_search_cases(path)

            versioned = valid_case()
            versioned["known_relevant"][0]["arxiv_id"] = "2401.00001v2"
            write_cases(path, versioned)
            with self.assertRaisesRegex(InvalidSearchEvalCase, "unversioned"):
                load_search_cases(path)

    def test_rejects_non_normalized_or_invalid_query(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            write_cases(path, valid_case(query=" LLM  agent memory "))

            with self.assertRaisesRegex(InvalidSearchEvalCase, "normalized"):
                load_search_cases(path)


class SearchScoringTest(unittest.TestCase):
    def make_run(self, output_root: Path):
        transport = OneResponseTransport()
        client = ArxivClient(
            transport=transport,
            clock=lambda: AT,
            monotonic=lambda: 0.0,
            sleeper=lambda seconds: None,
        )
        return run_search(
            "LLM agent memory",
            client=client,
            output_root=output_root,
            run_id="search-eval-fixture",
            clock=lambda: AT,
        )

    def test_scores_unversioned_hits_and_aggregates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "cases.jsonl"
            write_cases(path, valid_case())
            case = load_search_cases(path)[0]
            run = self.make_run(root / "runs")

            score = score_search_run(case, run.run_dir)
            aggregate = aggregate_search_scores((score,))

            self.assertEqual(score.hit_count, 2)
            self.assertEqual(score.known_count, 3)
            self.assertAlmostEqual(score.known_relevant_recall_at_20, 2 / 3)
            self.assertEqual(
                [(hit.arxiv_id, hit.display_rank) for hit in score.hits],
                [("2401.00001", 1), ("cs/9901001", 2)],
            )
            self.assertEqual(score.missing_arxiv_ids, ("9999.99999",))
            self.assertAlmostEqual(aggregate.macro_recall_at_20, 2 / 3)
            self.assertAlmostEqual(aggregate.micro_recall_at_20, 2 / 3)

    def test_rejects_a_run_for_a_different_query(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "cases.jsonl"
            write_cases(path, valid_case(query="LLM agent planning"))
            case = load_search_cases(path)[0]
            run = self.make_run(root / "runs")

            with self.assertRaisesRegex(InvalidSearchEvalCase, "does not match"):
                score_search_run(case, run.run_dir)


class SearchEvaluationRunTest(unittest.TestCase):
    def test_snapshots_cases_before_search_and_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.jsonl"
            write_cases(cases_path, valid_case())
            expected_bytes = cases_path.read_bytes()
            report_root = root / "reports"
            snapshot_path = report_root / "t02-fixture" / "cases.snapshot.jsonl"
            transport = SnapshotCheckingTransport(snapshot_path, expected_bytes)
            client = ArxivClient(
                transport=transport,
                clock=lambda: AT,
                monotonic=lambda: 0.0,
                sleeper=lambda seconds: None,
            )

            report, report_dir = run_frozen_search_evaluation(
                cases_path=cases_path,
                output_root=root / "runs",
                report_root=report_root,
                client=client,
                report_id="t02-fixture",
                clock=lambda: AT,
            )

            self.assertEqual(report_dir, report_root / "t02-fixture")
            self.assertEqual(snapshot_path.read_bytes(), expected_bytes)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["aggregate"]["hit_count"], 2)
            self.assertAlmostEqual(
                report["aggregate"]["known_relevant_recall_at_20_micro"],
                2 / 3,
            )
            self.assertEqual(report["cases"][0]["source"], "network")
            self.assertTrue((report_dir / "report.json").is_file())
            self.assertTrue((report_dir / "report.md").is_file())
            self.assertEqual(len(transport.requests), 1)

            judgments_path = root / "judgments.jsonl"
            write_cases(
                judgments_path,
                {
                    "schema_version": 1,
                    "case_id": "dev_fixture",
                    "run_id": report["cases"][0]["run_id"],
                    "display_rank": 1,
                    "arxiv_id": "2401.00001",
                    "relevant": True,
                    "rationale": "The abstract studies agent memory.",
                },
                {
                    "schema_version": 1,
                    "case_id": "dev_fixture",
                    "run_id": report["cases"][0]["run_id"],
                    "display_rank": 2,
                    "arxiv_id": "cs/9901001",
                    "relevant": False,
                    "rationale": "The fixture abstract is outside the case definition.",
                },
            )

            annotated = apply_precision_judgments(
                report_dir=report_dir,
                judgments_path=judgments_path,
                clock=lambda: AT,
            )

            self.assertEqual(annotated["manual_precision"]["status"], "completed")
            self.assertEqual(annotated["manual_precision"]["relevant_count"], 1)
            self.assertEqual(annotated["manual_precision"]["precision_at_10"], 0.1)
            self.assertEqual(
                annotated["manual_precision"]["returned_precision_at_10"], 0.5
            )
            self.assertTrue(
                (report_dir / "precision_judgments.snapshot.jsonl").is_file()
            )


if __name__ == "__main__":
    unittest.main()
