"""Frozen development cases and reproducible T02 search scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable
from uuid import uuid4

from backend.app.search.client import ArxivClient
from backend.app.search.service import (
    InvalidSearchQuery,
    InvalidSearchSnapshot,
    build_arxiv_query,
    replay_search_run,
    run_search,
)


SEARCH_EVAL_SCHEMA_VERSION = 1
SEARCH_RESULT_LIMIT = 20
_CASE_ID = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*\Z")
_UNVERSIONED_ARXIV_ID = re.compile(
    r"(?:\d{4}\.\d{4,5}|[A-Za-z0-9][A-Za-z0-9.-]*/\d{7})\Z"
)


class InvalidSearchEvalCase(ValueError):
    """Raised when a frozen case or its matching run is not trustworthy."""


@dataclass(frozen=True, slots=True)
class KnownRelevantPaper:
    arxiv_id: str
    title: str
    source_url: str


@dataclass(frozen=True, slots=True)
class SearchEvalCase:
    schema_version: int
    case_id: str
    split: str
    status: str
    frozen_at: str
    question_zh: str
    query: str
    relevance_definition: str
    known_relevant: tuple[KnownRelevantPaper, ...]
    selection_note: str


@dataclass(frozen=True, slots=True)
class KnownRelevantHit:
    arxiv_id: str
    title: str
    display_rank: int
    returned_title: str


@dataclass(frozen=True, slots=True)
class SearchCaseScore:
    case_id: str
    query: str
    run_id: str
    run_dir: Path
    candidate_count: int
    known_count: int
    hit_count: int
    hits: tuple[KnownRelevantHit, ...]
    missing_arxiv_ids: tuple[str, ...]
    known_relevant_recall_at_20: float


@dataclass(frozen=True, slots=True)
class SearchAggregateScore:
    case_count: int
    known_count: int
    hit_count: int
    macro_recall_at_20: float
    micro_recall_at_20: float


def run_frozen_search_evaluation(
    *,
    cases_path: Path,
    output_root: Path,
    report_root: Path,
    client: ArxivClient,
    report_id: str | None = None,
    clock: Callable[[], datetime] | None = None,
    use_cache: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Snapshot cases, run them sequentially, and write an auditable report."""

    resolved_report_id = report_id or f"t02-search-{uuid4().hex[:12]}"
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", resolved_report_id) is None:
        raise InvalidSearchEvalCase("report_id contains unsupported characters.")

    cases = load_search_cases(cases_path)
    try:
        case_bytes = cases_path.read_bytes()
    except OSError as error:
        raise InvalidSearchEvalCase(f"Cannot snapshot search cases: {cases_path}") from error

    report_dir = report_root / resolved_report_id
    try:
        report_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise InvalidSearchEvalCase(f"Report already exists: {report_dir}") from error
    snapshot_path = report_dir / "cases.snapshot.jsonl"
    snapshot_path.write_bytes(case_bytes)
    snapshot_digest = hashlib.sha256(case_bytes).hexdigest()

    current_time = clock or (lambda: datetime.now(timezone.utc))
    created_at = _format_time(current_time())
    case_results: list[dict[str, Any]] = []
    scores: list[SearchCaseScore] = []
    for case in cases:
        run = run_search(
            case.query,
            client=client,
            output_root=output_root,
            clock=current_time,
            use_cache=use_cache,
        )
        source = "cache" if run.cache_hit else run.snapshot_source
        if run.status != "success":
            case_results.append(
                {
                    "case_id": case.case_id,
                    "question_zh": case.question_zh,
                    "query": case.query,
                    "status": "failed",
                    "run_id": run.run_id,
                    "run_dir": str(run.run_dir.resolve()),
                    "source": source,
                    "fetched_at": run.fetched_at,
                    "error": {
                        "kind": run.error_kind,
                        "message": run.error_message,
                    },
                }
            )
            continue

        try:
            score = score_search_run(case, run.run_dir)
        except InvalidSearchEvalCase as error:
            case_results.append(
                {
                    "case_id": case.case_id,
                    "question_zh": case.question_zh,
                    "query": case.query,
                    "status": "score_failed",
                    "run_id": run.run_id,
                    "run_dir": str(run.run_dir.resolve()),
                    "source": source,
                    "fetched_at": run.fetched_at,
                    "error": {"kind": "invalid_score_input", "message": str(error)},
                }
            )
            continue

        scores.append(score)
        case_results.append(_score_data(case, score, source, run.fetched_at))

    aggregate = aggregate_search_scores(scores) if scores else None
    completed = len(scores) == len(cases)
    report: dict[str, Any] = {
        "schema_version": SEARCH_EVAL_SCHEMA_VERSION,
        "report_id": resolved_report_id,
        "status": "completed" if completed else "partial",
        "created_at": created_at,
        "metric_scope": (
            "Known-relevant-paper Recall@20 over a pre-frozen seed set; "
            "not full-domain recall."
        ),
        "cases_snapshot": {
            "path": snapshot_path.name,
            "sha256": snapshot_digest,
            "byte_count": len(case_bytes),
            "case_count": len(cases),
        },
        "aggregate": _aggregate_data(aggregate) if aggregate is not None else None,
        "cases": case_results,
        "manual_precision": {
            "status": "pending",
            "reason": (
                "Returned papers must be judged from title and abstract using each "
                "case's relevance definition; seed membership is not a relevance label."
            ),
        },
    }
    _write_json(report_dir / "report.json", report)
    (report_dir / "report.md").write_text(
        _render_markdown(report),
        encoding="utf-8",
    )
    return report, report_dir


def apply_precision_judgments(
    *,
    report_dir: Path,
    judgments_path: Path,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Validate manual top-10 labels against saved candidates and score them."""

    report_path = report_dir / "report.json"
    report = _read_json_object(report_path, "search evaluation report")
    manual_precision = report.get("manual_precision")
    if not isinstance(manual_precision, dict) or manual_precision.get("status") != "pending":
        raise InvalidSearchEvalCase("The report is not awaiting precision judgments.")

    try:
        judgment_bytes = judgments_path.read_bytes()
    except OSError as error:
        raise InvalidSearchEvalCase(
            f"Cannot read precision judgments: {judgments_path}"
        ) from error
    judgments = _parse_precision_judgments(judgment_bytes)

    successful_cases = [
        case for case in report.get("cases", []) if case.get("status") == "success"
    ]
    if not successful_cases:
        raise InvalidSearchEvalCase("The report has no successful cases to annotate.")

    expected: dict[tuple[str, int], tuple[str, str]] = {}
    for case in successful_cases:
        case_id = case.get("case_id")
        run_id = case.get("run_id")
        run_dir_value = case.get("run_dir")
        if not all(isinstance(value, str) and value for value in (case_id, run_id, run_dir_value)):
            raise InvalidSearchEvalCase("The report contains incomplete case metadata.")
        try:
            replay = replay_search_run(Path(run_dir_value))
        except InvalidSearchSnapshot as error:
            raise InvalidSearchEvalCase(
                f"Cannot replay candidates for {case_id}: {error}"
            ) from error
        if replay.source_run_id != run_id or not replay.candidates_match:
            raise InvalidSearchEvalCase(
                f"Saved candidates for {case_id} no longer match the report."
            )
        for candidate in replay.feed.candidates[:10]:
            expected[(case_id, candidate.display_rank)] = (
                run_id,
                candidate.arxiv_id,
            )

    seen: dict[tuple[str, int], dict[str, Any]] = {}
    for judgment in judgments:
        key = (judgment["case_id"], judgment["display_rank"])
        if key in seen:
            raise InvalidSearchEvalCase(
                f"Duplicate precision judgment for {key[0]} rank {key[1]}."
            )
        expected_identity = expected.get(key)
        if expected_identity is None:
            raise InvalidSearchEvalCase(
                f"Unexpected precision judgment for {key[0]} rank {key[1]}."
            )
        expected_run_id, expected_arxiv_id = expected_identity
        if judgment["run_id"] != expected_run_id:
            raise InvalidSearchEvalCase(
                f"Judgment run_id does not match {key[0]} rank {key[1]}."
            )
        if judgment["arxiv_id"] != expected_arxiv_id:
            raise InvalidSearchEvalCase(
                f"Judgment arXiv ID does not match {key[0]} rank {key[1]}."
            )
        seen[key] = judgment

    missing = sorted(set(expected) - set(seen))
    if missing:
        case_id, rank = missing[0]
        raise InvalidSearchEvalCase(
            f"Missing precision judgment for {case_id} rank {rank}."
        )

    case_scores: list[dict[str, Any]] = []
    total_relevant = 0
    total_judged = 0
    for case in successful_cases:
        case_id = case["case_id"]
        case_judgments = [
            seen[(case_id, rank)]
            for rank in sorted(
                rank for current_case, rank in expected if current_case == case_id
            )
        ]
        relevant_count = sum(item["relevant"] for item in case_judgments)
        judged_count = len(case_judgments)
        case_score = {
            "case_id": case_id,
            "judged_count": judged_count,
            "relevant_count": relevant_count,
            "precision_at_10": relevant_count / 10,
            "returned_precision_at_10": (
                relevant_count / judged_count if judged_count else None
            ),
        }
        case["manual_precision"] = case_score
        case_scores.append(case_score)
        total_relevant += relevant_count
        total_judged += judged_count

    snapshot_path = report_dir / "precision_judgments.snapshot.jsonl"
    if snapshot_path.exists():
        raise InvalidSearchEvalCase(f"Judgment snapshot already exists: {snapshot_path}")
    snapshot_path.write_bytes(judgment_bytes)
    current_time = clock or (lambda: datetime.now(timezone.utc))
    case_count = len(case_scores)
    report["manual_precision"] = {
        "status": "completed",
        "annotated_at": _format_time(current_time()),
        "judgment_snapshot": {
            "path": snapshot_path.name,
            "sha256": hashlib.sha256(judgment_bytes).hexdigest(),
            "byte_count": len(judgment_bytes),
            "judgment_count": len(judgments),
        },
        "case_count": case_count,
        "judged_count": total_judged,
        "relevant_count": total_relevant,
        "precision_at_10": total_relevant / (10 * case_count),
        "returned_precision_at_10": (
            total_relevant / total_judged if total_judged else None
        ),
        "precision_at_10_macro": sum(
            score["precision_at_10"] for score in case_scores
        )
        / case_count,
        "returned_precision_at_10_macro": _mean_non_null(
            score["returned_precision_at_10"] for score in case_scores
        ),
        "case_scores": case_scores,
    }
    _write_json(report_path, report)
    (report_dir / "report.md").write_text(
        _render_markdown(report),
        encoding="utf-8",
    )
    return report


def load_search_cases(path: Path) -> tuple[SearchEvalCase, ...]:
    """Load and validate line-delimited, pre-frozen search cases."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise InvalidSearchEvalCase(f"Cannot read search cases: {path}") from error

    cases: list[SearchEvalCase] = []
    seen_case_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise InvalidSearchEvalCase(
                f"Line {line_number} is not valid JSON."
            ) from error
        if not isinstance(raw, dict):
            raise InvalidSearchEvalCase(
                f"Line {line_number} must contain one JSON object."
            )
        case = _parse_case(raw, line_number=line_number)
        if case.case_id in seen_case_ids:
            raise InvalidSearchEvalCase(f"Duplicate case_id: {case.case_id}")
        seen_case_ids.add(case.case_id)
        cases.append(case)

    if not cases:
        raise InvalidSearchEvalCase("The search case file is empty.")
    return tuple(cases)


def score_search_run(case: SearchEvalCase, run_dir: Path) -> SearchCaseScore:
    """Score one saved run after verifying its immutable request and snapshot."""

    try:
        replay = replay_search_run(run_dir)
    except InvalidSearchSnapshot as error:
        raise InvalidSearchEvalCase(
            f"Run for {case.case_id} cannot be replayed: {error}"
        ) from error
    if not replay.candidates_match:
        raise InvalidSearchEvalCase(
            f"Saved candidates for {case.case_id} do not match the Atom snapshot."
        )

    request = _read_json_object(run_dir / "request.json", "run request")
    run_query = request.get("normalized_query_text")
    if run_query != case.query:
        raise InvalidSearchEvalCase(
            f"Run query {run_query!r} does not match case query {case.query!r}."
        )

    candidates = replay.feed.candidates
    if len(candidates) > SEARCH_RESULT_LIMIT:
        raise InvalidSearchEvalCase(
            f"Run returned {len(candidates)} candidates; Recall@20 requires at most 20."
        )

    ranks = {candidate.arxiv_id: candidate for candidate in candidates}
    hits: list[KnownRelevantHit] = []
    missing: list[str] = []
    for paper in case.known_relevant:
        candidate = ranks.get(paper.arxiv_id)
        if candidate is None:
            missing.append(paper.arxiv_id)
            continue
        hits.append(
            KnownRelevantHit(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                display_rank=candidate.display_rank,
                returned_title=candidate.title,
            )
        )

    hits.sort(key=lambda hit: hit.display_rank)
    known_count = len(case.known_relevant)
    return SearchCaseScore(
        case_id=case.case_id,
        query=case.query,
        run_id=replay.source_run_id,
        run_dir=replay.source_run_dir,
        candidate_count=len(candidates),
        known_count=known_count,
        hit_count=len(hits),
        hits=tuple(hits),
        missing_arxiv_ids=tuple(missing),
        known_relevant_recall_at_20=len(hits) / known_count,
    )


def aggregate_search_scores(
    scores: Iterable[SearchCaseScore],
) -> SearchAggregateScore:
    score_tuple = tuple(scores)
    if not score_tuple:
        raise InvalidSearchEvalCase("At least one successful score is required.")
    known_count = sum(score.known_count for score in score_tuple)
    hit_count = sum(score.hit_count for score in score_tuple)
    return SearchAggregateScore(
        case_count=len(score_tuple),
        known_count=known_count,
        hit_count=hit_count,
        macro_recall_at_20=sum(
            score.known_relevant_recall_at_20 for score in score_tuple
        )
        / len(score_tuple),
        micro_recall_at_20=hit_count / known_count,
    )


def _parse_case(raw: dict[str, Any], *, line_number: int) -> SearchEvalCase:
    prefix = f"Line {line_number}"
    if raw.get("schema_version") != SEARCH_EVAL_SCHEMA_VERSION:
        raise InvalidSearchEvalCase(f"{prefix} has an unsupported schema_version.")

    case_id = _required_text(raw, "case_id", prefix)
    if _CASE_ID.fullmatch(case_id) is None:
        raise InvalidSearchEvalCase(f"{prefix} has an invalid case_id.")
    if raw.get("split") != "development":
        raise InvalidSearchEvalCase(f"{prefix} must use the development split.")
    if raw.get("status") != "frozen":
        raise InvalidSearchEvalCase(f"{prefix} must have frozen status.")

    frozen_at = _required_text(raw, "frozen_at", prefix)
    try:
        parsed_time = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise InvalidSearchEvalCase(f"{prefix} has an invalid frozen_at.") from error
    if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
        raise InvalidSearchEvalCase(f"{prefix} frozen_at must include a timezone.")

    query = _required_text(raw, "query", prefix)
    try:
        normalized_query = build_arxiv_query(query).normalized_query_text
    except InvalidSearchQuery as error:
        raise InvalidSearchEvalCase(f"{prefix} has an invalid query: {error}") from error
    if query != normalized_query:
        raise InvalidSearchEvalCase(f"{prefix} query must already be normalized.")

    raw_papers = raw.get("known_relevant")
    if not isinstance(raw_papers, list) or not 3 <= len(raw_papers) <= 5:
        raise InvalidSearchEvalCase(
            f"{prefix} known_relevant must contain 3 to 5 papers."
        )
    papers: list[KnownRelevantPaper] = []
    seen_ids: set[str] = set()
    for paper_number, raw_paper in enumerate(raw_papers, start=1):
        paper_prefix = f"{prefix} paper {paper_number}"
        if not isinstance(raw_paper, dict):
            raise InvalidSearchEvalCase(f"{paper_prefix} must be an object.")
        arxiv_id = _required_text(raw_paper, "arxiv_id", paper_prefix)
        if _UNVERSIONED_ARXIV_ID.fullmatch(arxiv_id) is None:
            raise InvalidSearchEvalCase(
                f"{paper_prefix} must use a valid unversioned arXiv ID."
            )
        if arxiv_id in seen_ids:
            raise InvalidSearchEvalCase(f"{prefix} repeats arXiv ID {arxiv_id}.")
        seen_ids.add(arxiv_id)
        source_url = _required_text(raw_paper, "source_url", paper_prefix)
        expected_url = f"https://arxiv.org/abs/{arxiv_id}"
        if source_url != expected_url:
            raise InvalidSearchEvalCase(
                f"{paper_prefix} source_url must be {expected_url}."
            )
        papers.append(
            KnownRelevantPaper(
                arxiv_id=arxiv_id,
                title=_required_text(raw_paper, "title", paper_prefix),
                source_url=source_url,
            )
        )

    return SearchEvalCase(
        schema_version=SEARCH_EVAL_SCHEMA_VERSION,
        case_id=case_id,
        split="development",
        status="frozen",
        frozen_at=frozen_at,
        question_zh=_required_text(raw, "question_zh", prefix),
        query=query,
        relevance_definition=_required_text(raw, "relevance_definition", prefix),
        known_relevant=tuple(papers),
        selection_note=_required_text(raw, "selection_note", prefix),
    )


def _required_text(value: dict[str, Any], key: str, prefix: str) -> str:
    text = value.get(key)
    if not isinstance(text, str) or not text.strip():
        raise InvalidSearchEvalCase(f"{prefix} is missing {key}.")
    return text


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidSearchEvalCase(f"Cannot read {description}: {path}") from error
    if not isinstance(value, dict):
        raise InvalidSearchEvalCase(f"The {description} must be a JSON object.")
    return value


def _parse_precision_judgments(payload: bytes) -> tuple[dict[str, Any], ...]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise InvalidSearchEvalCase("Precision judgments are not UTF-8.") from error

    judgments: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise InvalidSearchEvalCase(
                f"Precision judgment line {line_number} is not valid JSON."
            ) from error
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise InvalidSearchEvalCase(
                f"Precision judgment line {line_number} has an invalid schema."
            )
        prefix = f"Precision judgment line {line_number}"
        case_id = _required_text(raw, "case_id", prefix)
        run_id = _required_text(raw, "run_id", prefix)
        arxiv_id = _required_text(raw, "arxiv_id", prefix)
        rank = raw.get("display_rank")
        relevant = raw.get("relevant")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= 10:
            raise InvalidSearchEvalCase(f"{prefix} has an invalid display_rank.")
        if not isinstance(relevant, bool):
            raise InvalidSearchEvalCase(f"{prefix} must label relevant as boolean.")
        judgments.append(
            {
                "case_id": case_id,
                "run_id": run_id,
                "display_rank": rank,
                "arxiv_id": arxiv_id,
                "relevant": relevant,
                "rationale": _required_text(raw, "rationale", prefix),
            }
        )
    return tuple(judgments)


def _score_data(
    case: SearchEvalCase,
    score: SearchCaseScore,
    source: str | None,
    fetched_at: str | None,
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "question_zh": case.question_zh,
        "query": case.query,
        "relevance_definition": case.relevance_definition,
        "status": "success",
        "run_id": score.run_id,
        "run_dir": str(score.run_dir),
        "source": source,
        "fetched_at": fetched_at,
        "candidate_count": score.candidate_count,
        "known_count": score.known_count,
        "hit_count": score.hit_count,
        "known_relevant_recall_at_20": score.known_relevant_recall_at_20,
        "hits": [
            {
                "arxiv_id": hit.arxiv_id,
                "seed_title": hit.title,
                "returned_title": hit.returned_title,
                "display_rank": hit.display_rank,
            }
            for hit in score.hits
        ],
        "missing_arxiv_ids": list(score.missing_arxiv_ids),
    }


def _aggregate_data(score: SearchAggregateScore) -> dict[str, Any]:
    return {
        "successful_case_count": score.case_count,
        "known_count": score.known_count,
        "hit_count": score.hit_count,
        "known_relevant_recall_at_20_macro": score.macro_recall_at_20,
        "known_relevant_recall_at_20_micro": score.micro_recall_at_20,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# T02 搜索开发样本报告",
        "",
        f"状态：`{report['status']}`。本报告只计算预先冻结种子的已知相关论文 Recall@20，不代表完整领域召回率。",
        "",
        f"案例快照：`{report['cases_snapshot']['path']}`，SHA-256 `{report['cases_snapshot']['sha256']}`。",
        "",
        "| 案例 | 查询 | 返回数 | 命中 | 已知相关论文 Recall@20 | 来源 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for case in report["cases"]:
        if case["status"] == "success":
            recall = f"{case['known_relevant_recall_at_20']:.3f}"
            returned = str(case["candidate_count"])
            hits = f"{case['hit_count']}/{case['known_count']}"
        else:
            recall = "N/A"
            returned = "N/A"
            hits = "N/A"
        lines.append(
            f"| `{case['case_id']}` | `{case['query']}` | {returned} | "
            f"{hits} | {recall} | `{case.get('source') or 'none'}` |"
        )

    aggregate = report.get("aggregate")
    lines.extend(["", "## 汇总", ""])
    if aggregate is None:
        lines.append("没有可计分的成功案例。")
    else:
        lines.append(
            "成功案例 "
            f"{aggregate['successful_case_count']} 个，共命中 "
            f"{aggregate['hit_count']}/{aggregate['known_count']} 篇已知相关论文；"
            f"macro Recall@20 = {aggregate['known_relevant_recall_at_20_macro']:.3f}，"
            f"micro Recall@20 = {aggregate['known_relevant_recall_at_20_micro']:.3f}。"
        )

    lines.extend(["", "## 逐题命中", ""])
    for case in report["cases"]:
        lines.append(f"### {case['case_id']}")
        lines.append("")
        if case["status"] != "success":
            error = case.get("error") or {}
            lines.append(
                f"运行失败：`{error.get('kind')}`，{error.get('message')}"
            )
            lines.append("")
            continue
        if case["hits"]:
            for hit in case["hits"]:
                lines.append(
                    f"- 命中 `{hit['arxiv_id']}`，展示排名 {hit['display_rank']}："
                    f"{hit['returned_title']}"
                )
        else:
            lines.append("- 没有命中预先冻结的已知相关论文。")
        missing = ", ".join(case["missing_arxiv_ids"]) or "无"
        lines.append(f"- 未命中 ID：{missing}")
        lines.append(f"- 运行目录：`{case['run_dir']}`")
        lines.append("")

    manual_precision = report.get("manual_precision") or {}
    if manual_precision.get("status") == "completed":
        lines.extend(
            [
                "## 人工 Precision@10",
                "",
                "| 案例 | 相关/已判定 | Precision@10 | 已返回结果精度 |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for score in manual_precision["case_scores"]:
            returned_precision = score["returned_precision_at_10"]
            returned_text = (
                f"{returned_precision:.3f}"
                if returned_precision is not None
                else "N/A"
            )
            lines.append(
                f"| `{score['case_id']}` | {score['relevant_count']}/"
                f"{score['judged_count']} | {score['precision_at_10']:.3f} | "
                f"{returned_text} |"
            )
        aggregate_returned = manual_precision["returned_precision_at_10"]
        aggregate_returned_text = (
            f"{aggregate_returned:.3f}"
            if aggregate_returned is not None
            else "N/A"
        )
        lines.extend(
            [
                "",
                f"四题合计 Precision@10 = {manual_precision['precision_at_10']:.3f}；"
                f"已返回结果精度 = {aggregate_returned_text}。"
                "每条判定依据标题和摘要，并保存在独立判断快照中。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 尚未计算的指标",
                "",
                "Precision@10 需要按各题相关性定义阅读真实返回结果的标题和摘要后人工标注。种子之外的论文可能仍然相关，不能按是否位于种子集合自动判错。",
                "",
            ]
        )
    return "\n".join(lines)


def _format_time(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise InvalidSearchEvalCase("Clock values must include a timezone.")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _mean_non_null(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
