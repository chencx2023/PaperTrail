"""Preview or run one scoped PaperQA answer against an existing frozen index."""

import argparse
import asyncio
from contextlib import closing
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from check_models import load_config
from model_config import build_settings
from backend.app.db import open_database
from backend.app.evidence.persistence import load_answer_run, save_answer_run
from backend.app.evidence.service import build_scoped_answer_bundle
from backend.app.qa.scope import prepare_answer_scope
from backend.app.qa.service import execute_scoped_answer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-database", required=True, type=Path)
    parser.add_argument("--revision-id", required=True)
    parser.add_argument("--index-artifact-id", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "runs")
    parser.add_argument("--max-timesteps", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.max_timesteps <= 8 or not 30 <= args.timeout_seconds <= 300:
        parser.error("Use 1..8 agent steps and 30..300 timeout seconds.")
    return args


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _backup_database(source: Path, target: Path) -> None:
    with closing(sqlite3.connect(source)) as original, closing(sqlite3.connect(target)) as copy:
        original.backup(copy)


async def _prepare(engine, args, settings):
    return await prepare_answer_scope(
        engine, args.revision_id, args.index_artifact_id,
        data_root=args.data_root, settings=settings,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.source_database.is_file():
        print("[FAIL] Source database does not exist.")
        return 1
    if not args.question.strip():
        print("[FAIL] A nonempty question is required.")
        return 1

    try:
        settings = build_settings(load_config("all"))
        settings.agent.max_timesteps = args.max_timesteps
        settings.agent.timeout = args.timeout_seconds
        outer_timeout_seconds = args.timeout_seconds + 120
        if not args.live:
            engine = open_database(args.source_database, create_schema=False)
            try:
                scope = asyncio.run(_prepare(engine, args, settings))
            finally:
                engine.dispose()
            print(json.dumps({
                "mode": "preview",
                "question": args.question,
                "revision_id": scope.revision_id,
                "index_artifact_id": scope.index_artifact_id,
                "documents": [document.versioned_id for document in scope.documents],
                "models": {
                    "agent": scope.settings.agent.agent_llm,
                    "summary": scope.settings.summary_llm,
                    "answer": scope.settings.llm,
                    "embedding": scope.settings.embedding,
                },
                "max_timesteps": args.max_timesteps,
                "timeout_seconds": args.timeout_seconds,
                "outer_timeout_seconds": outer_timeout_seconds,
                "model_calls": 0,
            }, ensure_ascii=False, indent=2))
            return 0

        run_dir = args.output_root / f"t05-answer-{uuid4().hex[:12]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        snapshot_database = run_dir / "papertrail.sqlite"
        _backup_database(args.source_database, snapshot_database)
        report = {
            "source_database": str(args.source_database),
            "database_ref": snapshot_database.name,
            "question": args.question,
            "revision_id": args.revision_id,
            "index_artifact_id": args.index_artifact_id,
            "max_timesteps": args.max_timesteps,
            "agent_timeout_seconds": args.timeout_seconds,
            "outer_timeout_seconds": outer_timeout_seconds,
            "started_at": _now(),
            "status": "running",
        }
        try:
            engine = open_database(snapshot_database)
            try:
                scope = asyncio.run(_prepare(engine, args, settings))

                async def answer_once():
                    async with asyncio.timeout(outer_timeout_seconds):
                        # Upstream exception traces may contain provider request
                        # details; the saved report uses safe status fields.
                        logging.getLogger("paperqa.agents.main").setLevel(logging.CRITICAL)
                        return await execute_scoped_answer(scope, args.question)

                execution = asyncio.run(answer_once())
                answer_run_id = uuid4().hex
                bundle = build_scoped_answer_bundle(answer_run_id, scope, execution)
                saved = save_answer_run(
                    engine, scope, execution, bundle, data_root=args.data_root,
                )
                reopened = load_answer_run(engine, answer_run_id, data_root=args.data_root)
            finally:
                engine.dispose()
            report.update({
                "status": reopened.job_status,
                "persistence_status": "saved",
                "answer_run_id": answer_run_id,
                "artifact_ref": saved.artifact_ref,
                "artifact_sha256": saved.artifact_sha256,
                "execution_status": reopened.job_status,
                "answer_outcome": reopened.answer_outcome,
                "upstream_status": reopened.upstream_status,
                "failure_kind": execution.failure_kind,
                "evidence_count": len(reopened.evidence),
                "citation_statuses": [citation.mapping_status for citation in reopened.citations],
                "usage": reopened.usage,
            })
            print(f"[SAVED] {reopened.job_status}, {reopened.answer_outcome}; "
                  f"{len(reopened.evidence)} evidence items")
            exit_code = 0 if reopened.job_status == "succeeded" else 1
        except Exception as error:
            # Provider exceptions can include request data. Keep the public
            # report to the exception class rather than serializing its text.
            report.update(status="failed", error_kind=type(error).__name__)
            print(f"[FAIL] {type(error).__name__}")
            exit_code = 1
        report["finished_at"] = _now()
        (run_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[REPORT] {run_dir / 'report.json'}")
        return exit_code
    except Exception as error:
        print(f"[FAIL] {type(error).__name__}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
