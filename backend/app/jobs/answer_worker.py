"""Execute one scoped answer and publish only through its active Job attempt."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from paperqa import Settings
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from backend.app.evidence.persistence import publish_staged_answer, stage_answer_run
from backend.app.evidence.service import build_scoped_answer_bundle
from backend.app.jobs.claim import ClaimedJob, timestamp
from backend.app.jobs.index_worker import _runtime_settings
from backend.app.jobs.ownership import LostClaim, finish_active_attempt
from backend.app.jobs.worker import maintain_attempt_lease, refresh_lease
from backend.app.models import AnswerSubmission
from backend.app.qa.scope import prepare_answer_scope
from backend.app.qa.service import execute_scoped_answer
from backend.app.qa.submission import get_answer_request


@dataclass(frozen=True, slots=True)
class ExecutedAnswer:
    job_id: str
    attempt_id: str
    status: str
    answer_run_id: str
    artifact_ref: str | None


def execute_claimed_answer(
    engine: Engine, claimed: ClaimedJob, *, data_root: Path,
    settings: Settings | None = None, answer_executor=None,
    lease_seconds: float = 30.0, heartbeat_interval_seconds: float = 5.0,
) -> ExecutedAnswer:
    if claimed.kind != "answer":
        raise ValueError("Only an answer Job can use the answer executor.")
    try:
        if not refresh_lease(
            engine, claimed.attempt_id, claimed.worker_id,
            lease_seconds=lease_seconds,
        ):
            raise LostClaim("Answer attempt is no longer active.")
        request = get_answer_request(engine, claimed.job_id)
        if request is None:
            raise RuntimeError("Queued answer request is unavailable.")
        configured = settings or _runtime_settings()
        executor = answer_executor or execute_scoped_answer

        async def answer_once():
            scope = await prepare_answer_scope(
                engine, request.revision_id, request.index_artifact_id,
                data_root=data_root, settings=configured,
            )
            async with asyncio.timeout(configured.agent.timeout + 120):
                execution = await executor(scope, request.question)
            return scope, execution

        with maintain_attempt_lease(
            engine, claimed.attempt_id, claimed.worker_id,
            lease_seconds=lease_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        ):
            scope, execution = asyncio.run(answer_once())
            if execution.question != request.question:
                raise ValueError("Answer execution question differs from the submitted question.")
            bundle = build_scoped_answer_bundle(claimed.resource_id, scope, execution)
            staged = stage_answer_run(
                engine, scope, execution, bundle, data_root=data_root,
            )
        status = "succeeded" if bundle.job_status == "succeeded" else "failed"

        def publish(session: Session) -> None:
            submission = session.get(AnswerSubmission, claimed.resource_id)
            if (
                submission is None or submission.job_id != claimed.job_id
                or submission.attempt_id != claimed.attempt_id
            ):
                raise RuntimeError("Answer submission no longer belongs to this attempt.")
            publish_staged_answer(session, staged, job_attempt_id=claimed.attempt_id)

        finish_active_attempt(
            engine, job_id=claimed.job_id, attempt_id=claimed.attempt_id,
            worker_id=claimed.worker_id, status=status,
            finished_at=timestamp(datetime.now(timezone.utc)),
            artifact_ref=staged.saved.artifact_ref,
            result_resource_id=claimed.resource_id if status == "succeeded" else None,
            publish=publish,
        )
        return ExecutedAnswer(
            claimed.job_id, claimed.attempt_id, status, claimed.resource_id,
            staged.saved.artifact_ref,
        )
    except LostClaim:
        raise
    except Exception as error:
        finish_active_attempt(
            engine, job_id=claimed.job_id, attempt_id=claimed.attempt_id,
            worker_id=claimed.worker_id, status="failed",
            finished_at=timestamp(datetime.now(timezone.utc)),
            error_kind=type(error).__name__, error_message=str(error),
        )
        raise
