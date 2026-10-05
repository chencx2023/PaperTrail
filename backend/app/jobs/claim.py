"""Atomically reserve a queued Job for one worker and create its attempt."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from backend.app.models import AnswerSubmission, Job, JobAttempt


@dataclass(frozen=True, slots=True)
class ClaimedJob:
    job_id: str
    kind: str
    resource_id: str
    attempt_id: str
    worker_id: str


def timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("A timezone-aware UTC time is required.")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def require_positive_lease(lease_seconds: float) -> None:
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive.")


def claim_next_job(
    engine: Engine,
    *,
    worker_id: str,
    supported_kinds: tuple[str, ...],
    lease_seconds: float = 30.0,
) -> ClaimedJob | None:
    """Claim only registered kinds; leave future work queued until supported."""

    require_positive_lease(lease_seconds)
    if not worker_id.strip():
        raise ValueError("A worker ID is required.")
    kinds = tuple(dict.fromkeys(supported_kinds))
    if not kinds or any(not kind.strip() for kind in kinds):
        raise ValueError("At least one supported Job kind is required.")

    new_attempt_id = uuid4().hex
    current_time = datetime.now(timezone.utc)
    now = timestamp(current_time)
    lease_until = timestamp(current_time + timedelta(seconds=lease_seconds))
    next_id = (
        select(Job.id)
        .where(Job.kind.in_(kinds), Job.status == "queued")
        .order_by(Job.created_at, Job.id)
        .limit(1)
        .scalar_subquery()
    )
    with Session(engine) as session:
        claimed = session.execute(
            update(Job)
            .where(
                Job.id == next_id,
                Job.kind.in_(kinds),
                Job.status == "queued",
            )
            .values(
                status="running",
                active_attempt_id=func.coalesce(Job.active_attempt_id, new_attempt_id),
                updated_at=now,
            )
            .returning(Job.id, Job.kind, Job.resource_id, Job.active_attempt_id)
        ).first()
        if claimed is None:
            session.rollback()
            return None
        attempt_no = (
            session.scalar(
                select(func.max(JobAttempt.attempt_no)).where(
                    JobAttempt.job_id == claimed.id
                )
            )
            or 0
        ) + 1
        session.add(
            JobAttempt(
                id=claimed.active_attempt_id,
                job_id=claimed.id,
                attempt_no=attempt_no,
                worker_id=worker_id,
                status="running",
                started_at=now,
                heartbeat_at=now,
                lease_until=lease_until,
            )
        )
        if claimed.kind == "answer":
            linked = session.execute(
                update(AnswerSubmission)
                .where(
                    AnswerSubmission.answer_run_id == claimed.resource_id,
                    AnswerSubmission.job_id == claimed.id,
                    AnswerSubmission.attempt_id.is_(None),
                )
                .values(attempt_id=claimed.active_attempt_id)
                .returning(AnswerSubmission.answer_run_id)
            ).scalar_one_or_none()
            if linked is None:
                session.rollback()
                raise RuntimeError("Queued answer has no unclaimed submission.")
        session.commit()
        return ClaimedJob(
            job_id=claimed.id,
            kind=claimed.kind,
            resource_id=claimed.resource_id,
            attempt_id=claimed.active_attempt_id,
            worker_id=worker_id,
        )
