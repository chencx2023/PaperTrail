"""Publish a Job result only while its worker still owns the active attempt."""

from pathlib import PurePosixPath
from typing import Callable

from sqlalchemy import update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from backend.app.models import Job, JobAttempt, JobAttemptResult


class LostClaim(RuntimeError):
    """The attempt stopped being the active execution before publication."""


def _check_artifact_ref(artifact_ref: str | None) -> None:
    if artifact_ref is None:
        return
    path = PurePosixPath(artifact_ref)
    if (
        not artifact_ref
        or "\\" in artifact_ref
        or ":" in artifact_ref
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != artifact_ref
    ):
        raise ValueError("Artifact reference must be a relative normalized path.")


def finish_active_attempt(
    engine: Engine,
    *,
    job_id: str,
    attempt_id: str,
    worker_id: str,
    status: str,
    finished_at: str,
    artifact_ref: str | None = None,
    error_kind: str | None = None,
    error_message: str | None = None,
    result_resource_id: str | None = None,
    publish: Callable[[Session], None] | None = None,
) -> None:
    """Fence status, result binding and domain writes in one short transaction.

    The publisher may perform database writes through the supplied Session.
    Network, parsing and file I/O must complete before calling this function.
    """

    if status not in ("succeeded", "failed"):
        raise ValueError("Only succeeded or failed attempts may finish.")
    if status == "succeeded" and not result_resource_id:
        raise ValueError("A successful attempt requires a result resource ID.")
    if status == "failed" and result_resource_id is not None:
        raise ValueError("A failed attempt cannot publish a result resource.")
    _check_artifact_ref(artifact_ref)

    with Session(engine) as session:
        owned_job = session.execute(
            update(Job)
            .where(
                Job.id == job_id,
                Job.status == "running",
                Job.active_attempt_id == attempt_id,
            )
            .values(status=status, updated_at=finished_at)
            .returning(Job.id)
        ).scalar_one_or_none()
        if owned_job is None:
            raise LostClaim("This attempt is no longer the active worker.")

        owned_attempt = session.execute(
            update(JobAttempt)
            .where(
                JobAttempt.id == attempt_id,
                JobAttempt.job_id == job_id,
                JobAttempt.worker_id == worker_id,
                JobAttempt.status == "running",
            )
            .values(
                status=status,
                finished_at=finished_at,
                artifact_ref=artifact_ref,
                error_kind=error_kind,
                error_message=error_message,
            )
            .returning(JobAttempt.id)
        ).scalar_one_or_none()
        if owned_attempt is None:
            raise LostClaim("This attempt is no longer owned by this worker.")

        if publish is not None:
            publish(session)
        if result_resource_id is not None:
            session.add(
                JobAttemptResult(
                    attempt_id=attempt_id,
                    resource_id=result_resource_id,
                    artifact_ref=artifact_ref,
                    created_at=finished_at,
                )
            )
        session.commit()
