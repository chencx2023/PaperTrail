"""Persist answer identities before execution and preserve them across retries."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.collections.service import utcnow
from backend.app.jobs.service import IdempotencyConflict
from backend.app.models import (
    AnswerRun, AnswerSubmission, CollectionRevision, CollectionRevisionDocument, IndexArtifact,
    Job, JobAttempt, JobAttemptResult, JobRequest, DocumentVersion,
)


class AnswerNotReady(ValueError):
    """The requested revision and index are not a current ready pair."""


class AnswerResourceMissing(LookupError):
    """A requested revision or index artifact does not exist."""


class InvalidAnswerRequest(ValueError):
    """The saved answer request or result identity is invalid."""


class AnswerRetryNotAllowed(ValueError):
    """Only the current failed or interrupted answer attempt may be retried."""


@dataclass(frozen=True, slots=True)
class AnswerJobRequest:
    job_id: str
    revision_id: str
    index_artifact_id: str
    question: str
    initial_answer_run_id: str


@dataclass(frozen=True, slots=True)
class AnswerSubmissionRecord:
    answer_run_id: str
    job_id: str


@dataclass(frozen=True, slots=True)
class AnswerRetry:
    job_id: str
    attempt_id: str
    answer_run_id: str
    previous_answer_run_id: str


def _request(session: Session, job: Job) -> AnswerJobRequest:
    saved = session.get(JobRequest, job.id)
    if saved is None or saved.format_version != 1:
        raise InvalidAnswerRequest("Answer request is unavailable.")
    if hashlib.sha256(saved.request_json.encode("utf-8")).hexdigest() != job.request_hash:
        raise InvalidAnswerRequest("Answer request hash differs.")
    try:
        payload = json.loads(saved.request_json)
    except (TypeError, ValueError) as error:
        raise InvalidAnswerRequest("Answer request JSON is invalid.") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"revision_id", "index_artifact_id", "question", "initial_answer_run_id"}
        or any(not isinstance(value, str) or not value for value in payload.values())
    ):
        raise InvalidAnswerRequest("Answer request shape is invalid.")
    initial = session.get(AnswerSubmission, payload["initial_answer_run_id"])
    if initial is None or initial.job_id != job.id or initial.previous_answer_run_id is not None:
        raise InvalidAnswerRequest("Initial answer identity is unavailable.")
    return AnswerJobRequest(
        job.id, payload["revision_id"], payload["index_artifact_id"],
        payload["question"], payload["initial_answer_run_id"],
    )


def get_answer_request(engine: Engine, job_id: str) -> AnswerJobRequest | None:
    with Session(engine) as session:
        job = session.get(Job, job_id)
        return _request(session, job) if job is not None and job.kind == "answer" else None


def _replay(session: Session, job: Job, revision_id: str, index_artifact_id: str,
            question: str) -> AnswerSubmissionRecord:
    request = _request(session, job)
    if (
        request.revision_id != revision_id
        or request.index_artifact_id != index_artifact_id
        or request.question != question
    ):
        raise IdempotencyConflict("Idempotency key belongs to another answer request.")
    return AnswerSubmissionRecord(request.initial_answer_run_id, job.id)


def submit_answer(
    engine: Engine, *, revision_id: str, index_artifact_id: str,
    question: str, idempotency_key: str,
) -> AnswerSubmissionRecord:
    if not question.strip() or len(question) > 10000:
        raise ValueError("A question of 1 to 10000 characters is required.")
    if not idempotency_key.strip():
        raise ValueError("An idempotency key is required.")
    with Session(engine) as session:
        existing = session.scalar(select(Job).where(
            Job.kind == "answer", Job.idempotency_key == idempotency_key,
        ))
        if existing is not None:
            return _replay(session, existing, revision_id, index_artifact_id, question)
        revision = session.get(CollectionRevision, revision_id)
        artifact = session.get(IndexArtifact, index_artifact_id)
        if revision is None or artifact is None:
            raise AnswerResourceMissing("Revision or index artifact was not found.")
        index_job = session.scalar(
            select(Job).where(Job.kind == "index", Job.resource_id == revision_id)
            .order_by(Job.created_at.desc(), Job.id.desc()).limit(1)
        )
        attempt = (
            session.get(JobAttempt, index_job.active_attempt_id)
            if index_job is not None and index_job.active_attempt_id else None
        )
        binding = session.get(JobAttemptResult, attempt.id) if attempt is not None else None
        if (
            revision is None or artifact is None or artifact.revision_id != revision_id
            or artifact.status != "ready"
            or artifact.artifact_ref != f"indexes/collections/{artifact.config_hash}"
            or index_job is None or index_job.status != "succeeded"
            or attempt is None or attempt.job_id != index_job.id or attempt.status != "succeeded"
            or binding is None or binding.resource_id != index_artifact_id
            or binding.artifact_ref != artifact.artifact_ref
            or attempt.artifact_ref != artifact.artifact_ref
        ):
            raise AnswerNotReady("A current ready index for this revision is required.")
        answer_run_id = uuid4().hex
        job_id = uuid4().hex
        request_json = json.dumps({
            "revision_id": revision_id,
            "index_artifact_id": index_artifact_id,
            "question": question,
            "initial_answer_run_id": answer_run_id,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        now = utcnow()
        session.add(Job(
            id=job_id, kind="answer", resource_id=answer_run_id, status="queued",
            idempotency_key=idempotency_key,
            request_hash=hashlib.sha256(request_json.encode("utf-8")).hexdigest(),
            created_at=now, updated_at=now,
        ))
        session.add(JobRequest(job_id=job_id, format_version=1, request_json=request_json))
        try:
            session.flush()
            session.add(AnswerSubmission(
                answer_run_id=answer_run_id, job_id=job_id, attempt_id=None,
                previous_answer_run_id=None, submitted_at=now,
            ))
            session.commit()
        except IntegrityError:
            session.rollback()
            existing = session.scalar(select(Job).where(
                Job.kind == "answer", Job.idempotency_key == idempotency_key,
            ))
            if existing is None:
                raise
            return _replay(session, existing, revision_id, index_artifact_id, question)
        return AnswerSubmissionRecord(answer_run_id, job_id)


def retry_answer(engine: Engine, job_id: str, *, expected_attempt_id: str) -> AnswerRetry:
    if not expected_attempt_id:
        raise AnswerRetryNotAllowed("A terminal attempt ID is required.")
    new_answer_id = uuid4().hex
    new_attempt_id = uuid4().hex
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if job is None or job.kind != "answer":
            raise AnswerRetryNotAllowed("Answer Job was not found.")
        _request(session, job)
        old_answer_id = job.resource_id
        old = session.get(AnswerSubmission, old_answer_id)
        attempt = session.get(JobAttempt, expected_attempt_id)
        if (
            job.status not in ("failed", "interrupted")
            or job.active_attempt_id != expected_attempt_id
            or old is None or old.job_id != job_id or old.attempt_id != expected_attempt_id
            or attempt is None or attempt.job_id != job_id or attempt.status != job.status
        ):
            raise AnswerRetryNotAllowed("Only the current terminal answer attempt may be retried.")
        changed = session.execute(
            update(Job).where(
                Job.id == job_id, Job.status == job.status,
                Job.active_attempt_id == expected_attempt_id,
                Job.resource_id == old_answer_id,
            ).values(
                status="queued", active_attempt_id=new_attempt_id,
                resource_id=new_answer_id, updated_at=utcnow(),
            ).returning(Job.id)
        ).scalar_one_or_none()
        if changed is None:
            raise AnswerRetryNotAllowed("Answer attempt changed before retry.")
        session.add(AnswerSubmission(
            answer_run_id=new_answer_id, job_id=job_id,
            attempt_id=None, previous_answer_run_id=old_answer_id,
            submitted_at=utcnow(),
        ))
        session.commit()
    return AnswerRetry(job_id, new_attempt_id, new_answer_id, old_answer_id)


def _read_saved_answer_payload(engine: Engine, run_id: str, *, data_root: Path) -> dict:
    from backend.app.evidence.persistence import load_answer_run
    from backend.app.evidence.service import summarize_document_usage, summarize_saved_context_coverage
    from backend.app.evidence.sources import frozen_document_sources

    bundle = load_answer_run(engine, run_id, data_root=data_root)
    with Session(engine) as session:
        row = session.get(AnswerRun, run_id)
        if row is None:
            raise LookupError("No committed answer run has this ID.")
        revision = session.get(CollectionRevision, row.collection_revision_id)
        research_context = (
            {"collection_id": revision.collection_id, "revision_id": revision.id}
            if revision is not None else None
        )
        members = session.scalars(
            select(CollectionRevisionDocument)
            .where(CollectionRevisionDocument.revision_id == row.collection_revision_id)
            .order_by(CollectionRevisionDocument.position)
        ).all()
        sources = frozen_document_sources(session, bundle.allowed_documents, data_root=data_root)
        titles = {key: source['title'] for key, source in sources.items()}
    payload = bundle.model_dump()
    payload["document_titles"] = titles
    payload["document_sources"] = sources
    payload["saved_context_coverage"] = summarize_saved_context_coverage(bundle, members)
    payload["document_usage"] = summarize_document_usage(bundle, members)
    payload["research_context"] = research_context
    return payload


def read_answer_state(engine: Engine, run_id: str, *, data_root: Path) -> dict | None:
    with Session(engine) as session:
        submission = session.get(AnswerSubmission, run_id)
        if submission is None:
            historical = session.get(AnswerRun, run_id)
            if historical is None:
                return None
            if historical.job_attempt_id is not None:
                raise InvalidAnswerRequest("An attempted answer lacks its submission identity.")
            result = {
                "answer_run_id": run_id, "job_id": None, "attempt_id": None,
                "previous_answer_run_id": None,
                "status": historical.execution_status, "answer": None,
            }
            result["answer"] = _read_saved_answer_payload(engine, run_id, data_root=data_root)
            return result
        job = session.get(Job, submission.job_id)
        if job is None or job.kind != "answer":
            raise InvalidAnswerRequest("Answer Job is unavailable.")
        attempt = session.get(JobAttempt, submission.attempt_id) if submission.attempt_id else None
        if submission.attempt_id is None:
            if job.resource_id != run_id or job.status != "queued":
                raise InvalidAnswerRequest("Unclaimed answer state is inconsistent.")
            status = "queued"
        elif attempt is None or attempt.job_id != job.id:
            raise InvalidAnswerRequest("Answer attempt is unavailable.")
        else:
            status = attempt.status
        row = session.get(AnswerRun, run_id)
        if row is not None and (attempt is None or row.job_attempt_id != attempt.id or status not in ("succeeded", "failed")):
            raise InvalidAnswerRequest("Committed answer belongs to another attempt.")
        result = {
            "answer_run_id": run_id, "job_id": job.id,
            "attempt_id": submission.attempt_id,
            "previous_answer_run_id": submission.previous_answer_run_id,
            "status": status, "answer": None,
        }
    if row is not None:
        result["answer"] = _read_saved_answer_payload(engine, run_id, data_root=data_root)
    return result
