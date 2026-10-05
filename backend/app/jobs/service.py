"""Submit search work without performing the slow search in the caller."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.models import Job, JobAttempt, JobRequest, SearchRun
from backend.app.search.service import QUERY_CONTRACT_VERSION, build_arxiv_query
from backend.app.search.query_builder import InvalidQueryPlan, plan_from_data


class IdempotencyConflict(ValueError):
    """The same key was submitted with a different request."""


class RetryNotAllowed(ValueError):
    """Only a completed failed or interrupted search attempt can be retried."""


@dataclass(frozen=True, slots=True)
class SearchSubmission:
    search_id: str
    job_id: str


@dataclass(frozen=True, slots=True)
class SearchRetry:
    job_id: str
    attempt_id: str


@dataclass(frozen=True, slots=True)
class SearchRecord:
    id: str
    question: str
    query_text: str
    normalized_query_text: str
    search_query: str
    snapshot_ref: str | None
    query_plan: dict | None = None


@dataclass(frozen=True, slots=True)
class JobRecord:
    id: str
    kind: str
    resource_id: str
    status: str
    active_attempt_id: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    id: str
    job_id: str
    attempt_no: int
    worker_id: str
    status: str
    started_at: str
    heartbeat_at: str | None
    lease_until: str | None
    finished_at: str | None
    artifact_ref: str | None
    error_kind: str | None
    error_message: str | None


def submit_search(
    engine: Engine,
    *,
    question: str,
    query_text: str,
    idempotency_key: str,
    query_plan: dict | None = None,
    plan_source: dict | None = None,
) -> SearchSubmission:
    if not question.strip():
        raise ValueError("A research question is required.")
    if not idempotency_key.strip():
        raise ValueError("An idempotency key is required.")
    plan = build_arxiv_query(query_text)
    multi = plan_from_data(query_plan) if query_plan is not None else None
    if multi is not None and multi.baseline_query_text != plan.normalized_query_text:
        raise InvalidQueryPlan("原检索词已修改，请重新扩展或取消多查询。")
    identity = {"question": question, "query_text": query_text, "query_contract_version": QUERY_CONTRACT_VERSION}
    if multi is not None:
        identity.update(query_plan=multi.as_data(), plan_source=plan_source)
    request_hash = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with Session(engine) as session:
        search_id = uuid4().hex
        job_id = uuid4().hex
        now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
        session.add(
            SearchRun(
                id=search_id,
                question=question,
                query_text=query_text,
                normalized_query_text=plan.normalized_query_text,
                search_query=plan.search_query,
                query_contract_version=QUERY_CONTRACT_VERSION,
                created_at=now,
            )
        )
        session.add(
            Job(
                id=job_id,
                kind="search",
                resource_id=search_id,
                status="queued",
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                created_at=now,
                updated_at=now,
            )
        )
        if multi is not None:
            session.add(JobRequest(job_id=job_id, format_version=1, request_json=json.dumps(
                {"kind": "multi_search", "plan": multi.as_data(), "generation": plan_source},
                ensure_ascii=False, sort_keys=True)))
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            existing = session.scalar(
                select(Job).where(
                    Job.kind == "search", Job.idempotency_key == idempotency_key
                )
            )
            if existing is None:
                raise
            if existing.request_hash != request_hash:
                raise IdempotencyConflict(
                    "Idempotency key already belongs to another search."
                )
            return SearchSubmission(search_id=existing.resource_id, job_id=existing.id)
        return SearchSubmission(search_id=search_id, job_id=job_id)


def retry_search(
    engine: Engine, job_id: str, *, expected_attempt_id: str
) -> SearchRetry:
    """Explicitly requeue a terminal search, reserving the next attempt ID."""

    if not expected_attempt_id:
        raise RetryNotAllowed("A completed attempt ID is required to retry.")
    attempt_id = uuid4().hex
    now = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )
    completed_attempt = (
        select(JobAttempt.id)
        .where(
            JobAttempt.id == Job.active_attempt_id,
            JobAttempt.job_id == Job.id,
            JobAttempt.status == Job.status,
        )
        .exists()
    )
    with Session(engine) as session:
        search_id = session.execute(
            update(Job)
            .where(
                Job.id == job_id,
                Job.kind == "search",
                Job.status.in_(("failed", "interrupted")),
                Job.active_attempt_id == expected_attempt_id,
                completed_attempt,
            )
            .values(
                status="queued",
                active_attempt_id=attempt_id,
                updated_at=now,
            )
            .returning(Job.resource_id)
        ).scalar_one_or_none()
        if search_id is None:
            session.rollback()
            raise RetryNotAllowed(
                "Only the current failed or interrupted search attempt may be retried."
            )
        cleared = session.execute(
            update(SearchRun)
            .where(SearchRun.id == search_id)
            .values(snapshot_ref=None, retrieved_at=None)
            .returning(SearchRun.id)
        ).scalar_one_or_none()
        if cleared is None:
            session.rollback()
            raise RuntimeError("A search job has no search resource.")
        session.commit()
    return SearchRetry(job_id=job_id, attempt_id=attempt_id)


def get_search(engine: Engine, search_id: str) -> SearchRecord | None:
    with Session(engine) as session:
        search = session.get(SearchRun, search_id)
        if search is None:
            return None
        job = session.scalar(select(Job).where(Job.kind == "search", Job.resource_id == search.id))
        request = read_search_request(session, job.id) if job is not None else None
        if request and request["plan"]["baseline_query_text"] != search.normalized_query_text:
            raise InvalidQueryPlan("保存的多查询与原检索词不一致。")
        return SearchRecord(
            id=search.id,
            question=search.question,
            query_text=search.query_text,
            normalized_query_text=search.normalized_query_text,
            search_query=search.search_query,
            snapshot_ref=search.snapshot_ref,
            query_plan=request["plan"] if request else None,
        )


def read_search_request(session: Session, job_id: str) -> dict | None:
    saved = session.get(JobRequest, job_id)
    if saved is None:
        return None
    try:
        value = json.loads(saved.request_json)
        if saved.format_version != 1 or set(value) != {"kind", "plan", "generation"} or value["kind"] != "multi_search":
            raise ValueError()
        if value["generation"] is not None and not isinstance(value["generation"], dict):
            raise ValueError()
        return {**value, "plan": plan_from_data(value["plan"]).as_data()}
    except (ValueError, TypeError, KeyError):
        raise InvalidQueryPlan("保存的多查询请求无法核验。") from None


def get_job(engine: Engine, job_id: str) -> JobRecord | None:
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if job is None:
            return None
        return JobRecord(
            id=job.id,
            kind=job.kind,
            resource_id=job.resource_id,
            status=job.status,
            active_attempt_id=job.active_attempt_id,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )


def get_search_job(engine: Engine, search_id: str) -> JobRecord | None:
    with Session(engine) as session:
        job = session.scalar(
            select(Job).where(Job.kind == "search", Job.resource_id == search_id)
        )
        if job is None:
            return None
        return JobRecord(
            id=job.id,
            kind=job.kind,
            resource_id=job.resource_id,
            status=job.status,
            active_attempt_id=job.active_attempt_id,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )


def get_attempt(engine: Engine, attempt_id: str) -> AttemptRecord | None:
    with Session(engine) as session:
        attempt = session.get(JobAttempt, attempt_id)
        if attempt is None:
            return None
        return _attempt_record(attempt)


def list_attempts(engine: Engine, job_id: str) -> tuple[AttemptRecord, ...]:
    """Return persisted attempts in sequence; a queued reservation is not started."""

    with Session(engine) as session:
        attempts = session.scalars(
            select(JobAttempt)
            .where(JobAttempt.job_id == job_id)
            .order_by(JobAttempt.attempt_no)
        ).all()
        return tuple(_attempt_record(attempt) for attempt in attempts)


def _attempt_record(attempt: JobAttempt) -> AttemptRecord:
    return AttemptRecord(
        id=attempt.id,
        job_id=attempt.job_id,
        attempt_no=attempt.attempt_no,
        worker_id=attempt.worker_id,
        status=attempt.status,
        started_at=attempt.started_at,
        heartbeat_at=attempt.heartbeat_at,
        lease_until=attempt.lease_until,
        finished_at=attempt.finished_at,
        artifact_ref=attempt.artifact_ref,
        error_kind=attempt.error_kind,
        error_message=attempt.error_message,
    )
