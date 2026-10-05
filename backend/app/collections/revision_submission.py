"""Confirm only the current fixed import report and queue its index atomically."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.collections.report_read import InvalidCurrentImportReport, read_current_import_report
from backend.app.collections.service import (
    StaleSelection, confirm_collection_revision, _revision_record, utcnow,
)
from backend.app.jobs.import_submission import InvalidJobRequest, _saved_request, _selections
from backend.app.jobs.service import IdempotencyConflict
from backend.app.models import (
    Collection, CollectionRevision, IndexArtifact, Job, JobAttempt,
    JobAttemptResult, JobRequest,
)


@dataclass(frozen=True, slots=True)
class RevisionSubmission:
    collection_id: str
    revision_id: str
    job_id: str


@dataclass(frozen=True, slots=True)
class IndexJobRequest:
    job_id: str
    revision_id: str
    snapshot_hash: str


class InvalidIndexRequest(ValueError):
    """The persisted index request no longer matches its Job."""


class IndexRetryNotAllowed(ValueError):
    """Only the current failed or interrupted index attempt may be retried."""


def _index_request(session: Session, job: Job) -> IndexJobRequest:
    saved = session.get(JobRequest, job.id)
    if saved is None or saved.format_version != 1:
        raise InvalidIndexRequest("Index request is unavailable.")
    if hashlib.sha256(saved.request_json.encode("utf-8")).hexdigest() != job.request_hash:
        raise InvalidIndexRequest("Index request hash differs.")
    try:
        payload = json.loads(saved.request_json)
    except (TypeError, ValueError) as error:
        raise InvalidIndexRequest("Index request JSON is invalid.") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"revision_id", "snapshot_hash", "collection_id", "selection_revision", "import_run_id", "document_ids"}
        or payload["revision_id"] != job.resource_id
        or not isinstance(payload["snapshot_hash"], str)
        or len(payload["snapshot_hash"]) != 64
        or not isinstance(payload["document_ids"], list)
    ):
        raise InvalidIndexRequest("Index request identity is invalid.")
    revision = session.get(CollectionRevision, job.resource_id)
    if (
        revision is None or revision.snapshot_hash != payload["snapshot_hash"]
        or revision.collection_id != payload["collection_id"]
        or revision.selection_revision != payload["selection_revision"]
        or revision.import_run_id != payload["import_run_id"]
        or [member.document_id for member in _revision_record(session, revision).members] != payload["document_ids"]
    ):
        raise InvalidIndexRequest("Frozen revision no longer matches index request.")
    return IndexJobRequest(job.id, revision.id, revision.snapshot_hash)


def get_index_request(engine: Engine, job_id: str) -> IndexJobRequest | None:
    with Session(engine) as session:
        job = session.get(Job, job_id)
        return _index_request(session, job) if job is not None and job.kind == "index" else None


def retry_index_job(engine: Engine, job_id: str, *, expected_attempt_id: str) -> str:
    if not expected_attempt_id:
        raise IndexRetryNotAllowed("A terminal attempt ID is required.")
    new_attempt_id = uuid4().hex
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if job is None or job.kind != "index":
            raise IndexRetryNotAllowed("Index Job was not found.")
        _index_request(session, job)
        attempt = session.get(JobAttempt, expected_attempt_id)
        if (
            job.status not in ("failed", "interrupted")
            or job.active_attempt_id != expected_attempt_id
            or attempt is None or attempt.job_id != job.id or attempt.status != job.status
        ):
            raise IndexRetryNotAllowed("Only the current terminal index attempt may be retried.")
        changed = session.execute(
            update(Job).where(
                Job.id == job_id, Job.status == job.status,
                Job.active_attempt_id == expected_attempt_id,
            ).values(status="queued", active_attempt_id=new_attempt_id, updated_at=utcnow())
            .returning(Job.id)
        ).scalar_one_or_none()
        if changed is None:
            raise IndexRetryNotAllowed("Index attempt changed before retry.")
        session.commit()
    return new_attempt_id


def _replay(session: Session, job: Job, collection_id: str, expected_revision: int,
            import_run_id: str, document_ids: tuple[str, ...]) -> RevisionSubmission:
    _index_request(session, job)
    payload = json.loads(session.get(JobRequest, job.id).request_json)
    if (
        payload["collection_id"] != collection_id
        or payload["selection_revision"] != expected_revision
        or payload["import_run_id"] != import_run_id
        or payload["document_ids"] != list(document_ids)
    ):
        raise IdempotencyConflict("Idempotency key belongs to another confirmation.")
    return RevisionSubmission(collection_id, job.resource_id, job.id)


def submit_revision_index(
    engine: Engine, collection_id: str, *, expected_selection_revision: int,
    import_run_id: str, document_ids: tuple[str, ...], idempotency_key: str,
    data_root: Path,
) -> RevisionSubmission:
    """Freeze the exact current report subset and queue indexing in one transaction."""

    if type(expected_selection_revision) is not int or expected_selection_revision < 1:
        raise ValueError("A positive selection revision is required.")
    if not idempotency_key.strip() or not import_run_id:
        raise ValueError("An idempotency key and import report ID are required.")
    if not document_ids or len(document_ids) != len(set(document_ids)) or any(not x for x in document_ids):
        raise ValueError("Confirm distinct successful document IDs explicitly.")

    with Session(engine) as session:
        existing = session.scalar(select(Job).where(Job.kind == "index", Job.idempotency_key == idempotency_key))
        if existing is not None:
            return _replay(session, existing, collection_id, expected_selection_revision, import_run_id, document_ids)
        selected_json = session.execute(
            update(Collection)
            .where(Collection.id == collection_id, Collection.selection_revision == expected_selection_revision)
            .values(updated_at=Collection.updated_at)
            .returning(Collection.selected_json)
        ).scalar_one_or_none()
        if selected_json is None:
            raise StaleSelection("Collection selection changed or does not exist.")
        try:
            selections = _selections(json.loads(selected_json))
        except (TypeError, ValueError) as error:
            raise ValueError("Current collection selection is invalid.") from error
        job = session.scalar(
            select(Job).where(Job.kind == "import", Job.resource_id == collection_id)
            .order_by(Job.created_at.desc(), Job.id.desc()).limit(1)
        )
        if job is None or job.status != "succeeded":
            raise StaleSelection("Current draft has no completed import report.")
        request = _saved_request(session, job)
        if request.selection_revision != expected_selection_revision or request.selections != selections:
            raise StaleSelection("Latest import belongs to a different draft selection.")
        report = read_current_import_report(session, job, request)
        if report is None or report["import_run_id"] != import_run_id:
            raise StaleSelection("Import report is not the current completed attempt.")
        success_ids = {item["document_id"] for item in report["items"] if item["status"] == "succeeded"}
        if any(document_id not in success_ids for document_id in document_ids):
            raise ValueError("Only successful documents in the current report may be confirmed.")
        revision = confirm_collection_revision(
            engine, collection_id, expected_selection_revision=expected_selection_revision,
            import_run_id=import_run_id, document_ids=document_ids,
            data_root=data_root, validated_session=session,
        )
        payload = {
            "collection_id": collection_id,
            "selection_revision": expected_selection_revision,
            "import_run_id": import_run_id,
            "document_ids": list(document_ids),
            "revision_id": revision.id,
            "snapshot_hash": revision.snapshot_hash,
        }
        request_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        job_id = uuid4().hex
        now = utcnow()
        session.add(Job(
            id=job_id, kind="index", resource_id=revision.id, status="queued",
            idempotency_key=idempotency_key,
            request_hash=hashlib.sha256(request_json.encode("utf-8")).hexdigest(),
            created_at=now, updated_at=now,
        ))
        session.add(JobRequest(job_id=job_id, format_version=1, request_json=request_json))
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            existing = session.scalar(select(Job).where(Job.kind == "index", Job.idempotency_key == idempotency_key))
            if existing is None:
                raise
            return _replay(session, existing, collection_id, expected_selection_revision, import_run_id, document_ids)
        return RevisionSubmission(collection_id, revision.id, job_id)


def read_revision_state(engine: Engine, collection_id: str, revision_id: str) -> dict | None:
    with Session(engine) as session:
        revision = session.get(CollectionRevision, revision_id)
        if revision is None or revision.collection_id != collection_id:
            return None
        record = _revision_record(session, revision)
        job = session.scalar(
            select(Job).where(Job.kind == "index", Job.resource_id == revision_id)
            .order_by(Job.created_at.desc(), Job.id.desc()).limit(1)
        )
        index_artifact_id = None
        if job is not None and job.status == "succeeded" and job.active_attempt_id:
            attempt = session.get(JobAttempt, job.active_attempt_id)
            binding = session.get(JobAttemptResult, job.active_attempt_id)
            artifact = session.get(IndexArtifact, binding.resource_id) if binding is not None else None
            if (
                attempt is None or attempt.job_id != job.id or attempt.status != "succeeded"
                or binding is None or artifact is None or artifact.revision_id != revision_id
                or artifact.status != "ready" or binding.artifact_ref != artifact.artifact_ref
                or artifact.artifact_ref != f"indexes/collections/{artifact.config_hash}"
                or attempt.artifact_ref != artifact.artifact_ref
            ):
                raise InvalidIndexRequest("Current index result is unavailable.")
            index_artifact_id = artifact.id
        return {
            **asdict(record), "index_job_id": job.id if job else None,
            "index_status": job.status if job else None,
            "index_artifact_id": index_artifact_id,
        }
