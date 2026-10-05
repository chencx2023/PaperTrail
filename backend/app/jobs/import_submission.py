"""Queue an import with the exact selected versions visible at submission."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import PurePosixPath
import re
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.collections.service import SavedSelection, StaleSelection, utcnow
from backend.app.jobs.service import IdempotencyConflict
from backend.app.models import Collection, Job, JobAttempt, JobRequest


class InvalidJobRequest(ValueError):
    """A saved Job request is missing, malformed, or differs from its hash."""


class ImportRetryNotAllowed(ValueError):
    """Only the current failed or interrupted import attempt may be retried."""


@dataclass(frozen=True, slots=True)
class ImportSubmission:
    job_id: str
    collection_id: str


@dataclass(frozen=True, slots=True)
class ImportRetry:
    job_id: str
    attempt_id: str


@dataclass(frozen=True, slots=True)
class ImportJobRequest:
    job_id: str
    collection_id: str
    selection_revision: int
    selections: tuple[SavedSelection, ...]
    source_search_ids: tuple[str, ...] | None = None


def _selections(value: object) -> tuple[SavedSelection, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= 10:
        raise InvalidJobRequest("Import request must contain 1 to 10 fixed versions.")
    result = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "source_ref", "versioned_id", "snapshot_sha256"
        }:
            raise InvalidJobRequest("Import selection is malformed.")
        source_ref = item["source_ref"]
        versioned_id = item["versioned_id"]
        snapshot_sha256 = item["snapshot_sha256"]
        if not all(isinstance(field, str) for field in item.values()):
            raise InvalidJobRequest("Import selection fields must be strings.")
        source_path = PurePosixPath(source_ref)
        if (
            "\\" in source_ref
            or ":" in source_ref
            or len(source_path.parts) != 2
            or source_path.parts[0] != "runs"
            or source_path.as_posix() != source_ref
            or not versioned_id
            or not re.fullmatch(r"[0-9a-f]{64}", snapshot_sha256)
        ):
            raise InvalidJobRequest("Import selection is not a fixed saved source.")
        result.append(SavedSelection(source_ref, versioned_id, snapshot_sha256))
    if len({item.versioned_id for item in result}) != len(result):
        raise InvalidJobRequest("A fixed version appears more than once.")
    return tuple(result)


def _saved_request(session: Session, job: Job) -> ImportJobRequest:
    row = session.get(JobRequest, job.id)
    if row is None or row.format_version not in (1, 2):
        raise InvalidJobRequest("The saved import request is unavailable.")
    if hashlib.sha256(row.request_json.encode("utf-8")).hexdigest() != job.request_hash:
        raise InvalidJobRequest("The saved import request hash differs.")
    try:
        payload = json.loads(row.request_json)
    except (TypeError, ValueError) as error:
        raise InvalidJobRequest("The saved import request is not valid JSON.") from error
    expected_fields = {"collection_id", "selection_revision", "selections"}
    if row.format_version == 2:
        expected_fields.add("source_search_ids")
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise InvalidJobRequest("The saved import request shape is invalid.")
    collection_id = payload["collection_id"]
    revision = payload["selection_revision"]
    if (
        not isinstance(collection_id, str)
        or collection_id != job.resource_id
        or type(revision) is not int
        or revision < 1
    ):
        raise InvalidJobRequest("The saved import request identity is invalid.")
    selections = _selections(payload["selections"])
    source_search_ids = None
    if row.format_version == 2:
        ids = payload["source_search_ids"]
        if (
            not isinstance(ids, list)
            or len(ids) != len(selections)
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value) for value in ids)
        ):
            raise InvalidJobRequest("The saved source search identities are invalid.")
        source_search_ids = tuple(ids)
    return ImportJobRequest(
        job_id=job.id,
        collection_id=collection_id,
        selection_revision=revision,
        selections=selections,
        source_search_ids=source_search_ids,
    )


def _replay(
    session: Session, job: Job, collection_id: str, expected_revision: int
) -> ImportSubmission:
    request = _saved_request(session, job)
    if (
        request.collection_id != collection_id
        or request.selection_revision != expected_revision
    ):
        raise IdempotencyConflict("Idempotency key belongs to another import request.")
    return ImportSubmission(job_id=job.id, collection_id=collection_id)


def submit_import_job(
    engine: Engine,
    collection_id: str,
    *,
    expected_selection_revision: int,
    idempotency_key: str,
) -> ImportSubmission:
    """Save the current ordered selection and Job in one short transaction."""

    if not collection_id:
        raise ValueError("A collection ID is required.")
    if type(expected_selection_revision) is not int or expected_selection_revision < 0:
        raise ValueError("A nonnegative selection revision is required.")
    if not idempotency_key.strip():
        raise ValueError("An idempotency key is required.")

    with Session(engine) as session:
        existing = session.scalar(
            select(Job).where(
                Job.kind == "import", Job.idempotency_key == idempotency_key
            )
        )
        if existing is not None:
            return _replay(session, existing, collection_id, expected_selection_revision)

        selected_json = session.execute(
            update(Collection)
            .where(
                Collection.id == collection_id,
                Collection.selection_revision == expected_selection_revision,
            )
            .values(updated_at=Collection.updated_at)
            .returning(Collection.selected_json)
        ).scalar_one_or_none()
        if selected_json is None:
            session.rollback()
            existing = session.scalar(
                select(Job).where(
                    Job.kind == "import", Job.idempotency_key == idempotency_key
                )
            )
            if existing is not None:
                return _replay(session, existing, collection_id, expected_selection_revision)
            raise StaleSelection("Collection selection changed or does not exist.")

        try:
            selections = _selections(json.loads(selected_json))
        except (TypeError, ValueError) as error:
            raise ValueError("Collection has no valid fixed selection.") from error
        request_json = json.dumps(
            {
                "collection_id": collection_id,
                "selection_revision": expected_selection_revision,
                "selections": [asdict(item) for item in selections],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        job_id = uuid4().hex
        now = utcnow()
        session.add(
            Job(
                id=job_id,
                kind="import",
                resource_id=collection_id,
                status="queued",
                idempotency_key=idempotency_key,
                request_hash=hashlib.sha256(request_json.encode("utf-8")).hexdigest(),
                created_at=now,
                updated_at=now,
            )
        )
        session.add(JobRequest(job_id=job_id, format_version=1, request_json=request_json))
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            existing = session.scalar(
                select(Job).where(
                    Job.kind == "import", Job.idempotency_key == idempotency_key
                )
            )
            if existing is None:
                raise
            return _replay(session, existing, collection_id, expected_selection_revision)
        return ImportSubmission(job_id=job_id, collection_id=collection_id)


def get_import_request(engine: Engine, job_id: str) -> ImportJobRequest | None:
    """Read the submitted list, independent of any later draft edits."""

    with Session(engine) as session:
        job = session.get(Job, job_id)
        if job is None or job.kind != "import":
            return None
        return _saved_request(session, job)


def retry_import_job(
    engine: Engine, job_id: str, *, expected_attempt_id: str
) -> ImportRetry:
    """Explicitly requeue one terminal import without changing its fixed request."""

    if not expected_attempt_id:
        raise ImportRetryNotAllowed("A completed attempt ID is required to retry.")
    new_attempt_id = uuid4().hex
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
        current = session.get(Job, job_id)
        if current is None or current.kind != "import":
            raise ImportRetryNotAllowed("Import Job was not found.")
        _saved_request(session, current)
        requeued = session.execute(
            update(Job)
            .where(
                Job.id == job_id,
                Job.kind == "import",
                Job.status.in_(("failed", "interrupted")),
                Job.active_attempt_id == expected_attempt_id,
                completed_attempt,
            )
            .values(
                status="queued",
                active_attempt_id=new_attempt_id,
                updated_at=now,
            )
            .returning(Job.id)
        ).scalar_one_or_none()
        if requeued is None:
            session.rollback()
            raise ImportRetryNotAllowed(
                "Only the current failed or interrupted import attempt may be retried."
            )
        session.commit()
    return ImportRetry(job_id=job_id, attempt_id=new_attempt_id)
