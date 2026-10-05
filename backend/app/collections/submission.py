"""Create a draft and its first fixed import Job from saved search resources."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.collections.report_read import read_current_import_report
from backend.app.collections.service import (
    SavedSelection,
    StaleSelection,
    checked_run_dir,
    utcnow,
)
from backend.app.ingestion.selection import InvalidDocumentSelection, select_fixed_version
from backend.app.jobs.import_submission import _saved_request, _selections
from backend.app.jobs.service import IdempotencyConflict
from backend.app.models import Collection, Job, JobAttempt, JobRequest, SearchRun
from backend.app.search.service import InvalidSearchSnapshot, replay_search_run


class CollectionSearchUnavailable(ValueError):
    """The referenced search does not exist."""


class CollectionSearchNotReady(ValueError):
    """The referenced search has no successful current attempt."""


class CollectionSnapshotInvalid(ValueError):
    """The current search attempt does not own a valid saved snapshot."""


class InvalidCollectionState(ValueError):
    """A saved collection draft is malformed."""


class CollectionUnavailable(ValueError):
    """The collection being edited does not exist."""


@dataclass(frozen=True, slots=True)
class SearchChoice:
    search_id: str
    versioned_id: str


@dataclass(frozen=True, slots=True)
class CollectionSubmission:
    collection_id: str
    job_id: str


@dataclass(frozen=True, slots=True)
class CollectionState:
    collection_id: str
    selection_revision: int
    selections: tuple[dict, ...]
    import_job_id: str | None
    import_job_status: str | None
    import_report: dict | None


def _replay_creation(session: Session, job: Job, choices: tuple[SearchChoice, ...]) -> CollectionSubmission:
    request = _saved_request(session, job)
    original = (
        tuple(
            SearchChoice(search_id, item.versioned_id)
            for search_id, item in zip(request.source_search_ids, request.selections, strict=True)
        )
        if request.source_search_ids is not None
        else None
    )
    if request.selection_revision != 1 or original != choices:
        raise IdempotencyConflict("Idempotency key belongs to another import request.")
    if session.get(Collection, request.collection_id) is None:
        raise CollectionSnapshotInvalid("The original collection is unavailable.")
    return CollectionSubmission(request.collection_id, job.id)


def _validate_choices(choices: tuple[SearchChoice, ...]) -> None:
    if not 1 <= len(choices) <= 10:
        raise ValueError("Select between 1 and 10 fixed document versions.")
    if any(not choice.search_id or not choice.versioned_id for choice in choices):
        raise ValueError("Every selection needs a search ID and fixed version.")
    if len({choice.versioned_id for choice in choices}) != len(choices):
        raise ValueError("A fixed version may be selected only once.")


def _resolve_choices(
    session: Session, *, data_root: Path, choices: tuple[SearchChoice, ...]
) -> tuple[SavedSelection, ...]:
    fixed = []
    for choice in choices:
        search = session.get(SearchRun, choice.search_id)
        if search is None:
            raise CollectionSearchUnavailable("Search not found.")
        job = session.scalar(
            select(Job).where(Job.kind == "search", Job.resource_id == search.id)
        )
        if job is None or job.status != "succeeded" or job.active_attempt_id is None:
            raise CollectionSearchNotReady("Search is not complete.")
        attempt = session.get(JobAttempt, job.active_attempt_id)
        expected_ref = f"runs/search-{job.active_attempt_id}/run.json"
        if (
            attempt is None
            or attempt.job_id != job.id
            or attempt.status != "succeeded"
            or attempt.artifact_ref != expected_ref
            or search.snapshot_ref != expected_ref
        ):
            raise CollectionSnapshotInvalid("Saved search result is unavailable.")
        try:
            run_dir, source_ref = checked_run_dir(
                Path(data_root) / "runs" / f"search-{attempt.id}", data_root
            )
            replayed = replay_search_run(run_dir)
        except (InvalidDocumentSelection, InvalidSearchSnapshot) as error:
            raise CollectionSnapshotInvalid("Saved search result is unavailable.") from error
        if not replayed.candidates_match or replayed.source_run_id != f"search-{attempt.id}":
            raise CollectionSnapshotInvalid("Saved search result is unavailable.")
        source = select_fixed_version(run_dir, choice.versioned_id)
        if source.source_run_id != replayed.source_run_id or source.snapshot_sha256 != replayed.snapshot_sha256:
            raise CollectionSnapshotInvalid("Saved search result changed during selection.")
        fixed.append(SavedSelection(source_ref, source.versioned_id, source.snapshot_sha256))
    return tuple(fixed)


def create_collection_with_import(
    engine: Engine,
    *,
    data_root: Path,
    choices: tuple[SearchChoice, ...],
    idempotency_key: str,
) -> CollectionSubmission:
    """Commit the draft, immutable import request, and queued Job together."""

    if not idempotency_key.strip():
        raise ValueError("An idempotency key is required.")
    _validate_choices(choices)

    with Session(engine) as session:
        existing = session.scalar(
            select(Job).where(Job.kind == "import", Job.idempotency_key == idempotency_key)
        )
        if existing is not None:
            return _replay_creation(session, existing, choices)

        fixed = _resolve_choices(session, data_root=data_root, choices=choices)

        collection_id = uuid4().hex
        job_id = uuid4().hex
        now = utcnow()
        selections_json = json.dumps(
            [asdict(item) for item in fixed], ensure_ascii=False, separators=(",", ":")
        )
        request_json = json.dumps(
            {
                "collection_id": collection_id,
                "selection_revision": 1,
                "selections": [asdict(item) for item in fixed],
                "source_search_ids": [choice.search_id for choice in choices],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        session.add(
            Collection(
                id=collection_id,
                selection_revision=1,
                selected_json=selections_json,
                created_at=now,
                updated_at=now,
            )
        )
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
        session.add(JobRequest(job_id=job_id, format_version=2, request_json=request_json))
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            existing = session.scalar(
                select(Job).where(Job.kind == "import", Job.idempotency_key == idempotency_key)
            )
            if existing is None:
                raise
            return _replay_creation(session, existing, choices)
        return CollectionSubmission(collection_id, job_id)


def edit_collection_draft(
    engine: Engine,
    collection_id: str,
    *,
    expected_revision: int,
    data_root: Path,
    choices: tuple[SearchChoice, ...],
) -> int:
    """Replace a draft only if the caller still holds its current revision."""

    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("A nonnegative selection revision is required.")
    _validate_choices(choices)
    with Session(engine) as session:
        collection = session.get(Collection, collection_id)
        if collection is None:
            raise CollectionUnavailable("Collection not found.")
        if collection.selection_revision != expected_revision:
            raise StaleSelection("Collection selection changed.")
        fixed = _resolve_choices(session, data_root=data_root, choices=choices)
        selected_json = json.dumps(
            [asdict(item) for item in fixed], ensure_ascii=False, separators=(",", ":")
        )
        changed = session.execute(
            update(Collection)
            .where(
                Collection.id == collection_id,
                Collection.selection_revision == expected_revision,
            )
            .values(
                selection_revision=expected_revision + 1,
                selected_json=selected_json,
                updated_at=utcnow(),
            )
            .returning(Collection.selection_revision)
        ).scalar_one_or_none()
        if changed is None:
            session.rollback()
            raise StaleSelection("Collection selection changed.")
        session.commit()
        return changed


def get_collection_state(engine: Engine, collection_id: str) -> CollectionState | None:
    """Read one consistent draft and the latest import submitted for its revision."""

    with Session(engine) as session:
        collection = session.get(Collection, collection_id)
        if collection is None:
            return None
        try:
            payload = json.loads(collection.selected_json)
            if not isinstance(payload, list) or (
                collection.selection_revision == 0 and payload
            ) or (collection.selection_revision > 0 and not payload):
                raise ValueError("Collection selection shape is invalid.")
            selections = _selections(payload) if payload else ()
        except (TypeError, ValueError) as error:
            raise InvalidCollectionState("Saved collection selection is unavailable.") from error
        shown = []
        for item in selections:
            search = session.scalar(
                select(SearchRun)
                .where(SearchRun.snapshot_ref == f"{item.source_ref}/run.json")
                .limit(1)
            )
            shown.append(
                {
                    "search_id": search.id if search is not None else None,
                    "versioned_id": item.versioned_id,
                    "snapshot_sha256": item.snapshot_sha256,
                }
            )
        import_job_id = None
        import_job_status = None
        import_report = None
        jobs = session.scalars(
            select(Job)
            .where(Job.kind == "import", Job.resource_id == collection.id)
            .order_by(Job.created_at.desc(), Job.id.desc())
        )
        for job in jobs:
            request = _saved_request(session, job)
            if (
                request.selection_revision == collection.selection_revision
                and request.selections == selections
            ):
                import_job_id = job.id
                import_job_status = job.status
                import_report = read_current_import_report(session, job, request)
                break
        return CollectionState(
            collection_id=collection.id,
            selection_revision=collection.selection_revision,
            selections=tuple(shown),
            import_job_id=import_job_id,
            import_job_status=import_job_status,
            import_report=import_report,
        )
