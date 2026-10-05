"""Keep a mutable selection separate from imported and frozen documents."""

from dataclasses import asdict, dataclass
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.ingestion.service import InvalidDocumentBlob, _checked_blob_ref
from backend.app.ingestion.selection import InvalidDocumentSelection, select_fixed_version
from backend.app.ingestion.storage import StoredPdf
from backend.app.models import (
    Collection,
    CollectionImportItem,
    CollectionImportRun,
    CollectionRevision,
    CollectionRevisionDocument,
    DocumentPreparation,
    DocumentVersion,
)


class StaleSelection(ValueError):
    """The caller edited or used an earlier draft revision."""


@dataclass(frozen=True, slots=True)
class SelectedSource:
    run_dir: Path
    versioned_id: str


@dataclass(frozen=True, slots=True)
class SavedSelection:
    source_ref: str
    versioned_id: str
    snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class CollectionRecord:
    id: str
    selection_revision: int
    selections: tuple[SavedSelection, ...]


@dataclass(frozen=True, slots=True)
class RevisionMemberRecord:
    document_id: str
    versioned_id: str
    sha256: str
    preparation_id: str
    parser_config_hash: str


@dataclass(frozen=True, slots=True)
class CollectionRevisionRecord:
    id: str
    collection_id: str
    import_run_id: str
    revision_no: int
    selection_revision: int
    snapshot_hash: str
    confirmed_at: str
    members: tuple[RevisionMemberRecord, ...]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def create_collection(engine: Engine) -> CollectionRecord:
    now = utcnow()
    collection = Collection(
        id=uuid4().hex,
        selection_revision=0,
        selected_json="[]",
        created_at=now,
        updated_at=now,
    )
    with Session(engine) as session:
        session.add(collection)
        session.commit()
        return _record(collection)


def set_collection_selection(
    engine: Engine,
    collection_id: str,
    *,
    expected_revision: int,
    data_root: Path,
    selections: tuple[SelectedSource, ...],
) -> CollectionRecord:
    if not 1 <= len(selections) <= 10:
        raise ValueError("Select between 1 and 10 fixed document versions.")
    if len({item.versioned_id for item in selections}) != len(selections):
        raise ValueError("A fixed version may be selected only once.")
    checked = []
    for item in selections:
        run_dir, ref = checked_run_dir(item.run_dir, data_root)
        source = select_fixed_version(run_dir, item.versioned_id)
        checked.append(SavedSelection(ref, source.versioned_id, source.snapshot_sha256))
    payload = json.dumps([asdict(item) for item in checked], separators=(",", ":"))
    with Session(engine) as session:
        changed = session.execute(
            update(Collection)
            .where(Collection.id == collection_id, Collection.selection_revision == expected_revision)
            .values(
                selection_revision=expected_revision + 1,
                selected_json=payload,
                updated_at=utcnow(),
            )
            .returning(Collection.id)
        ).scalar_one_or_none()
        if changed is None:
            session.rollback()
            raise StaleSelection("Collection selection changed or does not exist.")
        session.commit()
    return get_collection(engine, collection_id)


def get_collection(engine: Engine, collection_id: str) -> CollectionRecord | None:
    with Session(engine) as session:
        collection = session.get(Collection, collection_id)
        return None if collection is None else _record(collection)


def confirm_collection_revision(
    engine: Engine,
    collection_id: str,
    *,
    expected_selection_revision: int,
    import_run_id: str,
    document_ids: tuple[str, ...],
    data_root: Path,
    validated_session: Session | None = None,
) -> CollectionRevisionRecord:
    """Freeze selected successes; a supplied session has already validated the Job report."""

    if not document_ids or len(document_ids) != len(set(document_ids)):
        raise ValueError("Confirm a nonempty list of distinct successful documents.")
    owns_session = validated_session is None
    with Session(engine) if owns_session else nullcontext(validated_session) as session:
        collection = session.get(Collection, collection_id)
        if collection is None or collection.selection_revision != expected_selection_revision:
            raise StaleSelection("Collection selection changed or does not exist.")
        latest = (
            session.scalar(
                select(CollectionImportRun)
                .where(CollectionImportRun.collection_id == collection_id)
                .order_by(CollectionImportRun.created_at.desc(), CollectionImportRun.id.desc())
                .limit(1)
            ) if owns_session else session.get(CollectionImportRun, import_run_id)
        )
        if (
            latest is None
            or latest.id != import_run_id
            or latest.selection_revision != expected_selection_revision
            or latest.status not in ("succeeded", "partial_failed")
        ):
            raise StaleSelection("Confirm the latest completed import for the current draft.")
        items = session.scalars(
            select(CollectionImportItem).where(CollectionImportItem.import_run_id == import_run_id)
        ).all()
        successes = {item.document_id: item for item in items if item.status == "succeeded"}
        if any(document_id not in successes for document_id in document_ids):
            raise ValueError("Only successful imported documents may be confirmed.")

        members = []
        for document_id in document_ids:
            item = successes[document_id]
            document = session.get(DocumentVersion, document_id)
            preparation = session.get(DocumentPreparation, item.preparation_id)
            if (
                document is None or preparation is None
                or preparation.document_id != document_id
                or preparation.status != "succeeded"
                or document.versioned_source_id != item.versioned_id
            ):
                raise ValueError("Import result no longer matches a successful document preparation.")
            stored = StoredPdf(
                versioned_id=document.versioned_source_id,
                path=Path(data_root) / document.blob_ref,
                sha256=document.sha256,
                byte_count=document.byte_count,
                page_count=document.page_count,
                reused=True,
            )
            if _checked_blob_ref(stored, data_root) != document.blob_ref:
                raise InvalidDocumentBlob("Confirmed PDF blob reference is not canonical.")
            members.append(RevisionMemberRecord(
                document_id=document_id,
                versioned_id=document.versioned_source_id,
                sha256=document.sha256,
                preparation_id=preparation.id,
                parser_config_hash=preparation.parser_config_hash,
            ))
        snapshot = {
            "schema": 1,
            "documents": sorted([asdict(member) for member in members], key=lambda m: (m["versioned_id"], m["sha256"])),
        }
        snapshot_hash = hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        existing = session.scalar(
            select(CollectionRevision).where(
                CollectionRevision.collection_id == collection_id,
                CollectionRevision.import_run_id == import_run_id,
                CollectionRevision.snapshot_hash == snapshot_hash,
            )
        )
        if existing is not None:
            return _revision_record(session, existing)
        revision_no = (session.scalar(
            select(func.max(CollectionRevision.revision_no))
            .where(CollectionRevision.collection_id == collection_id)
        ) or 0) + 1
        revision = CollectionRevision(
            id=uuid4().hex,
            collection_id=collection_id,
            import_run_id=import_run_id,
            revision_no=revision_no,
            selection_revision=expected_selection_revision,
            snapshot_hash=snapshot_hash,
            confirmed_at=utcnow(),
        )
        session.add(revision)
        for position, member in enumerate(members):
            session.add(CollectionRevisionDocument(
                id=uuid4().hex,
                revision_id=revision.id,
                position=position,
                **asdict(member),
            ))
        try:
            session.flush()
            if owns_session:
                session.commit()
        except IntegrityError:
            if not owns_session:
                raise
            session.rollback()
            existing = session.scalar(
                select(CollectionRevision).where(
                    CollectionRevision.collection_id == collection_id,
                    CollectionRevision.import_run_id == import_run_id,
                    CollectionRevision.snapshot_hash == snapshot_hash,
                )
            )
            if existing is None:
                raise
            return _revision_record(session, existing)
        return _revision_record(session, revision)


def get_collection_revision(
    engine: Engine, revision_id: str
) -> CollectionRevisionRecord | None:
    with Session(engine) as session:
        revision = session.get(CollectionRevision, revision_id)
        return None if revision is None else _revision_record(session, revision)


def _revision_record(session: Session, revision: CollectionRevision) -> CollectionRevisionRecord:
    members = session.scalars(
        select(CollectionRevisionDocument)
        .where(CollectionRevisionDocument.revision_id == revision.id)
        .order_by(CollectionRevisionDocument.position)
    ).all()
    return CollectionRevisionRecord(
        id=revision.id,
        collection_id=revision.collection_id,
        import_run_id=revision.import_run_id,
        revision_no=revision.revision_no,
        selection_revision=revision.selection_revision,
        snapshot_hash=revision.snapshot_hash,
        confirmed_at=revision.confirmed_at,
        members=tuple(RevisionMemberRecord(
            document_id=member.document_id,
            versioned_id=member.versioned_id,
            sha256=member.sha256,
            preparation_id=member.preparation_id,
            parser_config_hash=member.parser_config_hash,
        ) for member in members),
    )


def checked_run_dir(run_dir: Path, data_root: Path) -> tuple[Path, str]:
    try:
        root = Path(data_root).resolve(strict=True)
        runs_root = (root / "runs").resolve(strict=True)
        path = Path(run_dir).resolve(strict=True)
        if path.parent != runs_root or not path.is_dir():
            raise ValueError("Saved search must be a direct child of the runs directory.")
        return path, path.relative_to(root).as_posix()
    except (OSError, ValueError) as error:
        raise InvalidDocumentSelection("Search run is outside the data runs directory.") from error


def run_dir_from_ref(source_ref: str, data_root: Path) -> Path:
    if not source_ref.startswith("runs/") or "/" in source_ref[5:]:
        raise InvalidDocumentSelection("Stored search run reference is invalid.")
    path, checked_ref = checked_run_dir(Path(data_root) / source_ref, data_root)
    if checked_ref != source_ref:
        raise InvalidDocumentSelection("Stored search run reference is not canonical.")
    return path


def _record(collection: Collection) -> CollectionRecord:
    return CollectionRecord(
        id=collection.id,
        selection_revision=collection.selection_revision,
        selections=tuple(SavedSelection(**item) for item in json.loads(collection.selected_json)),
    )
