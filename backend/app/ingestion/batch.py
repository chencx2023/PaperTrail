"""Import each selected fixed version and persist failures alongside successes."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import httpx
from sqlalchemy import select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from backend.app.collections.service import (
    SavedSelection,
    StaleSelection,
    get_collection,
    run_dir_from_ref,
    utcnow,
)
from backend.app.ingestion.downloader import stage_pdf_download
from backend.app.ingestion.preparation import prepare_document_version
from backend.app.ingestion.selection import FixedVersionSource, InvalidDocumentSelection, select_fixed_version
from backend.app.ingestion.service import (
    DocumentVersionRecord,
    find_downloaded_version,
    register_downloaded_version,
)
from backend.app.ingestion.storage import publish_staged_pdf
from backend.app.jobs.import_submission import ImportJobRequest
from backend.app.models import (
    CollectionImportItem,
    CollectionImportRun,
    DocumentPreparation,
    DocumentVersion,
)


@dataclass(frozen=True, slots=True)
class ImportItemRecord:
    id: str
    position: int
    source_ref: str
    versioned_id: str
    status: str
    document_id: str | None
    preparation_id: str | None
    reused_pdf: bool
    error_kind: str | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class ImportReport:
    id: str
    collection_id: str
    selection_revision: int
    status: str
    items: tuple[ImportItemRecord, ...]


@dataclass(frozen=True, slots=True)
class StagedImport:
    run_id: str
    collection_id: str
    selection_revision: int
    selections: tuple[SavedSelection, ...]


async def import_collection(
    engine: Engine,
    collection_id: str,
    *,
    expected_selection_revision: int,
    data_root: Path,
    client: httpx.Client,
) -> ImportReport:
    collection = get_collection(engine, collection_id)
    if collection is None or collection.selection_revision != expected_selection_revision:
        raise StaleSelection("Collection selection changed or does not exist.")
    if not collection.selections:
        raise ValueError("Select at least one fixed version before importing.")
    staged = await _stage_fixed_import(
        engine,
        collection_id=collection_id,
        selection_revision=expected_selection_revision,
        selections=collection.selections,
        run_id=uuid4().hex,
        data_root=data_root,
        client=client,
    )
    with Session(engine) as session:
        publish_import_run(session, staged, finished_at=utcnow())
        session.commit()
    return get_import_report(engine, staged.run_id)


async def stage_import_job(
    engine: Engine,
    request: ImportJobRequest,
    *,
    attempt_id: str,
    data_root: Path,
    client: httpx.Client,
) -> StagedImport:
    """Import only the submitted list; leave report publication to the Job fence."""

    return await _stage_fixed_import(
        engine,
        collection_id=request.collection_id,
        selection_revision=request.selection_revision,
        selections=request.selections,
        run_id=attempt_id,
        data_root=data_root,
        client=client,
    )


async def _stage_fixed_import(
    engine: Engine,
    *,
    collection_id: str,
    selection_revision: int,
    selections: tuple[SavedSelection, ...],
    run_id: str,
    data_root: Path,
    client: httpx.Client,
) -> StagedImport:
    if not selections:
        raise ValueError("Select at least one fixed version before importing.")
    with Session(engine) as session:
        session.add(CollectionImportRun(
            id=run_id,
            collection_id=collection_id,
            selection_revision=selection_revision,
            status="running",
            created_at=utcnow(),
        ))
        session.commit()

    for position, selected in enumerate(selections):
        document_id = preparation_id = None
        reused_pdf = False
        error_kind = error_message = None
        try:
            run_dir = run_dir_from_ref(selected.source_ref, data_root)
            source = select_fixed_version(run_dir, selected.versioned_id)
            if source.snapshot_sha256 != selected.snapshot_sha256:
                raise InvalidDocumentSelection("Saved source snapshot changed after selection.")
            document = find_downloaded_version(engine, source, data_root=data_root)
            reused_pdf = document is not None
            if document is None:
                document = await asyncio.to_thread(
                    _download_and_register,
                    engine,
                    source,
                    run_dir,
                    data_root,
                    client,
                )
            document_id = document.id
            preparation = await asyncio.to_thread(
                lambda: asyncio.run(
                    prepare_document_version(engine, document.id, data_root=data_root)
                )
            )
            preparation_id = preparation.id
            status = "succeeded" if preparation.status == "succeeded" else "preparation_failed"
            if preparation.status == "failed":
                error_kind = preparation.error_kind
                error_message = preparation.error_message
        except InvalidDocumentSelection as error:
            status = "selection_failed"
            error_kind, error_message = type(error).__name__, str(error)
        except Exception as error:
            status = "download_failed" if document_id is None else "preparation_failed"
            error_kind = getattr(error, "kind", type(error).__name__)
            error_message = str(error)

        with Session(engine) as session:
            session.add(CollectionImportItem(
                id=uuid4().hex,
                import_run_id=run_id,
                position=position,
                source_ref=selected.source_ref,
                versioned_id=selected.versioned_id,
                status=status,
                document_id=document_id,
                preparation_id=preparation_id,
                reused_pdf=reused_pdf,
                error_kind=error_kind,
                error_message=error_message[:800] if error_message else None,
            ))
            session.commit()

    return StagedImport(run_id, collection_id, selection_revision, selections)


def publish_import_run(session: Session, staged: StagedImport, *, finished_at: str) -> str:
    """Finalize a complete staged report inside the caller's transaction."""

    run = session.get(CollectionImportRun, staged.run_id)
    if (
        run is None
        or run.collection_id != staged.collection_id
        or run.selection_revision != staged.selection_revision
        or run.status != "running"
    ):
        raise RuntimeError("The staged import run is unavailable for publication.")
    items = session.scalars(
        select(CollectionImportItem)
        .where(CollectionImportItem.import_run_id == staged.run_id)
        .order_by(CollectionImportItem.position)
    ).all()
    if len(items) != len(staged.selections):
        raise RuntimeError("The staged import report does not match its request.")
    for position, (item, selected) in enumerate(zip(items, staged.selections)):
        if (
            item.position != position
            or item.source_ref != selected.source_ref
            or item.versioned_id != selected.versioned_id
            or item.status not in (
                "succeeded", "selection_failed", "download_failed", "preparation_failed"
            )
        ):
            raise RuntimeError("The staged import report does not match its request.")
        if item.status == "succeeded":
            if item.document_id is None or item.preparation_id is None:
                raise RuntimeError("A successful import item has no matching prepared document.")
            document = session.get(DocumentVersion, item.document_id)
            preparation = session.get(DocumentPreparation, item.preparation_id)
            if (
                document is None
                or preparation is None
                or document.versioned_source_id != selected.versioned_id
                or preparation.document_id != document.id
                or preparation.status != "succeeded"
            ):
                raise RuntimeError("A successful import item has no matching prepared document.")
    statuses = [item.status for item in items]
    status = (
        "succeeded" if all(item == "succeeded" for item in statuses)
        else "partial_failed" if any(item == "succeeded" for item in statuses)
        else "failed"
    )
    published = session.execute(
        update(CollectionImportRun)
        .where(CollectionImportRun.id == staged.run_id, CollectionImportRun.status == "running")
        .values(status=status, finished_at=finished_at)
        .returning(CollectionImportRun.id)
    ).scalar_one_or_none()
    if published is None:
        raise RuntimeError("The staged import run was already published.")
    return status


def get_import_report(engine: Engine, run_id: str) -> ImportReport | None:
    with Session(engine) as session:
        run = session.get(CollectionImportRun, run_id)
        if run is None:
            return None
        items = session.scalars(
            select(CollectionImportItem)
            .where(CollectionImportItem.import_run_id == run_id)
            .order_by(CollectionImportItem.position)
        ).all()
        return ImportReport(
            id=run.id,
            collection_id=run.collection_id,
            selection_revision=run.selection_revision,
            status=run.status,
            items=tuple(ImportItemRecord(
                id=item.id,
                position=item.position,
                source_ref=item.source_ref,
                versioned_id=item.versioned_id,
                status=item.status,
                document_id=item.document_id,
                preparation_id=item.preparation_id,
                reused_pdf=item.reused_pdf,
                error_kind=item.error_kind,
                error_message=item.error_message,
            ) for item in items),
        )


def _download_and_register(
    engine: Engine,
    source: FixedVersionSource,
    run_dir: Path,
    data_root: Path,
    client: httpx.Client,
) -> DocumentVersionRecord:
    with stage_pdf_download(source, client=client, staging_dir=Path(data_root) / "staging") as staged:
        stored = publish_staged_pdf(staged, store_dir=Path(data_root) / "blobs")
    return register_downloaded_version(
        engine,
        run_dir=run_dir,
        versioned_id=source.versioned_id,
        stored=stored,
        data_root=data_root,
    )
