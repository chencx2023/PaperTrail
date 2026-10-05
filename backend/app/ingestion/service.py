"""Persist a verified download separately from parsing and index readiness."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.ingestion.selection import FixedVersionSource, select_fixed_version
from backend.app.ingestion.storage import StoredPdf, _hash_file
from backend.app.models import DocumentVersion, Paper


class InvalidDocumentBlob(ValueError):
    """The file cannot be safely referenced as a published PDF blob."""


class VersionContentConflict(ValueError):
    """One fixed source version was already registered with different bytes."""


@dataclass(frozen=True, slots=True)
class DocumentVersionRecord:
    id: str
    paper_id: str
    versioned_id: str
    source_run_id: str
    snapshot_sha256: str
    source_url: str
    title: str
    authors: tuple[str, ...]
    published_at: str
    updated_at: str
    sha256: str
    blob_ref: str
    byte_count: int
    page_count: int
    download_status: str


def register_downloaded_version(
    engine: Engine,
    *,
    run_dir: Path,
    versioned_id: str,
    stored: StoredPdf,
    data_root: Path,
) -> DocumentVersionRecord:
    """Register one checked snapshot and blob; never replace a version's bytes."""

    source = select_fixed_version(run_dir, versioned_id)
    if stored.versioned_id != source.versioned_id:
        raise InvalidDocumentBlob("Published PDF belongs to another source version.")
    blob_ref = _checked_blob_ref(stored, data_root)
    for attempt in range(2):
        with Session(engine) as session:
            try:
                paper = session.scalar(
                    select(Paper).where(
                        Paper.source == "arxiv", Paper.source_id == source.arxiv_id
                    )
                )
                if paper is None:
                    paper = Paper(
                        id=uuid4().hex,
                        source="arxiv",
                        source_id=source.arxiv_id,
                        title=source.title,
                        authors_json=json.dumps(source.authors, ensure_ascii=False),
                        published_at=source.published_at,
                        updated_at=source.updated_at,
                    )
                    session.add(paper)
                    session.flush()

                existing = session.scalar(
                    select(DocumentVersion).where(
                        DocumentVersion.paper_id == paper.id,
                        DocumentVersion.versioned_source_id == source.versioned_id,
                    )
                )
                if existing is not None:
                    if existing.sha256 != stored.sha256:
                        raise VersionContentConflict(
                            "The selected version already has a different PDF hash."
                        )
                    existing_stored = StoredPdf(
                        versioned_id=existing.versioned_source_id,
                        path=Path(data_root) / existing.blob_ref,
                        sha256=existing.sha256,
                        byte_count=existing.byte_count,
                        page_count=existing.page_count,
                        reused=True,
                    )
                    if _checked_blob_ref(existing_stored, data_root) != existing.blob_ref:
                        raise InvalidDocumentBlob("Existing blob reference is not canonical.")
                    return _record(existing)

                document = DocumentVersion(
                    id=uuid4().hex,
                    paper_id=paper.id,
                    versioned_source_id=source.versioned_id,
                    source_run_id=source.source_run_id,
                    snapshot_sha256=source.snapshot_sha256,
                    source_url=source.source_pdf_url,
                    title=source.title,
                    authors_json=json.dumps(source.authors, ensure_ascii=False),
                    published_at=source.published_at,
                    updated_at=source.updated_at,
                    sha256=stored.sha256,
                    blob_ref=blob_ref,
                    byte_count=stored.byte_count,
                    page_count=stored.page_count,
                    download_status="downloaded",
                    created_at=datetime.now(timezone.utc)
                    .isoformat(timespec="microseconds")
                    .replace("+00:00", "Z"),
                )
                session.add(document)
                session.commit()
                return _record(document)
            except IntegrityError:
                session.rollback()
                if attempt == 1:
                    raise
    raise RuntimeError("Could not register a document version.")


def get_document_version(
    engine: Engine, document_id: str
) -> DocumentVersionRecord | None:
    with Session(engine) as session:
        document = session.get(DocumentVersion, document_id)
        return None if document is None else _record(document)


def find_downloaded_version(
    engine: Engine, source: FixedVersionSource, *, data_root: Path
) -> DocumentVersionRecord | None:
    """Reuse a fixed version only while its published bytes still match the record."""

    with Session(engine) as session:
        document = session.scalar(
            select(DocumentVersion)
            .join(Paper, Paper.id == DocumentVersion.paper_id)
            .where(
                Paper.source == "arxiv",
                Paper.source_id == source.arxiv_id,
                DocumentVersion.versioned_source_id == source.versioned_id,
            )
        )
        if document is None:
            return None
        stored = StoredPdf(
            versioned_id=document.versioned_source_id,
            path=Path(data_root) / document.blob_ref,
            sha256=document.sha256,
            byte_count=document.byte_count,
            page_count=document.page_count,
            reused=True,
        )
        if _checked_blob_ref(stored, data_root) != document.blob_ref:
            raise InvalidDocumentBlob("Existing PDF blob reference is not canonical.")
        return _record(document)


def _checked_blob_ref(stored: StoredPdf, data_root: Path) -> str:
    path = Path(stored.path)
    if stored.byte_count < 1 or stored.page_count < 1 or path.is_symlink() or not path.is_file():
        raise InvalidDocumentBlob("The published PDF is missing or invalid.")
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(Path(data_root).resolve(strict=True))
    except (OSError, ValueError) as error:
        raise InvalidDocumentBlob("PDF blob is outside the data root.") from error
    if resolved.name != f"{stored.sha256}.pdf":
        raise InvalidDocumentBlob("PDF filename does not match its hash.")
    actual_hash, actual_size = _hash_file(resolved)
    if actual_hash != stored.sha256 or actual_size != stored.byte_count:
        raise InvalidDocumentBlob("Published PDF bytes changed.")
    return relative.as_posix()


def _record(document: DocumentVersion) -> DocumentVersionRecord:
    return DocumentVersionRecord(
        id=document.id,
        paper_id=document.paper_id,
        versioned_id=document.versioned_source_id,
        source_run_id=document.source_run_id,
        snapshot_sha256=document.snapshot_sha256,
        source_url=document.source_url,
        title=document.title,
        authors=tuple(json.loads(document.authors_json)),
        published_at=document.published_at,
        updated_at=document.updated_at,
        sha256=document.sha256,
        blob_ref=document.blob_ref,
        byte_count=document.byte_count,
        page_count=document.page_count,
        download_status=document.download_status,
    )
