"""Check one downloaded PDF with PaperQA's text parser without building an index."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.ingestion.service import InvalidDocumentBlob, _checked_blob_ref
from backend.app.ingestion.storage import StoredPdf
from backend.app.models import DocumentPreparation, DocumentVersion

# PaperQA is imported below; keep its metadata lookup local, as in scripts/model_config.py.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from paperqa.readers import read_doc  # noqa: E402
from paperqa.settings import ParsingSettings  # noqa: E402
from paperqa.types import Doc  # noqa: E402
from paperqa.utils import maybe_is_text  # noqa: E402


class UnusableDocumentText(ValueError):
    """The PDF parses structurally but PaperQA cannot use its extracted text."""


@dataclass(frozen=True, slots=True)
class TextPreparationConfig:
    chunk_chars: int = 3000
    overlap: int = 200
    page_size_limit: int = 1_280_000

    def __post_init__(self) -> None:
        if self.chunk_chars < 1 or not 0 <= self.overlap < self.chunk_chars:
            raise ValueError("Text chunk size must exceed its nonnegative overlap.")
        if self.page_size_limit < 1:
            raise ValueError("Page size limit must be positive.")


@dataclass(frozen=True, slots=True)
class DocumentPreparationRecord:
    id: str
    document_id: str
    parser_config_hash: str
    parser_config_json: str
    status: str
    chunk_count: int | None
    error_kind: str | None
    error_message: str | None
    checked_at: str


async def prepare_document_version(
    engine: Engine,
    document_id: str,
    *,
    data_root: Path,
    config: TextPreparationConfig = TextPreparationConfig(),
) -> DocumentPreparationRecord:
    """Record text extraction for one parser configuration, separate from index state."""

    parsing = ParsingSettings(
        use_doc_details=False,
        multimodal=False,
        page_size_limit=config.page_size_limit,
        reader_config={"chunk_chars": config.chunk_chars, "overlap": config.overlap},
    )
    parser = parsing.parse_pdf
    parser_package = {
        "paperqa_pypdf": "paper-qa-pypdf",
        "paperqa_pymupdf": "paper-qa-pymupdf",
    }.get(parser.__module__.split(".", 1)[0])
    if parser_package is None:
        raise ValueError("Unsupported PaperQA PDF parser for this preparation check.")
    spec = {
        "schema": 1,
        "paperqa": version("paper-qa"),
        "pdf_parser": f"{parser.__module__}.{parser.__qualname__}",
        "pdf_parser_package": parser_package,
        "pdf_parser_version": version(parser_package),
        "pypdf": version("pypdf"),
        "chunk_chars": config.chunk_chars,
        "overlap": config.overlap,
        "page_size_limit": config.page_size_limit,
        "parse_media": False,
        "use_doc_details": False,
        "disable_doc_valid_check": False,
        "document_text_check": "paperqa-docs-aadd-v1",
    }
    spec_json = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    spec_hash = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()

    with Session(engine) as session:
        document = session.get(DocumentVersion, document_id)
        if document is None:
            raise LookupError(f"DocumentVersion {document_id!r} does not exist.")
        path = Path(data_root) / document.blob_ref
        stored = StoredPdf(
            versioned_id=document.versioned_source_id,
            path=path,
            sha256=document.sha256,
            byte_count=document.byte_count,
            page_count=document.page_count,
            reused=True,
        )
        if _checked_blob_ref(stored, data_root) != document.blob_ref:
            raise InvalidDocumentBlob("DocumentVersion blob reference is not canonical.")
        existing = session.scalar(
            select(DocumentPreparation).where(
                DocumentPreparation.document_id == document_id,
                DocumentPreparation.parser_config_hash == spec_hash,
            )
        )
        if existing is not None and existing.status == "succeeded":
            return _record(existing)
        doc = Doc(
            docname="P" + document.sha256[:10],
            dockey=document.versioned_source_id,
            citation=document.title,
            content_hash=document.sha256,
        )

    try:
        texts = await read_doc(
            path,
            doc,
            parse_pdf=parser,
            parse_media=False,
            page_size_limit=parsing.page_size_limit,
            **parsing.reader_config,
        )
        # Match the pinned Docs.aadd text checks before embeddings are attempted.
        if (
            not texts
            or len(texts[0].text) < 10
            or sum(len(text.text.replace("\n", "")) for text in texts[:2]) < 20
            or not maybe_is_text("".join(text.text for text in texts[:5]))
        ):
            raise UnusableDocumentText("PaperQA extracted no usable text from this PDF.")
        status, chunk_count, error_kind, error_message = "succeeded", len(texts), None, None
    except Exception as error:
        status, chunk_count = "failed", None
        error_kind = type(error).__name__
        error_message = str(error).replace(str(path), "[PDF]")[:800]

    # A replaced blob is an integrity error, not a parser failure for the old hash.
    if _checked_blob_ref(stored, data_root) != document.blob_ref:
        raise InvalidDocumentBlob("DocumentVersion blob reference is not canonical.")

    now = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    with Session(engine) as session:
        item = session.scalar(
            select(DocumentPreparation).where(
                DocumentPreparation.document_id == document_id,
                DocumentPreparation.parser_config_hash == spec_hash,
            )
        )
        if item is None:
            item = DocumentPreparation(
                id=uuid4().hex,
                document_id=document_id,
                parser_config_hash=spec_hash,
                parser_config_json=spec_json,
                status=status,
                chunk_count=chunk_count,
                error_kind=error_kind,
                error_message=error_message,
                checked_at=now,
            )
            session.add(item)
        elif item.status != "succeeded":
            item.status = status
            item.chunk_count = chunk_count
            item.error_kind = error_kind
            item.error_message = error_message
            item.checked_at = now
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            item = session.scalar(
                select(DocumentPreparation).where(
                    DocumentPreparation.document_id == document_id,
                    DocumentPreparation.parser_config_hash == spec_hash,
                )
            )
            if item is None:
                raise
        return _record(item)


def get_document_preparation(
    engine: Engine, document_id: str, parser_config_hash: str
) -> DocumentPreparationRecord | None:
    with Session(engine) as session:
        item = session.scalar(
            select(DocumentPreparation).where(
                DocumentPreparation.document_id == document_id,
                DocumentPreparation.parser_config_hash == parser_config_hash,
            )
        )
        return None if item is None else _record(item)


def _record(item: DocumentPreparation) -> DocumentPreparationRecord:
    return DocumentPreparationRecord(
        id=item.id,
        document_id=item.document_id,
        parser_config_hash=item.parser_config_hash,
        parser_config_json=item.parser_config_json,
        status=item.status,
        chunk_count=item.chunk_count,
        error_kind=item.error_kind,
        error_message=item.error_message,
        checked_at=item.checked_at,
    )
