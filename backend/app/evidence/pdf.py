"""Read the exact local PDF bound to one saved answer evidence item."""

from bisect import bisect_right
import hashlib
from io import BytesIO
from pathlib import Path
import unicodedata

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from sqlalchemy.engine import Engine

from backend.app.ingestion.service import _checked_blob_ref, get_document_version
from backend.app.ingestion.storage import StoredPdf


class InvalidSavedPdf(ValueError):
    """Saved evidence no longer resolves to its verified fixed-version PDF."""


def _normalized_pdf_text(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKC", value).casefold()
        if char.isalnum()
    )


def locate_original_text_pages(pdf_bytes: bytes, original_text: str | None) -> dict:
    """Find complete original text in extracted PDF text; never infer a page."""

    unknown = {"status": "unknown", "matches": []}
    needle = _normalized_pdf_text(original_text or "")
    if len(needle) < 80:
        return unknown
    try:
        pages = [_normalized_pdf_text(page.extract_text() or "")
                 for page in PdfReader(BytesIO(pdf_bytes)).pages]
    except PdfReadError:
        return unknown
    starts = []
    offset = 0
    for page in pages:
        starts.append(offset)
        offset += len(page)
    joined = "".join(pages)
    matches = []
    search_from = 0
    while (position := joined.find(needle, search_from)) >= 0:
        first = bisect_right(starts, position) - 1
        last = bisect_right(starts, position + len(needle) - 1) - 1
        if all(pages[first:last + 1]):
            matches.append({"start_page": first + 1, "end_page": last + 1})
        search_from = position + 1
    if not matches:
        return unknown
    return {"status": "unique" if len(matches) == 1 else "multiple", "matches": matches}


def read_saved_evidence_pdf(
    engine: Engine, answer: dict, evidence_id: str, *, data_root: Path
) -> tuple[bytes, str] | None:
    """Return PDF bytes and content hash, or None for an unknown evidence ID."""

    matches = [item for item in answer["evidence"] if item["evidence_id"] == evidence_id]
    if not matches:
        return None
    if len(matches) != 1:
        raise InvalidSavedPdf("Evidence ID is ambiguous in the saved answer.")
    evidence = matches[0]
    document_id = evidence["document_id"]
    allowed = [item for item in answer["allowed_documents"] if item["document_id"] == document_id]
    if len(allowed) != 1:
        raise InvalidSavedPdf("Evidence document is not uniquely bound to the answer scope.")
    identity = allowed[0]
    document = get_document_version(engine, document_id)
    if (
        document is None or document.download_status != "downloaded"
        or identity["versioned_id"] != evidence["upstream_dockey"]
        or identity["dockey"] != evidence["upstream_dockey"]
        or document.versioned_id != identity["versioned_id"]
        or document.sha256 != identity["sha256"]
    ):
        raise InvalidSavedPdf("PDF identity differs from the saved answer.")

    path = Path(data_root) / document.blob_ref
    stored = StoredPdf(
        versioned_id=document.versioned_id, path=path, sha256=document.sha256,
        byte_count=document.byte_count, page_count=document.page_count, reused=True,
    )
    if _checked_blob_ref(stored, data_root) != document.blob_ref:
        raise InvalidSavedPdf("PDF blob reference is not canonical.")
    # Verify the bytes returned to the browser after the path check as well.
    body = path.read_bytes()
    if len(body) != document.byte_count or hashlib.sha256(body).hexdigest() != document.sha256:
        raise InvalidSavedPdf("PDF bytes changed while being read.")
    return body, document.sha256
