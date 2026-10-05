"""Parse and atomically publish immutable PDF blobs by their SHA-256."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from backend.app.ingestion.downloader import StagedPdf


class InvalidPdfContent(ValueError):
    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        super().__init__(message)


class CorruptStoredPdf(RuntimeError):
    """A content-addressed file exists but no longer matches its name."""


@dataclass(frozen=True, slots=True)
class StoredPdf:
    versioned_id: str
    path: Path
    sha256: str
    byte_count: int
    page_count: int
    reused: bool


def publish_staged_pdf(staged: StagedPdf, *, store_dir: Path) -> StoredPdf:
    """Check basic structure before making a PDF blob visible under its hash."""

    actual_hash, actual_size = _hash_file(staged.path)
    if actual_hash != staged.sha256 or actual_size != staged.byte_count:
        raise InvalidPdfContent("staged_file_changed", "Staged PDF bytes changed.")

    try:
        with staged.path.open("rb") as source:
            reader = PdfReader(source, strict=False)
            if reader.is_encrypted:
                raise InvalidPdfContent(
                    "encrypted", "Encrypted PDFs cannot be prepared without a password."
                )
            page_count = len(reader.pages)
    except (PyPdfError, ValueError) as error:
        if isinstance(error, InvalidPdfContent):
            raise
        raise InvalidPdfContent("parse_error", f"PDF cannot be parsed: {error}") from error
    if page_count < 1:
        raise InvalidPdfContent("no_pages", "PDF contains no pages.")

    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    final_path = store_dir / f"{staged.sha256}.pdf"
    if final_path.exists() or final_path.is_symlink():
        _check_existing(final_path, staged.sha256, staged.byte_count)
        return StoredPdf(
            staged.versioned_id, final_path, staged.sha256, staged.byte_count, page_count, True
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".pdf-publish-",
            suffix=".tmp",
            dir=store_dir,
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            copied_hash = hashlib.sha256()
            copied_size = 0
            with staged.path.open("rb") as source:
                for chunk in iter(lambda: source.read(64 * 1024), b""):
                    output.write(chunk)
                    copied_hash.update(chunk)
                    copied_size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if copied_hash.hexdigest() != staged.sha256 or copied_size != staged.byte_count:
            raise InvalidPdfContent("staged_file_changed", "Staged PDF bytes changed.")
        try:
            # The completed temporary file becomes visible without replacing a peer.
            os.link(temporary_path, final_path)
        except FileExistsError:
            _check_existing(final_path, staged.sha256, staged.byte_count)
            return StoredPdf(
                staged.versioned_id,
                final_path,
                staged.sha256,
                staged.byte_count,
                page_count,
                True,
            )
        return StoredPdf(
            staged.versioned_id,
            final_path,
            staged.sha256,
            staged.byte_count,
            page_count,
            False,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(64 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _check_existing(path: Path, expected_hash: str, expected_size: int) -> None:
    if path.is_symlink() or not path.is_file():
        raise CorruptStoredPdf(f"Stored PDF path is not a regular file: {path}")
    actual_hash, actual_size = _hash_file(path)
    if actual_hash != expected_hash or actual_size != expected_size:
        raise CorruptStoredPdf(f"Stored PDF does not match its hash: {path}")
