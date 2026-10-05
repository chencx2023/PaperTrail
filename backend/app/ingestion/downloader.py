"""Stage bounded PDF bytes; parsing and permanent publication happen later."""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile
import time
from typing import Iterator
from urllib.parse import urljoin, urlsplit

import httpx

from backend.app.ingestion.selection import (
    FixedVersionSource,
    InvalidDocumentSelection,
    _checked_pdf_url,
)


DEFAULT_MAX_PDF_BYTES = 30 * 1024 * 1024
DEFAULT_PDF_TIMEOUT_SECONDS = 120.0
MAX_PDF_REDIRECTS = 1


class PdfStageError(RuntimeError):
    """A PDF response could not be safely staged for later parsing."""

    def __init__(
        self, kind: str, message: str, *, status_code: int | None = None
    ) -> None:
        self.kind = kind
        self.status_code = status_code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class StagedPdf:
    versioned_id: str
    path: Path
    sha256: str
    byte_count: int


def _checked_redirect_url(
    current_url: str, location: str | None, versioned_id: str, status_code: int
) -> str:
    if not location or any(char.isspace() or char == "\\" for char in location):
        raise PdfStageError(
            "redirect", "PDF redirect has no valid destination.", status_code=status_code
        )
    try:
        parsed = urlsplit(urljoin(current_url, location))
    except ValueError as error:
        raise PdfStageError(
            "redirect", "PDF redirect destination is malformed.", status_code=status_code
        ) from error
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "arxiv.org"
        or parsed.path not in (f"/pdf/{versioned_id}", f"/pdf/{versioned_id}.pdf")
        or parsed.query
        or parsed.fragment
    ):
        raise PdfStageError(
            "redirect",
            "PDF redirect must stay on the selected arXiv version.",
            status_code=status_code,
        )
    return f"https://arxiv.org{parsed.path}"


@contextmanager
def stage_pdf_download(
    source: FixedVersionSource,
    *,
    client: httpx.Client,
    staging_dir: Path,
    max_bytes: int = DEFAULT_MAX_PDF_BYTES,
    timeout_seconds: float = DEFAULT_PDF_TIMEOUT_SECONDS,
) -> Iterator[StagedPdf]:
    """Yield a temporary PDF-looking file and remove it on context exit.

    This does not make the file a parsed or published DocumentVersion.
    """

    if max_bytes < 1 or timeout_seconds <= 0:
        raise ValueError("PDF size and timeout limits must be positive.")
    try:
        checked_url = _checked_pdf_url(source.source_pdf_url, source.versioned_id)
    except InvalidDocumentSelection as error:
        raise PdfStageError("invalid_source", str(error)) from error
    if source.download_url != checked_url:
        raise PdfStageError("invalid_source", "The download target was changed.")

    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    deadline = time.monotonic() + timeout_seconds
    current_url = checked_url
    visited = {checked_url}
    redirect_count = 0
    try:
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PdfStageError("timeout", "PDF download time limit expired.")
                with client.stream(
                    "GET",
                    current_url,
                    headers={"User-Agent": "PaperTrail/0.1"},
                    timeout=remaining,
                    follow_redirects=False,
                ) as response:
                    if 300 <= response.status_code < 400:
                        target = _checked_redirect_url(
                            current_url,
                            response.headers.get("Location"),
                            source.versioned_id,
                            response.status_code,
                        )
                        if target in visited or redirect_count >= MAX_PDF_REDIRECTS:
                            raise PdfStageError(
                                "redirect",
                                "PDF redirect loop or hop limit reached.",
                                status_code=response.status_code,
                            )
                        visited.add(target)
                        redirect_count += 1
                        current_url = target
                        continue
                    if response.status_code != 200:
                        raise PdfStageError(
                            "http_error",
                            f"PDF source returned HTTP {response.status_code}.",
                            status_code=response.status_code,
                        )
                    content_type = response.headers.get("Content-Type", "")
                    if content_type.split(";", 1)[0].strip().lower() != "application/pdf":
                        raise PdfStageError(
                            "invalid_content_type", "PDF source did not return application/pdf."
                        )

                    digest = hashlib.sha256()
                    byte_count = 0
                    signature = bytearray()
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        prefix=".pdf-download-",
                        suffix=".part",
                        dir=staging_dir,
                        delete=False,
                    ) as output:
                        temporary_path = Path(output.name)
                        for chunk in response.iter_bytes():
                            if time.monotonic() >= deadline:
                                raise PdfStageError(
                                    "timeout", "PDF download time limit expired."
                                )
                            byte_count += len(chunk)
                            if byte_count > max_bytes:
                                raise PdfStageError(
                                    "too_large", "PDF exceeds the configured size limit."
                                )
                            output.write(chunk)
                            digest.update(chunk)
                            if len(signature) < 5:
                                signature.extend(chunk[: 5 - len(signature)])

                    if signature != b"%PDF-":
                        raise PdfStageError(
                            "invalid_signature", "PDF bytes lack the expected file signature."
                        )
                    break
        except httpx.RequestError as error:
            raise PdfStageError("transport_error", str(error)) from error

        yield StagedPdf(
            versioned_id=source.versioned_id,
            path=temporary_path,
            sha256=digest.hexdigest(),
            byte_count=byte_count,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
