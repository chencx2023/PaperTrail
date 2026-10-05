"""Resolve one saved arXiv search record to a fixed PDF version."""

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from backend.app.search.service import InvalidSearchSnapshot, replay_search_run


class InvalidDocumentSelection(ValueError):
    """A saved search does not justify downloading the requested version."""


@dataclass(frozen=True, slots=True)
class FixedVersionSource:
    source_run_id: str
    snapshot_sha256: str
    arxiv_id: str
    versioned_id: str
    version: int
    source_pdf_url: str
    download_url: str
    title: str
    authors: tuple[str, ...]
    published_at: str
    updated_at: str


def select_fixed_version(run_dir: Path, versioned_id: str) -> FixedVersionSource:
    """Check a saved search and return only the requested observed version."""

    try:
        replay = replay_search_run(run_dir)
    except InvalidSearchSnapshot as error:
        raise InvalidDocumentSelection(f"Invalid saved search: {error}") from error
    if not replay.candidates_match:
        raise InvalidDocumentSelection("Saved candidates differ from the Atom snapshot.")

    matches = [
        record
        for record in replay.feed.records
        if record.versioned_id == versioned_id
    ]
    if len(matches) != 1 or matches[0].version is None:
        raise InvalidDocumentSelection(
            "The requested explicit version is not in this saved search."
        )
    record = matches[0]
    download_url = _checked_pdf_url(record.pdf_url, versioned_id)
    return FixedVersionSource(
        source_run_id=replay.source_run_id,
        snapshot_sha256=replay.snapshot_sha256,
        arxiv_id=record.arxiv_id,
        versioned_id=record.versioned_id,
        version=record.version,
        source_pdf_url=record.pdf_url,
        download_url=download_url,
        title=record.title,
        authors=record.authors,
        published_at=record.published_at,
        updated_at=record.updated_at,
    )


def _checked_pdf_url(source_url: str | None, versioned_id: str) -> str:
    if source_url is None:
        raise InvalidDocumentSelection("The selected version has no PDF link.")
    try:
        parsed = urlsplit(source_url)
    except ValueError as error:
        raise InvalidDocumentSelection("The selected PDF link is invalid.") from error
    expected_path = f"/pdf/{versioned_id}"
    if (
        parsed.scheme not in ("http", "https")
        or parsed.netloc.lower() != "arxiv.org"
        or parsed.path != expected_path
        or parsed.query
        or parsed.fragment
    ):
        raise InvalidDocumentSelection(
            "The selected PDF link does not match its arXiv version."
        )
    return f"https://arxiv.org{expected_path}"
