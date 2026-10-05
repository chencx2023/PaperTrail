"""Parse arXiv Atom responses into deterministic application records."""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urlsplit
import xml.etree.ElementTree as ElementTree


ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
OPENSEARCH_NAMESPACE = "http://a9.com/-/spec/opensearch/1.1/"
ARXIV_NAMESPACE = "http://arxiv.org/schemas/atom"
ARXIV_PARSER_VERSION = 2

_ATOM = f"{{{ATOM_NAMESPACE}}}"
_OPENSEARCH = f"{{{OPENSEARCH_NAMESPACE}}}"
_ARXIV = f"{{{ARXIV_NAMESPACE}}}"
_ARXIV_IDENTIFIER = re.compile(
    r"(?P<base>(?:\d{4}\.\d{4,5}|[A-Za-z0-9][A-Za-z0-9.-]*/\d{7}))"
    r"(?:v(?P<version>[1-9]\d*))?\Z"
)


class InvalidArxivFeed(ValueError):
    """Raised when an Atom payload cannot produce trustworthy candidates."""


class ArxivAPIError(RuntimeError):
    """Raised when arXiv returns its documented Atom error entry."""

    def __init__(
        self,
        error_id: str,
        message: str,
        detail_url: str | None = None,
    ) -> None:
        self.error_id = error_id
        self.detail_url = detail_url
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ArxivVersionRef:
    versioned_id: str
    version: int | None
    source_rank: int


@dataclass(frozen=True, slots=True)
class ArxivRecord:
    source_rank: int
    arxiv_id: str
    versioned_id: str
    version: int | None
    entry_id_url: str
    abstract_url: str
    pdf_url: str | None
    title: str
    abstract: str
    authors: tuple[str, ...]
    published_at: str
    updated_at: str
    primary_category: str | None
    categories: tuple[str, ...]
    comment: str | None
    journal_ref: str | None
    doi: str | None
    license_url: str | None


@dataclass(frozen=True, slots=True)
class ArxivCandidate(ArxivRecord):
    display_rank: int
    versions: tuple[ArxivVersionRef, ...]


@dataclass(frozen=True, slots=True)
class ParsedArxivFeed:
    feed_title: str
    feed_id: str
    feed_updated_at: str
    self_url: str
    total_results: int
    start_index: int
    items_per_page: int
    records: tuple[ArxivRecord, ...]
    candidates: tuple[ArxivCandidate, ...]


def parse_atom_feed(payload: bytes | str) -> ParsedArxivFeed:
    """Parse one saved arXiv Atom response without performing network I/O."""

    try:
        root = ElementTree.fromstring(payload)
    except (ElementTree.ParseError, TypeError, ValueError) as error:
        raise InvalidArxivFeed("The response is not valid XML.") from error

    if root.tag != f"{_ATOM}feed":
        raise InvalidArxivFeed("The response root is not an Atom feed.")

    entries = root.findall(f"{_ATOM}entry")
    _raise_documented_api_error(entries)

    feed_title = _required_text(root, f"{_ATOM}title", "feed title")
    feed_id = _required_text(root, f"{_ATOM}id", "feed id")
    feed_updated_at = _required_text(root, f"{_ATOM}updated", "feed updated")
    self_url = _find_feed_self_url(root)
    if self_url is None:
        raise InvalidArxivFeed("The feed is missing its self link.")

    total_results = _required_nonnegative_int(
        root, f"{_OPENSEARCH}totalResults", "totalResults"
    )
    start_index = _required_nonnegative_int(
        root, f"{_OPENSEARCH}startIndex", "startIndex"
    )
    items_per_page = _required_nonnegative_int(
        root, f"{_OPENSEARCH}itemsPerPage", "itemsPerPage"
    )

    records: list[ArxivRecord] = []
    for source_rank, entry in enumerate(entries, start=1):
        try:
            records.append(_parse_record(entry, source_rank))
        except InvalidArxivFeed as error:
            raise InvalidArxivFeed(f"Entry {source_rank}: {error}") from error

    record_tuple = tuple(records)
    return ParsedArxivFeed(
        feed_title=feed_title,
        feed_id=feed_id,
        feed_updated_at=feed_updated_at,
        self_url=self_url,
        total_results=total_results,
        start_index=start_index,
        items_per_page=items_per_page,
        records=record_tuple,
        candidates=_deduplicate(record_tuple),
    )


def _raise_documented_api_error(entries: list[ElementTree.Element]) -> None:
    for entry in entries:
        entry_id = _optional_text(entry, f"{_ATOM}id")
        if entry_id is None or "/api/errors#" not in entry_id:
            continue
        error_id = entry_id.partition("#")[2] or "unknown_arxiv_error"
        message = (
            _optional_text(entry, f"{_ATOM}summary")
            or _optional_text(entry, f"{_ATOM}title")
            or "arXiv returned an unspecified API error."
        )
        raise ArxivAPIError(
            error_id=error_id,
            message=message,
            detail_url=_find_link(entry, rel="alternate"),
        )


def _parse_record(entry: ElementTree.Element, source_rank: int) -> ArxivRecord:
    entry_id_url = _required_text(entry, f"{_ATOM}id", "entry id")
    arxiv_id, id_version, id_versioned = _parse_identifier(entry_id_url)

    abstract_url = _find_link(entry, rel="alternate") or entry_id_url
    link_arxiv_id, link_version, link_versioned = _parse_identifier(abstract_url)
    if link_arxiv_id != arxiv_id:
        raise InvalidArxivFeed("The entry id and abstract link identify different papers.")

    version = link_version if link_version is not None else id_version
    versioned_id = link_versioned if link_version is not None else id_versioned
    authors = tuple(
        name
        for author in entry.findall(f"{_ATOM}author")
        if (name := _optional_text(author, f"{_ATOM}name")) is not None
    )
    if not authors:
        raise InvalidArxivFeed("The entry has no author names.")

    categories = tuple(
        term
        for category in entry.findall(f"{_ATOM}category")
        if (term := _clean(category.get("term"))) is not None
    )
    primary_category_element = entry.find(f"{_ARXIV}primary_category")
    primary_category = (
        _clean(primary_category_element.get("term"))
        if primary_category_element is not None
        else None
    )

    return ArxivRecord(
        source_rank=source_rank,
        arxiv_id=arxiv_id,
        versioned_id=versioned_id,
        version=version,
        entry_id_url=entry_id_url,
        abstract_url=abstract_url,
        pdf_url=_find_link(entry, rel="related", title="pdf"),
        title=_required_text(entry, f"{_ATOM}title", "title"),
        abstract=_required_text(entry, f"{_ATOM}summary", "summary"),
        authors=authors,
        published_at=_required_text(entry, f"{_ATOM}published", "published"),
        updated_at=_required_text(entry, f"{_ATOM}updated", "updated"),
        primary_category=primary_category,
        categories=categories,
        comment=_optional_text(entry, f"{_ARXIV}comment"),
        journal_ref=_optional_text(entry, f"{_ARXIV}journal_ref"),
        doi=_optional_text(entry, f"{_ARXIV}doi"),
        license_url=_optional_text(entry, f"{_ARXIV}license"),
    )


def _parse_identifier(value: str) -> tuple[str, int | None, str]:
    path = urlsplit(value).path
    marker = "/abs/"
    if marker not in path:
        raise InvalidArxivFeed(f"Not an arXiv abstract URL: {value}")
    identifier = path.partition(marker)[2].rstrip("/")
    match = _ARXIV_IDENTIFIER.fullmatch(identifier)
    if match is None:
        raise InvalidArxivFeed(f"Invalid arXiv identifier: {identifier}")
    version_text = match.group("version")
    return match.group("base"), int(version_text) if version_text else None, identifier


def _deduplicate(records: tuple[ArxivRecord, ...]) -> tuple[ArxivCandidate, ...]:
    groups: dict[str, list[ArxivRecord]] = {}
    for record in records:
        groups.setdefault(record.arxiv_id, []).append(record)

    candidates: list[ArxivCandidate] = []
    for display_rank, group in enumerate(groups.values(), start=1):
        selected = max(group, key=lambda item: item.version or 0)
        candidates.append(
            ArxivCandidate(
                source_rank=selected.source_rank,
                arxiv_id=selected.arxiv_id,
                versioned_id=selected.versioned_id,
                version=selected.version,
                entry_id_url=selected.entry_id_url,
                abstract_url=selected.abstract_url,
                pdf_url=selected.pdf_url,
                title=selected.title,
                abstract=selected.abstract,
                authors=selected.authors,
                published_at=selected.published_at,
                updated_at=selected.updated_at,
                primary_category=selected.primary_category,
                categories=selected.categories,
                comment=selected.comment,
                journal_ref=selected.journal_ref,
                doi=selected.doi,
                license_url=selected.license_url,
                display_rank=display_rank,
                versions=tuple(
                    ArxivVersionRef(
                        versioned_id=record.versioned_id,
                        version=record.version,
                        source_rank=record.source_rank,
                    )
                    for record in group
                ),
            )
        )
    return tuple(candidates)


def _required_text(
    parent: ElementTree.Element,
    path: str,
    field_name: str,
) -> str:
    value = _optional_text(parent, path)
    if value is None:
        raise InvalidArxivFeed(f"Missing required {field_name}.")
    return value


def _optional_text(parent: ElementTree.Element, path: str) -> str | None:
    element = parent.find(path)
    if element is None:
        return None
    return _clean("".join(element.itertext()))


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _required_nonnegative_int(
    parent: ElementTree.Element,
    path: str,
    field_name: str,
) -> int:
    value = _required_text(parent, path, field_name)
    try:
        number = int(value)
    except ValueError as error:
        raise InvalidArxivFeed(f"{field_name} is not an integer.") from error
    if number < 0:
        raise InvalidArxivFeed(f"{field_name} cannot be negative.")
    return number


def _find_link(
    parent: ElementTree.Element,
    *,
    rel: str,
    title: str | None = None,
) -> str | None:
    for link in parent.findall(f"{_ATOM}link"):
        if link.get("rel") != rel:
            continue
        if title is not None and link.get("title") != title:
            continue
        href = _clean(link.get("href"))
        if href is not None:
            return href
    return None


def _find_feed_self_url(feed: ElementTree.Element) -> str | None:
    standard = _find_link(feed, rel="self")
    if standard is not None:
        return standard

    current_style_links = [
        href
        for link in feed.findall(f"{_ATOM}link")
        if link.get("rel") is None
        and link.get("type") == "application/atom+xml"
        and (href := _clean(link.get("href"))) is not None
    ]
    if len(current_style_links) == 1:
        return current_style_links[0]
    return None
