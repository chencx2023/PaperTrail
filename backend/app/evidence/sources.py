"""Read frozen paper identities and abstracts without loading mutable Paper metadata."""

from datetime import datetime
import json
from pathlib import Path

from backend.app.models import DocumentVersion
from backend.app.search.service import InvalidSearchSnapshot, replay_search_run


def frozen_document_sources(session, allowed_documents, *, data_root: Path) -> dict:
    sources, snapshots = {}, {}
    runs = (Path(data_root).resolve() / 'runs').resolve()
    for allowed in allowed_documents:
        document = session.get(DocumentVersion, allowed.get('document_id'))
        if (document is None or document.versioned_source_id != allowed.get('versioned_id')
                or document.sha256 != allowed.get('sha256')):
            continue
        try:
            authors = json.loads(document.authors_json)
            if not isinstance(authors, list) or any(not isinstance(name, str) or not name.strip() for name in authors):
                authors = []
        except (TypeError, ValueError):
            authors = []
        try:
            year = datetime.fromisoformat(document.published_at.replace('Z', '+00:00')).year
        except (TypeError, ValueError):
            year = None
        source = {'document_id': document.id, 'versioned_id': document.versioned_source_id,
            'title': document.title, 'authors': authors, 'year': year,
            'published_at': document.published_at, 'abstract': None, 'abstract_status': 'unavailable'}
        run_id = document.source_run_id
        path = (runs / run_id).resolve()
        if path.parent != runs:
            source['abstract_status'] = 'unverified'
        elif path.is_dir():
            if run_id not in snapshots:
                try:
                    snapshots[run_id] = replay_search_run(path)
                except (InvalidSearchSnapshot, ValueError, OSError):
                    snapshots[run_id] = None
            snapshot = snapshots[run_id]
            if snapshot and snapshot.candidates_match and snapshot.snapshot_sha256 == document.snapshot_sha256:
                records = [r for r in snapshot.feed.records if r.versioned_id == document.versioned_source_id]
                if len(records) == 1:
                    record = records[0]
                    if record.title == document.title and list(record.authors) == authors and record.published_at == document.published_at:
                        source.update(abstract=record.abstract or None, abstract_status='saved' if record.abstract else 'unavailable')
                    else:
                        source['abstract_status'] = 'unverified'
                else:
                    source['abstract_status'] = 'unverified'
            else:
                source['abstract_status'] = 'unverified'
        sources[document.id] = source
    return sources
