"""Paginated metadata from existing records; never recover or rerun a task."""

import base64
import json
from urllib.parse import urlencode

from sqlalchemy import text

_HISTORY = """
SELECT 'search' kind, s.id, s.question, s.query_text, COALESCE(j.status, 'unknown') status,
       s.created_at, NULL previous_answer_run_id, NULL collection_id, NULL revision_no,
       NULL document_count
FROM search_runs s LEFT JOIN jobs j ON j.kind = 'search' AND j.resource_id = s.id
UNION ALL
SELECT 'answer', s.answer_run_id, COALESCE(a.question,
       CASE WHEN json_valid(r.request_json) THEN json_extract(r.request_json, '$.question') END, ''),
       '', CASE WHEN s.attempt_id IS NULL THEN
         CASE WHEN j.status = 'queued' AND j.resource_id = s.answer_run_id THEN 'queued' ELSE 'unknown' END
         ELSE COALESCE(t.status, 'unknown') END,
       s.submitted_at, s.previous_answer_run_id, NULL, NULL, NULL
FROM answer_submissions s LEFT JOIN answer_runs a ON a.id = s.answer_run_id
LEFT JOIN jobs j ON j.id = s.job_id
LEFT JOIN job_requests r ON r.job_id = s.job_id
LEFT JOIN job_attempts t ON t.id = s.attempt_id AND t.job_id = s.job_id
UNION ALL
SELECT 'answer', a.id, a.question, '',
       CASE WHEN a.job_attempt_id IS NULL THEN a.execution_status ELSE 'unknown' END,
       a.saved_at, NULL, NULL, NULL, NULL
FROM answer_runs a WHERE NOT EXISTS (SELECT 1 FROM answer_submissions s WHERE s.answer_run_id = a.id)
UNION ALL
SELECT 'collection', c.id, '', '', COALESCE(j.status, 'draft'), c.updated_at, NULL, c.id, NULL,
       CASE WHEN json_valid(c.selected_json) THEN json_array_length(c.selected_json) END
FROM collections c LEFT JOIN jobs j ON j.id = (
    SELECT id FROM jobs WHERE kind = 'import' AND resource_id = c.id ORDER BY created_at DESC, id DESC LIMIT 1)
UNION ALL
SELECT 'revision', c.id, '', '', COALESCE(j.status, 'confirmed'), c.confirmed_at, NULL,
       c.collection_id, c.revision_no,
       (SELECT COUNT(*) FROM collection_revision_documents d WHERE d.revision_id = c.id)
FROM collection_revisions c LEFT JOIN jobs j ON j.kind = 'index' AND j.resource_id = c.id
"""


def read_history(engine, *, kind="all", limit=25, cursor=None):
    if kind not in {"all", "search", "answer", "collection"} or not 1 <= limit <= 100:
        raise ValueError("历史类型或每页数量无效。")
    filters = []
    params = {"limit": limit + 1}
    if kind != "all":
        filters.append("kind IN ('collection', 'revision')" if kind == "collection" else "kind = :kind")
        params["kind"] = kind
    if cursor:
        try:
            if len(cursor) > 500:
                raise ValueError()
            timestamp, last_kind, last_id = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
            if not all(isinstance(value, str) for value in (timestamp, last_kind, last_id)):
                raise ValueError()
        except (ValueError, UnicodeError, TypeError):
            raise ValueError("历史分页链接无效，请重新打开历史记录。") from None
        filters.append("(created_at < :timestamp OR (created_at = :timestamp AND kind > :last_kind) OR (created_at = :timestamp AND kind = :last_kind AND id < :last_id))")
        params.update(timestamp=timestamp, last_kind=last_kind, last_id=last_id)
    where = " WHERE " + " AND ".join(filters) if filters else ""
    with engine.connect() as connection:
        rows = connection.execute(text("WITH history AS (" + _HISTORY + ") SELECT * FROM history" + where + " ORDER BY created_at DESC, kind ASC, id DESC LIMIT :limit"), params).mappings().all()
    items = [dict(row) for row in rows[:limit]]
    for item in items:
        identifiers = {"revision": item["id"], "collection": item["collection_id"]} if item["kind"] == "revision" else {item["kind"]: item["id"]}
        item["href"] = "/?" + urlencode(identifiers)
    next_cursor = None
    if len(rows) > limit:
        last = items[-1]
        next_cursor = base64.urlsafe_b64encode(json.dumps([last["created_at"], last["kind"], last["id"]]).encode()).decode()
    return {"items": items, "next_cursor": next_cursor}
