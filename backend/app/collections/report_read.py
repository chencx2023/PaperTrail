"""Read only the report published by a collection's current import attempt."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.jobs.import_submission import ImportJobRequest
from backend.app.models import (
    CollectionImportItem,
    CollectionImportRun,
    DocumentPreparation,
    DocumentVersion,
    Job,
    JobAttempt,
    JobAttemptResult,
)


class InvalidCurrentImportReport(ValueError):
    """A successful import Job lacks a valid current result."""


_FAILURE_REASONS = {
    "selection_failed": "保存的论文来源无法核验，无法导入。",
    "download_failed": "PDF 下载失败，可稍后再次导入。",
    "preparation_failed": "PDF 已下载，但正文解析失败。",
}


def _failure_reason(row: CollectionImportItem) -> str | None:
    if (row.status == "download_failed" and row.error_kind == "transport_error"
            and "[WinError 10013]" in (row.error_message or "")):
        return "运行环境禁止联网，PDF 无法下载。请以正常网络权限重新启动后台，再次导入。"
    # Return known explanations, never raw provider text, paths or credentials.
    return _FAILURE_REASONS.get(row.status)


def read_current_import_report(
    session: Session, job: Job, request: ImportJobRequest
) -> dict | None:
    """Require the active attempt's binding and the exact submitted item list."""

    if job.status != "succeeded":
        return None
    attempt = session.get(JobAttempt, job.active_attempt_id) if job.active_attempt_id else None
    binding = session.get(JobAttemptResult, attempt.id) if attempt is not None else None
    run = session.get(CollectionImportRun, binding.resource_id) if binding is not None else None
    if (
        attempt is None
        or attempt.job_id != job.id
        or attempt.status != "succeeded"
        or attempt.artifact_ref is not None
        or binding is None
        or binding.resource_id != attempt.id
        or binding.artifact_ref is not None
        or run is None
        or run.collection_id != request.collection_id
        or run.selection_revision != request.selection_revision
        or run.status not in ("succeeded", "partial_failed", "failed")
        or run.finished_at is None
    ):
        raise InvalidCurrentImportReport("Current import report is unavailable.")

    rows = session.scalars(
        select(CollectionImportItem)
        .where(CollectionImportItem.import_run_id == run.id)
        .order_by(CollectionImportItem.position)
    ).all()
    if len(rows) != len(request.selections):
        raise InvalidCurrentImportReport("Current import report is incomplete.")

    shown = []
    for position, (row, selected) in enumerate(zip(rows, request.selections, strict=True)):
        if (
            row.position != position
            or row.source_ref != selected.source_ref
            or row.versioned_id != selected.versioned_id
            or row.status not in ("succeeded", *_FAILURE_REASONS)
        ):
            raise InvalidCurrentImportReport("Current import items do not match the request.")
        title = None
        if row.status == "succeeded":
            document = session.get(DocumentVersion, row.document_id) if row.document_id else None
            preparation = (
                session.get(DocumentPreparation, row.preparation_id)
                if row.preparation_id else None
            )
            if (
                document is None
                or preparation is None
                or document.versioned_source_id != selected.versioned_id
                or preparation.document_id != document.id
                or preparation.status != "succeeded"
            ):
                raise InvalidCurrentImportReport("A successful import item is unavailable.")
            title = document.title
        shown.append(
            {
                "position": position,
                "versioned_id": row.versioned_id,
                "title": title,
                "status": row.status,
                "document_id": row.document_id if row.status == "succeeded" else None,
                "reused_pdf": row.reused_pdf,
                "failure_reason": _failure_reason(row),
            }
        )

    statuses = [row.status for row in rows]
    expected_status = (
        "succeeded" if all(status == "succeeded" for status in statuses)
        else "partial_failed" if any(status == "succeeded" for status in statuses)
        else "failed"
    )
    if run.status != expected_status:
        raise InvalidCurrentImportReport("Current import status disagrees with its items.")
    return {
        "import_run_id": run.id,
        "selection_revision": run.selection_revision,
        "status": run.status,
        "items": shown,
    }
