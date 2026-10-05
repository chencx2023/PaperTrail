"""Read-only view of the execution queue and cooperating worker."""

from datetime import datetime, timezone
from sqlalchemy import text

from backend.app.jobs.process_lock import worker_status


def read_runtime(engine, database_path):
    with engine.connect() as connection:
        queued = connection.execute(text("SELECT COUNT(*) FROM jobs WHERE status = 'queued'")).scalar_one()
        active = connection.execute(text(
            "SELECT j.id AS job_id, j.kind, a.started_at, a.heartbeat_at, a.lease_until "
            "FROM jobs j LEFT JOIN job_attempts a ON a.id = j.active_attempt_id "
            "WHERE j.status = 'running' ORDER BY j.created_at, j.id LIMIT 1"
        )).mappings().first()
    return {
        "worker_status": worker_status(database_path),
        "queued_count": queued,
        "active_job": dict(active) if active else None,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
