"""Execute a queued fixed-selection import and fence report publication."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sqlalchemy.engine import Engine

from backend.app.ingestion.batch import publish_import_run, stage_import_job
from backend.app.jobs.claim import ClaimedJob, timestamp
from backend.app.jobs.import_submission import get_import_request
from backend.app.jobs.ownership import LostClaim, finish_active_attempt
from backend.app.jobs.worker import maintain_attempt_lease


@dataclass(frozen=True, slots=True)
class ExecutedImport:
    job_id: str
    attempt_id: str
    status: str
    result_resource_id: str


def execute_claimed_import(
    engine: Engine,
    claimed: ClaimedJob,
    *,
    data_root: Path,
    client: httpx.Client | None = None,
    lease_seconds: float = 30.0,
    heartbeat_interval_seconds: float = 5.0,
) -> ExecutedImport:
    if claimed.kind != "import":
        raise ValueError("Only an import Job can run through the import executor.")
    owned_client = client is None
    resolved_client = client or httpx.Client()
    try:
        try:
            request = get_import_request(engine, claimed.job_id)
            if request is None:
                raise RuntimeError("The queued import request is missing.")
            with maintain_attempt_lease(
                engine,
                claimed.attempt_id,
                claimed.worker_id,
                lease_seconds=lease_seconds,
                heartbeat_interval_seconds=heartbeat_interval_seconds,
            ):
                staged = asyncio.run(
                    stage_import_job(
                        engine,
                        request,
                        attempt_id=claimed.attempt_id,
                        data_root=data_root,
                        client=resolved_client,
                    )
                )
        except LostClaim:
            raise
        except Exception as error:
            finish_active_attempt(
                engine,
                job_id=claimed.job_id,
                attempt_id=claimed.attempt_id,
                worker_id=claimed.worker_id,
                status="failed",
                finished_at=timestamp(datetime.now(timezone.utc)),
                error_kind=type(error).__name__,
                error_message=str(error),
            )
            raise

        finished_at = timestamp(datetime.now(timezone.utc))
        try:
            finish_active_attempt(
                engine,
                job_id=claimed.job_id,
                attempt_id=claimed.attempt_id,
                worker_id=claimed.worker_id,
                status="succeeded",
                finished_at=finished_at,
                result_resource_id=staged.run_id,
                publish=lambda session: publish_import_run(
                    session, staged, finished_at=finished_at
                ),
            )
        except LostClaim:
            raise
        except Exception as error:
            finish_active_attempt(
                engine,
                job_id=claimed.job_id,
                attempt_id=claimed.attempt_id,
                worker_id=claimed.worker_id,
                status="failed",
                finished_at=timestamp(datetime.now(timezone.utc)),
                error_kind=type(error).__name__,
                error_message=str(error),
            )
            raise
        return ExecutedImport(
            job_id=claimed.job_id,
            attempt_id=claimed.attempt_id,
            status="succeeded",
            result_resource_id=staged.run_id,
        )
    finally:
        if owned_client:
            resolved_client.close()
