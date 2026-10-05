"""Build an index outside the transaction, publishing ready only for its owner."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from paperqa import Settings
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from backend.app.collections.indexing import build_collection_index
from backend.app.collections.revision_submission import get_index_request
from backend.app.jobs.claim import ClaimedJob, timestamp
from backend.app.jobs.ownership import LostClaim, finish_active_attempt
from backend.app.jobs.worker import maintain_attempt_lease
from backend.app.models import IndexArtifact


@dataclass(frozen=True, slots=True)
class ExecutedIndex:
    job_id: str
    attempt_id: str
    status: str
    result_resource_id: str
    artifact_ref: str


def _runtime_settings() -> Settings:
    from scripts.check_models import load_config
    from scripts.model_config import build_settings
    return build_settings(load_config("all"))


def execute_claimed_index(
    engine: Engine, claimed: ClaimedJob, *, data_root: Path,
    settings: Settings | None = None, embedding_model=None,
    lease_seconds: float = 30.0, heartbeat_interval_seconds: float = 5.0,
) -> ExecutedIndex:
    if claimed.kind != "index":
        raise ValueError("Only an index Job can use the index executor.")
    try:
        request = get_index_request(engine, claimed.job_id)
        if request is None or request.revision_id != claimed.resource_id:
            raise RuntimeError("Queued index request is unavailable.")
        configured = settings or _runtime_settings()
        model = embedding_model or configured.get_embedding_model()
        with maintain_attempt_lease(
            engine, claimed.attempt_id, claimed.worker_id,
            lease_seconds=lease_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        ):
            artifact = asyncio.run(build_collection_index(
                engine, request.revision_id, data_root=data_root,
                settings=configured, embedding_model=model, defer_ready=True,
                active_claim=(claimed.job_id, claimed.attempt_id, claimed.worker_id),
            ))
        artifact_ref = f"indexes/collections/{artifact.config_hash}"

        def publish(session: Session) -> None:
            row = session.get(IndexArtifact, artifact.id)
            if row is None or row.revision_id != request.revision_id or row.config_hash != artifact.config_hash:
                raise RuntimeError("Built index no longer matches its frozen revision.")
            if row.status not in ("building", "ready"):
                raise RuntimeError("Built index is not publishable.")
            row.status = "ready"
            row.artifact_ref = artifact_ref
            row.error_kind = None
            row.error_message = None
            row.updated_at = timestamp(datetime.now(timezone.utc))

        finish_active_attempt(
            engine, job_id=claimed.job_id, attempt_id=claimed.attempt_id,
            worker_id=claimed.worker_id, status="succeeded",
            finished_at=timestamp(datetime.now(timezone.utc)),
            artifact_ref=artifact_ref, result_resource_id=artifact.id,
            publish=publish,
        )
        return ExecutedIndex(claimed.job_id, claimed.attempt_id, "succeeded", artifact.id, artifact_ref)
    except LostClaim:
        raise
    except Exception as error:
        def fail_artifact(session: Session) -> None:
            row = session.scalar(select(IndexArtifact).where(
                IndexArtifact.revision_id == claimed.resource_id,
                IndexArtifact.status == "building",
            ))
            if row is not None:
                row.status = "failed"
                row.error_kind = type(error).__name__
                row.error_message = str(error)[:800]
                row.updated_at = timestamp(datetime.now(timezone.utc))

        finish_active_attempt(
            engine, job_id=claimed.job_id, attempt_id=claimed.attempt_id,
            worker_id=claimed.worker_id, status="failed",
            finished_at=timestamp(datetime.now(timezone.utc)),
            error_kind=type(error).__name__, error_message=str(error),
            publish=fail_artifact,
        )
        raise
