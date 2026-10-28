"""Dispatch registered queued Job kinds from one worker process."""

from dataclasses import dataclass
from pathlib import Path

import httpx
from sqlalchemy.engine import Engine

from backend.app.jobs.claim import claim_next_job, require_positive_lease
from backend.app.jobs.import_worker import execute_claimed_import
from backend.app.jobs.worker import execute_claimed_search, load_claimed_search
from backend.app.search.client import ArxivClient


@dataclass(frozen=True, slots=True)
class ExecutedJob:
    kind: str
    job_id: str
    attempt_id: str
    status: str
    result_resource_id: str
    artifact_ref: str | None


# Register a kind only when its saved request and executor are implemented.
SUPPORTED_KINDS = ("search", "import", "index", "answer")


def _run_one_job(
    engine: Engine,
    *,
    worker_id: str,
    data_root: Path,
    search_client: ArxivClient | None = None,
    search_fallback=None,
    import_client: httpx.Client | None = None,
    index_settings=None,
    index_embedding_model=None,
    answer_settings=None,
    answer_executor=None,
    lease_seconds: float = 30.0,
    heartbeat_interval_seconds: float = 5.0,
) -> ExecutedJob | None:
    require_positive_lease(lease_seconds)
    if not 0 < heartbeat_interval_seconds < lease_seconds:
        raise ValueError("heartbeat interval must be positive and shorter than lease.")
    from backend.app.research.workflow import advance_research_operations
    advance_research_operations(engine, data_root=data_root, index_settings=index_settings)
    claimed = claim_next_job(
        engine,
        worker_id=worker_id,
        supported_kinds=SUPPORTED_KINDS,
        lease_seconds=lease_seconds,
    )
    if claimed is None:
        return None
    if claimed.kind == "search":
        search = load_claimed_search(engine, claimed)
        executed = execute_claimed_search(
            engine,
            search,
            data_root=data_root,
            client=search_client,
            fallback=search_fallback,
            lease_seconds=lease_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )
        return ExecutedJob(
            kind="search",
            job_id=executed.job_id,
            attempt_id=executed.attempt_id,
            status=executed.status,
            result_resource_id=claimed.resource_id,
            artifact_ref=executed.artifact_ref,
        )
    if claimed.kind == "import":
        executed = execute_claimed_import(
            engine,
            claimed,
            data_root=data_root,
            client=import_client,
            lease_seconds=lease_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )
        return ExecutedJob(
            kind="import",
            job_id=executed.job_id,
            attempt_id=executed.attempt_id,
            status=executed.status,
            result_resource_id=executed.result_resource_id,
            artifact_ref=None,
        )
    if claimed.kind == "index":
        from backend.app.jobs.index_worker import execute_claimed_index
        executed = execute_claimed_index(
            engine, claimed, data_root=data_root,
            settings=index_settings, embedding_model=index_embedding_model,
            lease_seconds=lease_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )
        return ExecutedJob(
            kind="index", job_id=executed.job_id, attempt_id=executed.attempt_id,
            status=executed.status, result_resource_id=executed.result_resource_id,
            artifact_ref=executed.artifact_ref,
        )
    if claimed.kind == "answer":
        from backend.app.jobs.answer_worker import execute_claimed_answer
        executed = execute_claimed_answer(
            engine, claimed, data_root=data_root,
            settings=answer_settings, answer_executor=answer_executor,
            lease_seconds=lease_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )
        return ExecutedJob(
            kind="answer", job_id=executed.job_id, attempt_id=executed.attempt_id,
            status=executed.status, result_resource_id=executed.answer_run_id,
            artifact_ref=executed.artifact_ref,
        )
    raise RuntimeError(f"No executor is registered for Job kind {claimed.kind!r}.")


def run_one_job(engine: Engine, *, worker_id: str, data_root: Path,
    search_client: ArxivClient | None = None, import_client: httpx.Client | None = None,
    search_fallback=None,
    index_settings=None, index_embedding_model=None, answer_settings=None, answer_executor=None,
    lease_seconds: float = 30.0, heartbeat_interval_seconds: float = 5.0) -> ExecutedJob | None:
    require_positive_lease(lease_seconds)
    if not 0 < heartbeat_interval_seconds < lease_seconds:
        raise ValueError("heartbeat interval must be positive and shorter than lease.")
    try:
        return _run_one_job(engine, worker_id=worker_id, data_root=data_root,
            search_client=search_client, import_client=import_client,
            search_fallback=search_fallback,
            index_settings=index_settings, index_embedding_model=index_embedding_model,
            answer_settings=answer_settings, answer_executor=answer_executor,
            lease_seconds=lease_seconds, heartbeat_interval_seconds=heartbeat_interval_seconds)
    finally:
        # Persist the chain's terminal state even when its executor raises and the
        # worker stops. This never retries a failed job; retry needs a new intent.
        from backend.app.research.workflow import advance_research_operations
        advance_research_operations(engine, data_root=data_root, index_settings=index_settings)
