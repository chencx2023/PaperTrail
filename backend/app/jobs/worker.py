"""Claim queued searches, execute outside the transaction, and save results."""

from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Thread
import warnings
from sqlalchemy import select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from backend.app.models import Job, JobAttempt, SearchRun
from backend.app.jobs.claim import (
    ClaimedJob,
    claim_next_job,
    require_positive_lease as _require_positive_lease,
    timestamp as _timestamp,
)
from backend.app.jobs.ownership import LostClaim, finish_active_attempt
from backend.app.jobs.service import read_search_request
from backend.app.search.query_builder import plan_from_data, InvalidQueryPlan
from backend.app.search.multi_query import run_multi_search
from backend.app.search.client import ArxivClient
from backend.app.search.service import (
    InvalidSearchSnapshot,
    QUERY_CONTRACT_VERSION,
    build_arxiv_query,
    publish_search_cache,
    run_search,
)


class QueryContractMismatch(RuntimeError):
    """A queued query cannot be executed under a changed query contract."""


class HeartbeatFailure(RuntimeError):
    """The worker could not verify that its lease remains current."""


@dataclass(frozen=True, slots=True)
class ClaimedSearch:
    job_id: str
    search_id: str
    attempt_id: str
    worker_id: str
    query_text: str
    normalized_query_text: str
    search_query: str
    query_contract_version: int
    query_plan: dict | None = None
    plan_source: dict | None = None


@dataclass(frozen=True, slots=True)
class ExecutedSearch:
    job_id: str
    attempt_id: str
    status: str
    artifact_ref: str


def claim_next_search(
    engine: Engine, *, worker_id: str, lease_seconds: float = 30.0
) -> ClaimedSearch | None:
    claimed = claim_next_job(
        engine,
        worker_id=worker_id,
        supported_kinds=("search",),
        lease_seconds=lease_seconds,
    )
    return load_claimed_search(engine, claimed) if claimed is not None else None


def load_claimed_search(engine: Engine, claimed: ClaimedJob) -> ClaimedSearch:
    """Load the saved query for an already claimed search attempt."""

    if claimed.kind != "search":
        raise ValueError("Only a search Job can be loaded as a search attempt.")
    with Session(engine) as session:
        search = session.get(SearchRun, claimed.resource_id)
        if search is not None:
            try:
                request = read_search_request(session, claimed.job_id)
            except InvalidQueryPlan as error:
                finish_active_attempt(engine, job_id=claimed.job_id, attempt_id=claimed.attempt_id,
                    worker_id=claimed.worker_id, status="failed", finished_at=_timestamp(datetime.now(timezone.utc)),
                    error_kind="invalid_query_plan", error_message=str(error))
                raise
            return ClaimedSearch(
                job_id=claimed.job_id,
                search_id=claimed.resource_id,
                attempt_id=claimed.attempt_id,
                worker_id=claimed.worker_id,
                query_text=search.query_text,
                normalized_query_text=search.normalized_query_text,
                search_query=search.search_query,
                query_contract_version=search.query_contract_version,
                query_plan=request["plan"] if request else None,
                plan_source=request["generation"] if request else None,
            )

    finish_active_attempt(
        engine,
        job_id=claimed.job_id,
        attempt_id=claimed.attempt_id,
        worker_id=claimed.worker_id,
        status="failed",
        finished_at=_timestamp(datetime.now(timezone.utc)),
        error_kind="missing_search_resource",
        error_message="The queued search resource is missing.",
    )
    raise RuntimeError("A queued search job has no search resource.")


def refresh_lease(
    engine: Engine,
    attempt_id: str,
    worker_id: str,
    *,
    lease_seconds: float = 30.0,
) -> bool:
    """Renew only an attempt still owned by this worker and active Job."""

    _require_positive_lease(lease_seconds)
    current_time = datetime.now(timezone.utc)
    active_job = (
        select(Job.id)
        .where(
            Job.id == JobAttempt.job_id,
            Job.status == "running",
            Job.active_attempt_id == attempt_id,
        )
        .exists()
    )
    with Session(engine) as session:
        renewed = session.execute(
            update(JobAttempt)
            .where(
                JobAttempt.id == attempt_id,
                JobAttempt.worker_id == worker_id,
                JobAttempt.status == "running",
                active_job,
            )
            .values(
                heartbeat_at=_timestamp(current_time),
                lease_until=_timestamp(
                    current_time + timedelta(seconds=lease_seconds)
                ),
            )
            .returning(JobAttempt.id)
        ).scalar_one_or_none()
        session.commit()
        return renewed is not None


@contextmanager
def maintain_attempt_lease(
    engine: Engine,
    attempt_id: str,
    worker_id: str,
    *,
    lease_seconds: float,
    heartbeat_interval_seconds: float,
):
    """Keep the active attempt leased while slow work runs outside the DB."""

    _require_positive_lease(lease_seconds)
    if not 0 < heartbeat_interval_seconds < lease_seconds:
        raise ValueError("heartbeat interval must be positive and shorter than lease.")
    stop_heartbeat = Event()
    heartbeat_state = {"lost": False, "error": None}

    def heartbeat_loop() -> None:
        while not stop_heartbeat.wait(heartbeat_interval_seconds):
            try:
                if not refresh_lease(
                    engine, attempt_id, worker_id, lease_seconds=lease_seconds
                ):
                    heartbeat_state["lost"] = True
                    break
                heartbeat_state["error"] = None
            except Exception as error:
                heartbeat_state["error"] = error

    heartbeat = Thread(
        target=heartbeat_loop,
        name=f"job-heartbeat-{attempt_id[:8]}",
        daemon=True,
    )
    heartbeat.start()
    try:
        yield
    finally:
        stop_heartbeat.set()
        heartbeat.join()
    if heartbeat_state["lost"]:
        raise LostClaim("Heartbeat found this attempt is no longer active.")
    if heartbeat_state["error"] is not None:
        raise HeartbeatFailure("The last heartbeat could not renew the lease.") from heartbeat_state["error"]


def interrupt_expired_attempt(
    engine: Engine,
    attempt_id: str,
    *,
    worker_stopped_confirmed: bool,
    now: datetime | None = None,
) -> bool:
    """Mark an expired attempt interrupted after an operator confirms worker stop."""

    if not worker_stopped_confirmed:
        raise ValueError("An operator must confirm the old worker stopped.")
    timestamp = _timestamp(now or datetime.now(timezone.utc))
    expired_attempt = (
        select(JobAttempt.id)
        .where(
            JobAttempt.id == attempt_id,
            JobAttempt.job_id == Job.id,
            JobAttempt.status == "running",
            JobAttempt.lease_until.is_not(None),
            JobAttempt.lease_until <= timestamp,
        )
        .exists()
    )
    with Session(engine) as session:
        job_id = session.execute(
            update(Job)
            .where(
                Job.status == "running",
                Job.active_attempt_id == attempt_id,
                expired_attempt,
            )
            .values(status="interrupted", updated_at=timestamp)
            .returning(Job.id)
        ).scalar_one_or_none()
        if job_id is None:
            session.rollback()
            return False
        interrupted = session.execute(
            update(JobAttempt)
            .where(
                JobAttempt.id == attempt_id,
                JobAttempt.job_id == job_id,
                JobAttempt.status == "running",
                JobAttempt.lease_until <= timestamp,
            )
            .values(
                status="interrupted",
                finished_at=timestamp,
                error_kind="lease_expired",
                error_message="Operator confirmed the worker stopped after lease expiry.",
            )
            .returning(JobAttempt.id)
        ).scalar_one_or_none()
        if interrupted is None:
            session.rollback()
            return False
        session.commit()
        return True


def _commit_terminal(
    engine: Engine,
    claimed: ClaimedSearch,
    *,
    status: str,
    artifact_ref: str | None = None,
    retrieved_at: str | None = None,
    error_kind: str | None = None,
    error_message: str | None = None,
) -> None:
    """Publish the search snapshot only while this worker owns its attempt."""

    now = _timestamp(datetime.now(timezone.utc))
    def publish(session: Session) -> None:
        if artifact_ref is not None:
            search_id = session.execute(
                update(SearchRun)
                .where(SearchRun.id == claimed.search_id)
                .values(snapshot_ref=artifact_ref, retrieved_at=retrieved_at)
                .returning(SearchRun.id)
            ).scalar_one_or_none()
            if search_id is None:
                raise RuntimeError("A search job has no search resource.")

    finish_active_attempt(
        engine,
        job_id=claimed.job_id,
        attempt_id=claimed.attempt_id,
        worker_id=claimed.worker_id,
        status=status,
        finished_at=now,
        artifact_ref=artifact_ref,
        error_kind=error_kind,
        error_message=error_message,
        result_resource_id=claimed.search_id if status == "succeeded" else None,
        publish=publish,
    )


def run_one_search(
    engine: Engine,
    *,
    worker_id: str,
    data_root: Path,
    client: ArxivClient | None = None,
    lease_seconds: float = 30.0,
    heartbeat_interval_seconds: float = 5.0,
) -> ExecutedSearch | None:
    _require_positive_lease(lease_seconds)
    if not 0 < heartbeat_interval_seconds < lease_seconds:
        raise ValueError("heartbeat interval must be positive and shorter than lease.")
    claimed = claim_next_search(
        engine, worker_id=worker_id, lease_seconds=lease_seconds
    )
    if claimed is None:
        return None
    return execute_claimed_search(
        engine,
        claimed,
        data_root=data_root,
        client=client,
        lease_seconds=lease_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )


def execute_claimed_search(
    engine: Engine,
    claimed: ClaimedSearch,
    *,
    data_root: Path,
    client: ArxivClient | None = None,
    lease_seconds: float = 30.0,
    heartbeat_interval_seconds: float = 5.0,
) -> ExecutedSearch:
    """Run one already claimed search without taking a second Job."""

    _require_positive_lease(lease_seconds)
    if not 0 < heartbeat_interval_seconds < lease_seconds:
        raise ValueError("heartbeat interval must be positive and shorter than lease.")
    try:
        current_plan = build_arxiv_query(claimed.query_text)
        if (
            claimed.query_contract_version != QUERY_CONTRACT_VERSION
            or claimed.normalized_query_text != current_plan.normalized_query_text
            or claimed.search_query != current_plan.search_query
        ):
            raise QueryContractMismatch(
                "Saved query contract differs from the current search implementation."
            )
        with maintain_attempt_lease(
            engine,
            claimed.attempt_id,
            claimed.worker_id,
            lease_seconds=lease_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        ):
            if claimed.query_plan is not None:
                multi = plan_from_data(claimed.query_plan)
                if multi.baseline_query_text != current_plan.normalized_query_text:
                    raise QueryContractMismatch("Saved multi-query baseline differs from the current query.")
                run = run_multi_search(multi, client=client or ArxivClient(), output_root=Path(data_root) / "runs",
                    run_id=f"search-{claimed.attempt_id}", generation=claimed.plan_source)
            else:
                run = run_search(
                    claimed.query_text, client=client or ArxivClient(),
                    output_root=Path(data_root) / "runs", run_id=f"search-{claimed.attempt_id}",
                    cache_requires_commit_marker=True,
                )
    except LostClaim:
        raise
    except Exception as error:
        _commit_terminal(
            engine,
            claimed,
            status="failed",
            error_kind=type(error).__name__,
            error_message=str(error),
        )
        raise
    artifact_ref = f"runs/{run.run_id}/run.json"
    status = "succeeded" if run.status == "success" else "failed"
    _commit_terminal(
        engine,
        claimed,
        status=status,
        artifact_ref=artifact_ref,
        retrieved_at=run.fetched_at if status == "succeeded" else None,
        error_kind=run.error_kind,
        error_message=run.error_message,
    )
    if status == "succeeded":
        try:
            publish_search_cache(run.run_dir)
        except (InvalidSearchSnapshot, OSError) as error:
            warnings.warn(
                f"Search job committed, but its cache marker was not published: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
    return ExecutedSearch(
        job_id=claimed.job_id,
        attempt_id=claimed.attempt_id,
        status=status,
        artifact_ref=artifact_ref,
    )
