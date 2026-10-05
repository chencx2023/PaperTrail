"""Publish one immutable answer artifact before committing its database pointer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import re
from uuid import uuid4
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from backend.app.collections.service import utcnow
from backend.app.models import AnswerRun, CollectionRevisionDocument, IndexArtifact
from backend.app.pqa.adapter import AnswerBundle

if TYPE_CHECKING:
    from backend.app.qa.scope import AnswerScope
    from backend.app.qa.service import AnswerExecution


@dataclass(frozen=True, slots=True)
class SavedAnswerRun:
    id: str
    artifact_ref: str
    artifact_sha256: str
    saved_at: str


@dataclass(frozen=True, slots=True)
class StagedAnswerRun:
    saved: SavedAnswerRun
    scope: AnswerScope
    execution: AnswerExecution
    bundle: AnswerBundle


def _artifact_ref(run_id: str) -> str:
    if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise ValueError("A generated 32-character answer run ID is required.")
    return f"answers/{run_id}.json"


def _answer_path(data_root: Path, run_id: str, *, create_parent: bool) -> Path:
    root = Path(data_root).resolve()
    directory = root / "answers"
    if create_parent:
        directory.mkdir(parents=True, exist_ok=True)
    if not directory.resolve().is_relative_to(root):
        raise ValueError("The answer directory escapes the data root.")
    return directory / f"{run_id}.json"


def _safe_run_config(scope: AnswerScope, index_config_hash: str) -> dict:
    from backend.app.qa.context import context_policy_identity
    from backend.app.qa.prompts import answer_prompt_identity

    settings = scope.settings
    return {
        "coverage": "selected_effective_fields; provider_configs_excluded",
        "paperqa_version": version("paper-qa"),
        "index_config_hash": index_config_hash,
        "model_roles": {
            "agent": settings.agent.agent_llm,
            "evidence_summary": settings.summary_llm,
            "answer": settings.llm,
            "embedding": settings.embedding,
        },
        "agent": {
            "type": settings.agent.agent_type,
            "tools": list(settings.agent.tool_names or []),
            "search_count": settings.agent.search_count,
            "max_timesteps": settings.agent.max_timesteps,
            "timeout_seconds": settings.agent.timeout,
        },
        "answer": {
            "evidence_k": settings.answer.evidence_k,
            "answer_max_sources": settings.answer.answer_max_sources,
            "answer_length": settings.answer.answer_length,
            "max_concurrent_requests": settings.answer.max_concurrent_requests,
        },
        "prompts": {
            "qa": answer_prompt_identity(settings.prompts.qa),
            "context": context_policy_identity(settings.custom_context_serializer),
        },
    }


def stage_answer_run(
    engine: Engine,
    scope: AnswerScope,
    execution: AnswerExecution,
    bundle: AnswerBundle,
    *,
    data_root: Path,
) -> StagedAnswerRun:
    """Validate and publish bytes, without making them a committed answer."""

    artifact_ref = _artifact_ref(bundle.run_id)
    from backend.app.evidence.service import build_scoped_answer_bundle
    if bundle.schema_version != 2 or bundle != build_scoped_answer_bundle(
        bundle.run_id, scope, execution
    ):
        raise ValueError("The answer bundle differs from the scoped execution.")

    with Session(engine) as session:
        if session.get(AnswerRun, bundle.run_id) is not None:
            raise FileExistsError("This answer run has already been saved.")
        artifact = session.get(IndexArtifact, scope.index_artifact_id)
        if artifact is None or artifact.revision_id != scope.revision_id or artifact.status != "ready":
            raise ValueError("The answer index is no longer a ready artifact for this revision.")
        members = session.scalars(
            select(CollectionRevisionDocument).where(
                CollectionRevisionDocument.revision_id == scope.revision_id
            )
        ).all()
        actual = {(m.document_id, m.versioned_id, m.sha256) for m in members}
        allowed = {(d.document_id, d.versioned_id, d.sha256) for d in scope.documents}
        if not actual or actual != allowed or len(members) != len(scope.documents):
            raise ValueError("The frozen revision differs from the answer scope.")
        index_config_hash = artifact.config_hash

    saved_bundle = bundle.model_copy(update={
        "run_config": _safe_run_config(scope, index_config_hash),
    })
    from backend.app.qa.context import original_context_serializer, saved_selection_report
    if scope.settings.custom_context_serializer is original_context_serializer:
        saved_bundle.run_config['context_selection'] = saved_selection_report(
            saved_bundle, scope.settings.answer.evidence_relevance_score_cutoff)
    body = saved_bundle.model_dump_json(indent=2).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    target = _answer_path(data_root, bundle.run_id, create_parent=True)
    temporary = target.with_name(f".{bundle.run_id}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
        # A hard link publishes all bytes at once without replacing another run.
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

    saved = SavedAnswerRun(bundle.run_id, artifact_ref, digest, utcnow())
    return StagedAnswerRun(saved, scope, execution, saved_bundle)


def publish_staged_answer(
    session: Session, staged: StagedAnswerRun, *, job_attempt_id: str | None,
) -> None:
    """Insert the answer row inside its owning Job terminal transaction."""

    bundle = staged.bundle
    scope = staged.scope
    execution = staged.execution
    saved = staged.saved
    if session.get(AnswerRun, bundle.run_id) is not None:
        raise FileExistsError("This answer run has already been saved.")
    artifact = session.get(IndexArtifact, scope.index_artifact_id)
    if artifact is None or artifact.status != "ready" or artifact.revision_id != scope.revision_id:
        raise ValueError("The answer index is no longer ready for this revision.")
    session.add(AnswerRun(
        id=bundle.run_id,
        job_attempt_id=job_attempt_id,
        collection_revision_id=scope.revision_id,
        index_artifact_id=scope.index_artifact_id,
        question=bundle.question,
        execution_status=bundle.job_status,
        answer_outcome=bundle.answer_outcome,
        upstream_status=bundle.upstream_status,
        termination_reason=bundle.termination_reason,
        artifact_ref=saved.artifact_ref,
        artifact_sha256=saved.artifact_sha256,
        started_at=execution.started_at,
        finished_at=execution.finished_at,
        saved_at=saved.saved_at,
    ))


def save_answer_run(
    engine: Engine,
    scope: AnswerScope,
    execution: AnswerExecution,
    bundle: AnswerBundle,
    *,
    data_root: Path,
) -> SavedAnswerRun:
    """Keep the T05 direct-save entry point for a standalone verified run."""

    staged = stage_answer_run(engine, scope, execution, bundle, data_root=data_root)
    with Session(engine) as session:
        publish_staged_answer(session, staged, job_attempt_id=None)
        session.commit()
    return staged.saved


def load_answer_run(engine: Engine, run_id: str, *, data_root: Path) -> AnswerBundle:
    """Read and verify a committed answer without invoking PaperQA or a model."""

    expected_ref = _artifact_ref(run_id)
    with Session(engine) as session:
        row = session.get(AnswerRun, run_id)
        if row is None:
            raise LookupError("No committed answer run has this ID.")
        if row.artifact_ref != expected_ref:
            raise ValueError("Answer artifact reference does not match its run ID.")
        expected_hash = row.artifact_sha256
        metadata = (row.question, row.execution_status, row.answer_outcome,
                    row.upstream_status, row.termination_reason,
                    row.collection_revision_id, row.index_artifact_id)

    path = _answer_path(data_root, run_id, create_parent=False)
    if path.is_symlink():
        raise ValueError("Answer artifact cannot be a symbolic link.")
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_hash:
        raise ValueError("Answer artifact hash does not match its database record.")
    bundle = AnswerBundle.model_validate_json(body)
    if (
        bundle.run_id != run_id
        or (bundle.question, bundle.job_status, bundle.answer_outcome,
            bundle.upstream_status, bundle.termination_reason,
            bundle.scope_observation.get("revision_id"),
            bundle.scope_observation.get("index_artifact_id")) != metadata
    ):
        raise ValueError("Answer artifact metadata differs from its database record.")
    return bundle
