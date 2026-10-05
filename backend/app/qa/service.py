"""Run one PaperQA answer with a fresh session and observable scope snapshots."""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging

from paperqa import Docs
from paperqa.agents.env import PaperQAEnvironment
from paperqa.agents.main import run_agent
from paperqa.agents.models import AnswerResponse

from backend.app.qa.scope import AnswerScope


@dataclass(frozen=True, slots=True)
class ScopeSnapshot:
    stage: str
    loaded_dockeys: tuple[str, ...]
    text_dockeys: tuple[str, ...]
    context_dockeys: tuple[str, ...]
    final_context: str | None


@dataclass(frozen=True, slots=True)
class CapturedContext:
    upstream_context_id: str
    upstream_dockey: str
    summary: str
    original_text: str | None
    score: float | None
    upstream_locator: str | None


@dataclass(frozen=True, slots=True)
class AnswerExecution:
    question: str
    started_at: str
    finished_at: str
    execution_status: str
    failure_kind: str | None
    error_message: str | None
    response: AnswerResponse | None
    snapshots: tuple[ScopeSnapshot, ...]
    contexts: tuple[CapturedContext, ...]
    final_context: str | None
    final_context_coverage: str


class ScopeViolation(ValueError):
    """PaperQA loaded or collected a document outside the frozen revision."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


class _ScopeRecorder:
    def __init__(self, allowed_dockeys: set[str]) -> None:
        self.allowed_dockeys = allowed_dockeys
        self.snapshots: list[ScopeSnapshot] = []
        self.contexts: tuple[CapturedContext, ...] = ()
        self.state = None
        self.violation: str | None = None

    def capture(self, stage: str, state) -> None:
        self.state = state
        snapshot = ScopeSnapshot(
            stage=stage,
            loaded_dockeys=tuple(sorted(str(key) for key in state.docs.docs)),
            text_dockeys=tuple(sorted({str(text.doc.dockey) for text in state.docs.texts})),
            context_dockeys=tuple(sorted({
                str(context.text.doc.dockey) for context in state.session.contexts
            })),
            final_context=state.session.context or None,
        )
        self.snapshots.append(snapshot)
        if stage != "finished":
            # AnswerResponse removes Context.text.text in place. Keep only the
            # evidence fields needed by our adapter before that happens.
            self.contexts = tuple(
                CapturedContext(
                    upstream_context_id=context.id,
                    upstream_dockey=str(context.text.doc.dockey),
                    summary=context.context,
                    original_text=context.text.text or None,
                    score=context.score,
                    upstream_locator=context.text.name or None,
                )
                for context in state.session.contexts
            )
        observed = set(snapshot.loaded_dockeys + snapshot.text_dockeys + snapshot.context_dockeys)
        outside = observed - self.allowed_dockeys
        if outside:
            self.violation = f"Observed document keys outside the confirmed revision: {sorted(outside)}"
            raise ScopeViolation(self.violation)


def _recording_environment(recorder: _ScopeRecorder) -> type[PaperQAEnvironment]:
    class RecordingEnvironment(PaperQAEnvironment):
        async def reset(self):
            result = await super().reset()
            recorder.capture("reset", self.state)
            return result

        async def exec_tool_calls(self, message, *args, **kwargs):
            if recorder.violation is not None:
                # PaperQA may try a fallback gen_answer after a tool error.
                raise ScopeViolation(recorder.violation)
            try:
                result = await super().exec_tool_calls(message, *args, **kwargs)
            finally:
                # Also inspect mutations made just before a tool raises. PaperQA
                # can otherwise generate a fallback answer from that state.
                recorder.capture(
                    ", ".join(call.function.name for call in message.tool_calls),
                    self.state,
                )
            return result

    return RecordingEnvironment


async def execute_scoped_answer(scope: AnswerScope, question: str) -> AnswerExecution:
    """Execute one answer; no database result is committed by this step."""

    if not question.strip():
        raise ValueError("A nonempty research question is required.")
    # Pinned PaperQA logs its full Settings (including API keys) at INFO.
    upstream_logger = logging.getLogger("paperqa.agents.main")
    if upstream_logger.getEffectiveLevel() < logging.WARNING:
        upstream_logger.setLevel(logging.WARNING)
    recorder = _ScopeRecorder({document.versioned_id for document in scope.documents})
    started_at = _utcnow()
    try:
        response = await run_agent(
            Docs(), question, scope.settings.model_copy(deep=True),
            agent_type="ToolSelector", env_class=_recording_environment(recorder),
        )
        if recorder.state is not None:
            recorder.capture("finished", recorder.state)
    except ScopeViolation as error:
        return AnswerExecution(
            question=question,
            started_at=started_at, finished_at=_utcnow(),
            execution_status="failed", failure_kind="scope_violation",
            error_message=str(error), response=None,
            snapshots=tuple(recorder.snapshots),
            contexts=recorder.contexts,
            final_context=recorder.snapshots[-1].final_context if recorder.snapshots else None,
            final_context_coverage="paperqa_session_context_only; full_model_request_unobservable",
        )
    succeeded = response.status.value in {"success", "unsure", "truncated"}
    return AnswerExecution(
        question=question,
        started_at=started_at, finished_at=_utcnow(),
        execution_status="succeeded" if succeeded else "failed",
        failure_kind=None if succeeded else "upstream_failed",
        error_message=None,
        response=response,
        snapshots=tuple(recorder.snapshots),
        contexts=recorder.contexts,
        final_context=response.session.context or None,
        final_context_coverage="paperqa_session_context_only; full_model_request_unobservable",
    )
