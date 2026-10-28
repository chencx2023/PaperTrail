"""Run one PaperQA answer with a fresh session and observable scope snapshots."""

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
import hashlib
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
    generation_records: tuple[dict, ...] = ()
    session_token_counts: dict | None = None
    recovery_events: tuple[dict, ...] = ()


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
        self.generation_records: list[dict] = []
        self.window_proofs: dict[str, dict] = {}

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
        def make_tools(self):
            tools = super().make_tools()
            if not getattr(self._settings, "grounded_answers", False):
                return tools
            for tool in tools:
                if tool.info.name != "gen_answer":
                    continue
                original = tool._tool_fn
                owner = original.__self__

                @wraps(original)
                async def grounded_answer(state):
                    from backend.app.qa.grounded import generate_grounded_answer
                    repair = getattr(self._settings, "grounded_profile", "audited") == "compact_repair"
                    if getattr(self._settings, "grounded_profile", "audited") in {"compact", "compact_repair"}:
                        from backend.app.qa.compact import generate_compact_grounded_answer
                        generate_grounded_answer = generate_compact_grounded_answer
                        if repair:
                            from functools import partial
                            generate_grounded_answer = partial(generate_grounded_answer, repair=True,
                                insufficient_facts_compat=getattr(self._settings, "insufficient_facts_compat", False),
                                citation_scope_review=getattr(self._settings, "citation_scope_review", False))
                    # Check document scope before sending any material to a model.
                    recorder.capture("generation_start", state)
                    if repair:
                        from backend.app.qa.recovery import recovery_events
                        if any(e.get("decision") == "not_retryable" and e.get("http_status") in {401,402,403}
                               for e in (recovery_events.get() or [])):
                            raise ValueError("Provider authentication, balance or access failed; fallback generation is unavailable.")
                    if not state.session.contexts and self._settings.answer.get_evidence_if_no_contexts:
                        state.session = await state.docs.aget_evidence(state.session,
                            settings=self._settings, embedding_model=owner.embedding_model,
                            summary_llm_model=owner.summary_llm_model)
                        recorder.capture("generation_evidence", state)
                    from backend.app.qa.windows import expand_original_windows, WINDOW_POLICY, REPAIR_WINDOW_POLICY
                    from backend.app.qa.context import window_context_serializer, repair_context_serializer
                    reader = self._settings.parsing.reader_config
                    window_enabled = self._settings.custom_context_serializer in (window_context_serializer, repair_context_serializer)
                    window_policy = REPAIR_WINDOW_POLICY if repair else WINDOW_POLICY
                    contexts, window_records = (expand_original_windows(state.session.contexts,
                        state.docs.texts, overlap=reader.get("overlap", 0), chunk_chars=reader.get("chunk_chars", 3000),
                        cutoff=self._settings.answer.evidence_relevance_score_cutoff, include_predecessor=repair)
                        if window_enabled else (list(state.session.contexts), []))
                    recorder.window_proofs.update({row["window_context_id"]: row for row in window_records})
                    window_records = [recorder.window_proofs[c.id] for c in contexts if c.id in recorder.window_proofs]
                    state.session.contexts = contexts
                    recorder.capture("generation_source_windows", state)
                    context = await self._settings.context_serializer(
                        contexts=contexts, question=state.session.question, pre_str=None)
                    context_hash = hashlib.sha256(context.encode()).hexdigest()
                    previous = recorder.generation_records[-1] if recorder.generation_records else None
                    if previous is not None and previous.get("final_context_sha256") != context_hash:
                        previous = None
                    if previous is not None:
                        if previous["answer_published"]:
                            self._grounded_ready = True
                            return f"{state.session.answer} | {state.status}"
                        raise ValueError("This evidence already failed generation validation; gather different evidence.")
                    state.session.context = context
                    record_index, counted_results = None, 0
                    def save_generation_progress(audit, llm_results):
                        from copy import deepcopy
                        nonlocal record_index, counted_results
                        saved = {**deepcopy(audit), "answer_published": False,
                                 "source_window_policy": {**window_policy, "enabled": window_enabled},
                                 "source_window_expansions": deepcopy(window_records)}
                        if record_index is None:
                            record_index = len(recorder.generation_records)
                            recorder.generation_records.append(saved)
                        else:
                            recorder.generation_records[record_index] = saved
                        for llm_result in llm_results[counted_results:]:
                            state.session.add_tokens(llm_result)
                        counted_results = len(llm_results)
                    result = await generate_grounded_answer(question=state.session.question,
                        final_context=context, model=owner.llm_model, on_progress=save_generation_progress)
                    save_generation_progress(result.audit, result.llm_results)
                    recorder.generation_records[record_index].update(status=result.status,
                        answer_published=result.answer is not None)
                    if result.answer is None:
                        recorder.capture("generation_failed", state)
                        reason = result.audit.get("error_code") or result.audit.get("error_kind") or "unknown"
                        raise ValueError(f"Grounded generation failed validation ({reason}); no answer was published.")
                    state.session.raw_answer = result.answer
                    state.session.contexts = contexts
                    state.session.populate_formatted_answers_and_bib_from_raw_answer()
                    recorder.capture("generation_finished", state)
                    self._grounded_ready = True
                    return f"{state.session.answer} | {state.status}"

                tool._tool_fn = grounded_answer
                tool.concurrency_safe = False
            return tools

        async def step(self, action):
            self._grounded_ready = False
            messages, reward, done, truncated = await super().step(action)
            calls = getattr(action, "tool_calls", ())
            if (self._grounded_ready and len(calls) == 1
                    and calls[0].function.name == "gen_answer"):
                from paperqa.agents.tools import Complete
                final = recorder.generation_records[-1]["stages"]["grounded_focus"]["parsed"]
                certain = all(part["status"] == "supported" for part in final["parts"])
                await Complete().complete(has_successful_answer=certain, state=self.state)
                recorder.generation_records[-1]["completion_mode"] = "local_native_complete_after_validated_generation"
                recorder.capture("grounded_local_complete", self.state)
                done = True
            return messages, reward, done, truncated

        async def reset(self):
            self._grounded_ready = False
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
    from backend.app.qa.recovery import recovery_events as recovery_context
    recoveries = []
    recovery_token = recovery_context.set(recoveries)
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
            generation_records=tuple(recorder.generation_records),
            session_token_counts=recorder.state.session.token_counts if recorder.state is not None else None,
            recovery_events=tuple(recoveries),
        )
    finally:
        recovery_context.reset(recovery_token)
    if recorder.generation_records and not recorder.generation_records[-1]["answer_published"]:
        return AnswerExecution(question=question, started_at=started_at, finished_at=_utcnow(),
            execution_status="failed", failure_kind="grounded_generation_failed", error_message=None,
            response=None, snapshots=tuple(recorder.snapshots), contexts=recorder.contexts,
            final_context=recorder.snapshots[-1].final_context if recorder.snapshots else None,
            final_context_coverage="paperqa_session_context_only; full_model_request_unobservable",
            generation_records=tuple(recorder.generation_records),
            session_token_counts=recorder.state.session.token_counts if recorder.state is not None else None,
            recovery_events=tuple(recoveries))
    succeeded = response.status.value in {"success", "unsure", "truncated"}
    recovery_failed = any(e["decision"] in {"exhausted", "not_retryable"} for e in recoveries)
    if getattr(scope.settings, "grounded_profile", "audited") == "compact_repair":
        succeeded = response.status.value in {"success", "unsure"} and not recovery_failed
    return AnswerExecution(
        question=question,
        started_at=started_at, finished_at=_utcnow(),
        execution_status="succeeded" if succeeded else "failed",
        failure_kind=None if succeeded else "agent_provider_failed" if recovery_failed else "upstream_failed",
        error_message=None,
        response=response,
        snapshots=tuple(recorder.snapshots),
        contexts=recorder.contexts,
        final_context=response.session.context or None,
        final_context_coverage="paperqa_session_context_only; full_model_request_unobservable",
        generation_records=tuple(recorder.generation_records),
        session_token_counts=response.session.token_counts,
        recovery_events=tuple(recoveries),
    )
