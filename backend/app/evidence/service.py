"""Convert one scoped PaperQA execution into a reviewable answer bundle."""

import re

from backend.app.pqa.adapter import AnswerBundle, PaperQAAdapter
from backend.app.qa.context import CONTEXT_POLICY, LEGACY_CONTEXT_POLICY, evidence_excerpt
from backend.app.qa.scope import AnswerScope
from backend.app.qa.service import AnswerExecution


_FINAL_CONTEXT_COVERAGE = "paperqa_session_context_only; full_model_request_unobservable"
_VALID_KEYS = re.compile(
    r"(?:^|\n)Valid Keys: (pqac-[A-Za-z0-9_-]+(?:, pqac-[A-Za-z0-9_-]+)*)\s*\Z"
)
_CONTEXT_LABEL = re.compile(r"^([A-Za-z0-9_-]+): ", re.MULTILINE)


def summarize_saved_context_coverage(bundle: AnswerBundle, frozen_members) -> dict:
    """Describe only the PaperQA session context saved with this answer."""

    members = list(frozen_members)
    result = {
        "status": "unknown", "basis": "saved_paperqa_session_context",
        "total_documents": len(members),
        "included_documents": None, "missing_documents": None,
    }
    member_by_id = {member.document_id: member for member in members}
    if not members or len(member_by_id) != len(members):
        return result
    if bundle.schema_version < 2 or bundle.scope_observation.get(
        "final_context_coverage"
    ) != _FINAL_CONTEXT_COVERAGE:
        return result
    snapshots = bundle.scope_observation.get("snapshots")
    if (not isinstance(snapshots, list) or not snapshots
            or not isinstance(snapshots[-1], dict)
            or snapshots[-1].get("stage") != "finished"
            or snapshots[-1].get("final_context") != bundle.final_context):
        return result
    allowed = bundle.allowed_documents
    if len(allowed) != len(members) or any(
        not isinstance(document, dict)
        or document.get("dockey") != document.get("versioned_id")
        for document in allowed
    ):
        return result
    allowed_keys = [
        (document.get("document_id"), document.get("versioned_id"), document.get("sha256"))
        for document in allowed
    ]
    frozen_keys = [
        (member.document_id, member.versioned_id, member.sha256) for member in members
    ]
    if len(set(allowed_keys)) != len(members) or set(allowed_keys) != set(frozen_keys):
        return result
    context = bundle.final_context
    if not context or (match := _VALID_KEYS.search(context)) is None:
        return result
    selected_keys = match.group(1).split(", ")
    context_body = context[:match.start()]
    labels = list(_CONTEXT_LABEL.finditer(context_body))
    if (len(set(selected_keys)) != len(selected_keys)
            or [label.group(1) for label in labels] != selected_keys):
        return result
    by_context_id = {}
    for evidence in bundle.evidence:
        context_id = evidence.upstream_context_id
        if context_id in by_context_id:
            return result
        by_context_id[context_id] = evidence
    prompts = bundle.run_config.get("prompts", {})
    if not isinstance(prompts, dict):
        return result
    policy = prompts.get("context")
    original_policy = policy in (CONTEXT_POLICY, LEGACY_CONTEXT_POLICY)
    if "context" in prompts and not original_policy and policy != {"version": "upstream_summary"}:
        return result
    # Optional background is explicitly quoted and is never counted as paper evidence.
    if original_policy and '\n\nExtra background (quoted data, not source evidence):\n' in context_body:
        context_body, background = context_body.split('\n\nExtra background (quoted data, not source evidence):\n', 1)
        if any(not line.startswith('    ') for line in background.splitlines()):
            return result
        labels = list(_CONTEXT_LABEL.finditer(context_body))
    included_ids = set()
    for index, context_id in enumerate(selected_keys):
        evidence = by_context_id.get(context_id)
        member = member_by_id.get(evidence.document_id) if evidence else None
        if member is None or evidence.upstream_dockey != member.versioned_id:
            return result
        section_end = labels[index + 1].start() if index + 1 < len(labels) else len(context_body)
        section = context_body[labels[index].start():section_end]
        if original_policy:
            expected = evidence_excerpt(context_id, evidence.original_text, evidence.summary)
            if re.fullmatch(re.escape(expected) + r"\nFrom [^\n]*\n*", section) is None:
                return result
        elif section.startswith((f"{context_id}: Original passage (", f"{context_id}: Summary fallback (")):
            return result
        elif not evidence.summary or evidence.summary not in section:
            return result
        included_ids.add(member.document_id)
    document_identity = lambda member: {
        "document_id": member.document_id, "versioned_id": member.versioned_id,
    }
    ordered = sorted(members, key=lambda member: member.position)
    return {
        **result, "status": "known",
        "included_documents": [document_identity(member) for member in ordered
                               if member.document_id in included_ids],
        "missing_documents": [document_identity(member) for member in ordered
                              if member.document_id not in included_ids],
    }


def summarize_document_usage(bundle: AnswerBundle, frozen_members) -> dict:
    """Keep frozen scope, saved evidence, final context, and visible citations distinct."""

    members = sorted(frozen_members, key=lambda member: member.position)
    member_by_id = {member.document_id: member for member in members}
    known_members = bool(members) and len(member_by_id) == len(members)
    answer_config = bundle.run_config.get("answer")
    limit = answer_config.get("answer_max_sources") if isinstance(answer_config, dict) else None
    result = {
        "basis": "saved_answer_records", "total_documents": len(members) if known_members else None,
        "evidence_documents": None, "evidence_fragments": None,
        "context_documents": None, "context_fragments": None,
        "cited_documents": None,
        "evidence_fragment_limit": limit if type(limit) is int and limit > 0 else None,
        "context_selection": bundle.run_config.get("context_selection"),
        "documents": [
            {"document_id": member.document_id, "versioned_id": member.versioned_id,
             "has_evidence": None, "in_context": None, "is_cited": None}
            for member in members if known_members
        ],
    }
    allowed = bundle.allowed_documents
    if (not known_members or len(allowed) != len(members)
            or any(not isinstance(document, dict) or any(
                not isinstance(document.get(key), str)
                for key in ("document_id", "versioned_id", "dockey", "sha256")
            ) or document["dockey"] != document["versioned_id"] for document in allowed)):
        return result
    allowed_keys = {(document["document_id"], document["versioned_id"], document["sha256"])
                    for document in allowed}
    frozen_keys = {(member.document_id, member.versioned_id, member.sha256) for member in members}
    if len(allowed_keys) != len(members) or allowed_keys != frozen_keys:
        return result

    def mapped_document(evidence):
        member = member_by_id.get(evidence.document_id) if evidence else None
        return member if member and evidence.upstream_dockey == member.versioned_id else None

    def identities(document_ids):
        return [{"document_id": member.document_id, "versioned_id": member.versioned_id}
                for member in members if member.document_id in document_ids]

    by_id = {evidence.evidence_id: evidence for evidence in bundle.evidence}
    evidence_ids = {evidence.document_id for evidence in bundle.evidence}
    if (len(by_id) == len(bundle.evidence)
            and len({e.upstream_context_id for e in bundle.evidence}) == len(bundle.evidence)
            and all(mapped_document(evidence) for evidence in bundle.evidence)):
        result["evidence_documents"] = identities(evidence_ids)
        result["evidence_fragments"] = len(bundle.evidence)

    context = summarize_saved_context_coverage(bundle, members)
    if context["status"] == "known":
        result["context_documents"] = context["included_documents"]
        result["context_fragments"] = len(_VALID_KEYS.search(bundle.final_context).group(1).split(", "))

    cited_ids = set()
    citations_known = bool(bundle.raw_answer) and len(by_id) == len(bundle.evidence)
    for citation in bundle.citations:
        evidence = by_id.get(citation.evidence_id)
        if (citation.mapping_status != "mapped" or not mapped_document(evidence)
                or citation.upstream_context_id != evidence.upstream_context_id
                or not 0 <= citation.start < citation.end <= len(bundle.raw_answer)
                or bundle.raw_answer[citation.start:citation.end] != citation.upstream_context_id):
            citations_known = False
            break
        cited_ids.add(evidence.document_id)
    if citations_known:
        result["cited_documents"] = identities(cited_ids)

    for document in result["documents"]:
        for flag, stage in (("has_evidence", "evidence_documents"),
                            ("in_context", "context_documents"), ("is_cited", "cited_documents")):
            if result[stage] is not None:
                document[flag] = any(item["document_id"] == document["document_id"] for item in result[stage])
    return result


def build_scoped_answer_bundle(
    run_id: str, scope: AnswerScope, execution: AnswerExecution,
) -> AnswerBundle:
    """Map raw Context IDs before persistence; never infer evidence from display text."""

    response = execution.response
    session = response.session if response is not None else None
    report = {
        "schema_version": 2,
        "status": response.status.value if response is not None else "unavailable",
        "scope_violation": execution.failure_kind == "scope_violation",
        "documents": [
            {
                "document_id": document.document_id,
                "dockey": document.versioned_id,
                "versioned_id": document.versioned_id,
                "sha256": document.sha256,
            }
            for document in scope.documents
        ],
        "token_counts": session.token_counts if session is not None else {},
        "run_config": {"coverage": "mapping_only"},
        "scope_observation": {
            "revision_id": scope.revision_id,
            "index_artifact_id": scope.index_artifact_id,
            "execution_started_at": execution.started_at,
            "execution_finished_at": execution.finished_at,
            "failure_kind": execution.failure_kind,
            "error_message": execution.error_message,
            "snapshots": [
                {
                    "stage": snapshot.stage,
                    "loaded_dockeys": list(snapshot.loaded_dockeys),
                    "text_dockeys": list(snapshot.text_dockeys),
                    "context_dockeys": list(snapshot.context_dockeys),
                    "final_context": snapshot.final_context,
                }
                for snapshot in execution.snapshots
            ],
            "final_context_coverage": execution.final_context_coverage,
        },
    }
    session_data = {
        "question": execution.question,
        "raw_answer": session.raw_answer if session is not None else "",
        "answer": session.answer if session is not None else "",
        "formatted_answer": session.formatted_answer if session is not None else "",
        "has_successful_answer": (
            session.has_successful_answer if session is not None else None
        ),
        "context": execution.final_context,
        "contexts": [
            {
                "id": context.upstream_context_id,
                "context": context.summary,
                "score": context.score,
                "text": {
                    "text": context.original_text,
                    "name": context.upstream_locator,
                    "doc": {"dockey": context.upstream_dockey},
                },
            }
            for context in execution.contexts
        ],
    }
    return PaperQAAdapter.to_answer_bundle(run_id, report, session_data)
