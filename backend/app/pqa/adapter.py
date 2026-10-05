"""Versioned application output boundary. Loading an answer never calls a model.

This is the result-conversion part of the planned adapter; job orchestration and
prepare/answer service methods belong to T04/T05, not to this contract probe.
"""
import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    evidence_id: str
    upstream_context_id: str
    document_id: str | None
    upstream_dockey: str
    summary: str
    original_text: str | None
    score: float | None
    upstream_locator: str | None
    locator_status: str
    pdf_page: int | None = None


class Citation(BaseModel):
    upstream_context_id: str
    start: int
    end: int
    evidence_id: str | None
    mapping_status: str


class AnswerBundle(BaseModel):
    schema_version: int = 1
    run_id: str
    question: str
    job_status: str
    answer_outcome: str
    upstream_status: str
    termination_reason: str
    raw_answer: str
    display_answer: str
    final_context: str | None
    allowed_documents: list[dict]
    evidence: list[Evidence]
    citations: list[Citation]
    usage: dict
    run_config: dict = Field(default_factory=dict)
    scope_observation: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class PaperQAAdapter:
    @staticmethod
    def to_answer_bundle(run_id, report, session):
        documents = report.get("documents", [])
        by_key = {}
        for doc in documents:
            key = doc.get("dockey")
            if key is None or not doc.get("sha256"):
                raise ValueError("Explicit dockey to PDF hash mapping is required.")
            if str(key) in by_key:
                raise ValueError("Ambiguous upstream document identity.")
            by_key[str(key)] = doc
        evidence, lookup, warnings = [], {}, []
        outside = False
        for context in session.get("contexts", []):
            cid = context["id"]
            if cid in lookup:
                raise ValueError("Duplicate upstream Context ID in one run.")
            text = context["text"]
            key = str(text["doc"]["dockey"])
            source = by_key.get(key)
            outside |= source is None
            eid = run_id + ":" + cid
            lookup[cid] = (eid, source is not None)
            evidence.append(Evidence(
                evidence_id=eid, upstream_context_id=cid,
                document_id=(source.get("document_id") or source["sha256"])
                if source else None, upstream_dockey=key,
                summary=context.get("context", ""), original_text=text.get("text") or None,
                score=context.get("score"), upstream_locator=text.get("name"),
                locator_status="unverified_upstream_label" if text.get("name") else "unknown"))
        raw = session.get("raw_answer") or ""
        citations = []
        # Preserve every token and offset, including unmatched IDs removed by the
        # upstream display formatter. This is not a claim-level support evaluator.
        for match in re.finditer(r"(?<![A-Za-z0-9_-])pqac-[A-Za-z0-9_-]+", raw):
            cid = match.group()
            entry = lookup.get(cid)
            citations.append(Citation(upstream_context_id=cid, start=match.start(), end=match.end(),
                                      evidence_id=entry[0] if entry else None,
                                      mapping_status=("mapped" if entry[1] else "outside_scope")
                                      if entry else "missing_context"))
        status = report.get("status", "unknown")
        job, outcome, reason = "failed", "unknown", "unknown"
        if status in ("success", "unsure", "truncated"):
            job = "succeeded"
            outcome = {"success": "reported_sufficient", "unsure": "insufficient", "truncated": "truncated"}[status]
            reason = "completed" if status != "truncated" else "unknown"
            if status == "success" and not session.get("has_successful_answer"):
                outcome = "unknown"
        elif status == "interrupted":
            job, reason = "interrupted", "interrupted"
        if not raw and not session.get("answer"):
            outcome = "no_answer"
        if outside or report.get("scope_violation"):
            job, outcome, reason = "failed", "scope_violation", "scope_violation"
        if any(c.mapping_status != "mapped" for c in citations):
            warnings.append("Some raw citation IDs cannot be mapped to allowed evidence.")
        if any(e.original_text is None for e in evidence):
            warnings.append("Some original evidence text is unavailable.")
        return AnswerBundle(
            schema_version=report.get("schema_version", 1),
            run_id=run_id, question=session.get("question", report.get("question", "")),
            job_status=job, answer_outcome=outcome, upstream_status=status, termination_reason=reason,
            raw_answer=raw, display_answer=session.get("formatted_answer") or session.get("answer", ""),
            final_context=session.get("context"), allowed_documents=documents,
            evidence=evidence, citations=citations,
            run_config=report.get("run_config", {"models": report.get("models", {}), "coverage": "partial_historical_config"}),
            scope_observation=report.get("scope_observation", {"loaded_dockeys": None, "coverage": "historical_count_only"}),
            usage={"coverage": "partial", "tokens_by_upstream_model": report.get("token_counts", session.get("token_counts", {})),
                   "requests_by_role": None, "embedding_tokens": None, "cost_usd": None,
                   "note": "Model aggregates cannot distinguish roles sharing a model. Embedding, retries and full request totals are not measured."},
            warnings=warnings)

    @staticmethod
    def export_run(directory: Path, cache_directory: Path | None = None):
        report = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        session_file = directory / "session.json"
        session = json.loads(session_file.read_text(encoding="utf-8")) if session_file.exists() else {}
        # Historical probes omitted dockey in manifests. Recover only from an
        # exact, hash-verified cache snapshot; never infer it from citation labels.
        for doc in report.get("documents", []):
            if "dockey" in doc:
                continue
            if cache_directory is None:
                raise ValueError("Historical run requires its original smoke cache.")
            for path in cache_directory.glob("*.json"):
                content = path.read_bytes()
                if hashlib.sha256(content).hexdigest() == doc.get("cache_sha256"):
                    cached = json.loads(content)
                    if cached["spec"]["sha256"] != doc["sha256"]:
                        raise ValueError("Cache PDF identity mismatch.")
                    doc["dockey"] = str(cached["doc"]["dockey"])
                    break
        bundle = PaperQAAdapter.to_answer_bundle(directory.name, report, session)
        target = directory / "answer_bundle.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(target)
        reloaded = AnswerBundle.model_validate_json(target.read_text(encoding="utf-8"))
        if reloaded != bundle:
            raise ValueError("Answer bundle round-trip mismatch.")
        return reloaded
