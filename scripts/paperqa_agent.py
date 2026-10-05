"""Native PaperQA agent over a local, immutable snapshot of cached PDFs.

Default: offline preflight. --index-only: build native index without API calls.
--live: allow billable agent, summary, answer and query embedding calls.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from check_models import ROOT, load_config
sys.path.insert(0, str(ROOT))
from backend.app.pqa.adapter import PaperQAAdapter
from model_config import build_settings
from paperqa_smoke import validate_vectors, write_json

from paperqa import Docs
from paperqa.agents.env import PaperQAEnvironment, settings_to_tools
from paperqa.agents.main import run_agent
from paperqa.agents.search import SearchIndex, get_directory_index
from paperqa.types import Doc, Text


def cached_papers(config, root=ROOT):
    """Match caches to current PDFs and embedding settings before reusing vectors."""
    papers = sorted((root / "data" / "papers").glob("*.pdf"))
    if not papers:
        raise ValueError("No PDFs found in data/papers.")
    result = []
    for paper in papers:
        spec = {
            "schema": 1, "filename": paper.name,
            "sha256": hashlib.sha256(paper.read_bytes()).hexdigest(),
            "paperqa": version("paper-qa"), "lmi": version("fhlmi"),
            "embedding_model": config["EMBEDDING_MODEL"],
            "embedding_base_url": config["EMBEDDING_BASE_URL"],
            "dimensions": config["EMBEDDING_DIMENSIONS"],
            "chunk_chars": 3000, "overlap": 200, "multimodal": False,
        }
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        cache = root / "data" / "indexes" / "smoke" / f"{key}.json"
        if not cache.is_file():
            raise ValueError(f"No matching smoke cache for {paper.name}; import it with paperqa_smoke.py first.")
        content = cache.read_bytes()
        saved = json.loads(content)
        if saved["spec"] != spec:
            raise ValueError(f"Cache metadata mismatch for {paper.name}.")
        doc = Doc.model_validate(saved["doc"])
        texts = [Text.model_validate(item) for item in saved["texts"]]
        if not texts or any(str(t.doc.dockey) != str(doc.dockey) for t in texts):
            raise ValueError(f"Missing or mismatched cached text for {paper.name}.")
        validate_vectors(texts, config["EMBEDDING_DIMENSIONS"])
        result.append({"spec": spec, "cache_sha256": hashlib.sha256(content).hexdigest(),
                       "doc": doc, "texts": texts})
    if len({str(p["doc"].dockey) for p in result}) != len(result):
        raise ValueError("Duplicate PDF content detected; keep one copy of each paper in data/papers.")
    return result


def configure_agent(settings, papers, root=ROOT, max_steps=8, timeout=300):
    manifest = [{**p["spec"], "cache_sha256": p["cache_sha256"], "dockey": str(p["doc"].dockey), "chunks": len(p["texts"])}
                for p in papers]
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:20]
    settings.agent.index.name = "local-" + fingerprint
    settings.agent.index.paper_directory = str(root / "data" / "papers")
    settings.agent.index.index_directory = str(root / "data" / "indexes" / "native-agent")
    settings.agent.index.recurse_subdirectories = False
    settings.agent.index.sync_with_paper_directory = False
    settings.agent.rebuild_index = False
    settings.agent.agent_type = "ToolSelector"
    settings.agent.tool_names = ["paper_search", "gather_evidence", "gen_answer", "complete"]
    settings.agent.search_count = min(3, len(papers))
    settings.agent.max_timesteps = max_steps
    settings.agent.timeout = timeout
    return manifest


async def prepare_index(settings, papers, manifest):
    index = SearchIndex(fields=[*SearchIndex.REQUIRED_FIELDS, "title", "year"],
                        index_name=settings.agent.index.name,
                        index_directory=settings.agent.index.index_directory)
    embedding = settings.get_embedding_model()
    for paper in papers:
        filename = paper["spec"]["filename"]
        docs = Docs()
        await docs.aadd_texts(paper["texts"], paper["doc"], settings=settings, embedding_model=embedding)
        await index.add_document({"title": Path(filename).stem, "year": "Unknown year",
                                  "file_location": filename, "body": "\n".join(t.text for t in docs.texts)},
                                 document=docs, lock_acquisition_max_retries=3)
    if index.changed:
        await index.save_index()
    # Reopen through the exact function used by paper_search; verify scope and search.
    opened = await get_directory_index(settings, build=False)
    expected = {p["spec"]["filename"] for p in papers}
    if set(await opened.index_files) != expected or await opened.count != len(expected):
        raise ValueError("Native index does not match the selected paper snapshot.")
    for paper in papers:
        stored = await opened.get_saved_object(paper["spec"]["filename"])
        if not isinstance(stored, Docs) or len(stored.texts) != len(paper["texts"]):
            raise ValueError("Native index document validation failed.")
    directory = Path(settings.agent.index.index_directory) / settings.agent.index.name
    write_json(directory / "manifest.json", manifest)
    print(f"[INDEX READY] {len(papers)} papers, {sum(len(p['texts']) for p in papers)} chunks: {directory}", flush=True)
    return opened


class Recorder:
    def __init__(self, directory, secrets, allowed_keys):
        self.directory = directory
        self.secrets = [s for s in secrets if s]
        self.allowed_keys = allowed_keys
        self.state = None
        self.planned_ids = set()
        self.tool_batches = 0
        self.stage = "starting"

    def clean(self, text):
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]")
        return text

    def event(self, kind, **data):
        payload = {"time": datetime.now(timezone.utc).isoformat(), "event": kind, **data}
        with (self.directory / "trajectory.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(self.clean(json.dumps(payload, ensure_ascii=False)) + "\n")

    def snapshot(self, state):
        self.state = state
        if not {str(k) for k in state.docs.docs}.issubset(self.allowed_keys):
            raise ValueError("Agent accessed a document outside the selected snapshot.")
        observed = [*state.docs.texts, *(c.text for c in state.session.contexts)]
        if any(str(t.doc.dockey) not in self.allowed_keys for t in observed):
            raise ValueError("Agent accessed text outside the selected snapshot.")
        data = state.session.model_dump(mode="json", exclude={
            "contexts": {"__all__": {"text": {"embedding", "media"}}},
        })
        # This is saved before AnswerResponse strips the evidence's original text.
        (self.directory / "session.json").write_text(
            self.clean(json.dumps(data, ensure_ascii=False, indent=2)), encoding="utf-8")
        return {"papers": len(state.docs.docs), "evidence": len(state.session.contexts),
                "loaded_dockeys": sorted(str(k) for k in state.docs.docs)}

    async def action(self, action, ledger):
        self.planned_ids.update(c.id for c in getattr(action, "tool_calls", []))
        self.event("agent_action", action=action.model_dump(mode="json"))


def recording_environment(recorder):
    class RecordedEnvironment(PaperQAEnvironment):
        async def reset(self):
            result = await super().reset()
            recorder.event("reset", **recorder.snapshot(self.state),
                           tools=[tool.info.name for tool in self.tools])
            return result

        async def exec_tool_calls(self, message, *args, **kwargs):
            ids = {c.id for c in message.tool_calls}
            origin = "agent" if ids and ids.issubset(recorder.planned_ids) else "framework_fallback"
            names = [c.function.name for c in message.tool_calls]
            recorder.tool_batches += 1
            recorder.stage = ", ".join(names)
            recorder.event("tools_started", origin=origin, action=message.model_dump(mode="json"))
            print(f"[TOOLS {recorder.tool_batches}] {origin}: {recorder.stage}", flush=True)
            started = time.monotonic()
            try:
                result = await super().exec_tool_calls(message, *args, **kwargs)
                counts = recorder.snapshot(self.state)
                recorder.event("tools_finished", origin=origin, seconds=round(time.monotonic()-started, 2),
                               observations=[m.model_dump(mode="json") for m in (result or [])], **counts)
                print(f"[STATE] {counts['papers']} papers, {counts['evidence']} evidence items", flush=True)
                return result
            finally:
                recorder.stage = "agent deciding next action"
    return RecordedEnvironment


async def execute_agent(settings, question, recorder):
    async def heartbeat():
        while True:
            await asyncio.sleep(15)
            print(f"[WAIT] {recorder.stage}", flush=True)
    task = asyncio.create_task(heartbeat())
    try:
        # Extra outer bound also covers native fallback generation after truncation.
        async with asyncio.timeout(settings.agent.timeout + 120):
            response = await run_agent(Docs(), question, settings, agent_type="ToolSelector",
                                       env_class=recording_environment(recorder),
                                       on_agent_action_callback=recorder.action)
        recorder.event("finished", status=response.status.value,
                       tool_history=response.session.tool_history)
        # Native fallback records its action after exec_tool_calls returns.
        # Merge final metadata without replacing the preserved original evidence.
        snapshot_path = recorder.directory / "session.json"
        if snapshot_path.exists():
            saved = json.loads(snapshot_path.read_text(encoding="utf-8"))
            for key in ("tool_history", "has_successful_answer", "token_counts"):
                saved[key] = getattr(response.session, key)
            snapshot_path.write_text(recorder.clean(json.dumps(saved, ensure_ascii=False, indent=2)),
                                     encoding="utf-8")
        return response
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--index-only", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--question", default="According to the Mem0 paper, how are memories extracted and updated? Describe the main steps and cite the supporting evidence.")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--run-id", help="Optional unique output name under data/runs (used by contract checks).")
    args = parser.parse_args()
    if args.max_steps < 1 or not 0 < args.timeout <= 3600:
        parser.error("max-steps must be positive; timeout must be in (0, 3600].")
    if args.run_id and not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.run_id):
        parser.error("run-id must contain 1-80 ASCII letters, digits, underscores or hyphens.")
    config, report, directory, recorder = {}, {}, None, None
    started = time.monotonic()
    try:
        config = load_config("all")
        secrets = [config["DEEPSEEK_API_KEY"], config["EMBEDDING_API_KEY"]]
        class SafeFormatter(logging.Formatter):
            def format(self, record):
                text = super().format(record)
                for secret in secrets:
                    text = text.replace(secret, "[REDACTED]")
                return text
        handler = logging.StreamHandler()
        handler.setFormatter(SafeFormatter("%(levelname)s: %(message)s"))
        logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)
        logging.getLogger("paperqa.agents.main").setLevel(logging.WARNING)
        settings = build_settings(config)
        papers = cached_papers(config)
        manifest = configure_agent(settings, papers, max_steps=args.max_steps, timeout=args.timeout)
        tools = settings_to_tools(settings)
        settings.make_aviary_tool_selector("ToolSelector")
        print(f"[READY] {len(papers)} cached papers; tools: {', '.join(t.info.name for t in tools)}")
        if not (args.live or args.index_only):
            print("[LOCAL ONLY] No API calls. Use --index-only to build the index, or --live to run the agent.")
            return 0
        asyncio.run(prepare_index(settings, papers, manifest))
        if args.index_only:
            print("[LOCAL ONLY] Native index built from existing vectors. No model API calls.")
            return 0
        candidate = ROOT / "data" / "runs" / (args.run_id or ("agent-" + uuid.uuid4().hex[:12]))
        candidate.mkdir(parents=True)
        directory = candidate  # Existing runs are never overwritten on mkdir failure.
        recorder = Recorder(directory, secrets, {str(p['doc'].dockey) for p in papers})
        report = {"status": "running", "question": args.question,
                  "started_at": datetime.now(timezone.utc).isoformat(), "documents": manifest,
                  "index": settings.agent.index.name, "tools": [t.info.name for t in tools],
                  "models": {"llm": config["DEEPSEEK_MODEL"], "embedding": config["EMBEDDING_MODEL"]},
                  "max_steps": args.max_steps, "timeout": args.timeout,
                  "run_config": {"paperqa": version("paper-qa"), "lmi": version("fhlmi"),
                                 "model_roles": {"agent": settings.agent.agent_llm, "answer": settings.llm,
                                                 "summary": settings.summary_llm, "embedding": settings.embedding},
                                 "llm_base_url": config["DEEPSEEK_BASE_URL"],
                                 "embedding_base_url": config["EMBEDDING_BASE_URL"],
                                 "thinking": "disabled", "max_output_tokens": 1800,
                                 "request_timeout": 90, "router_retries": 0,
                                 "evidence_k": settings.answer.evidence_k,
                                 "answer_max_sources": settings.answer.answer_max_sources,
                                 "summary_concurrency": settings.answer.max_concurrent_requests,
                                 "embedding_dimensions": config["EMBEDDING_DIMENSIONS"]},
                  "cost_estimate_usd": None,
                  "note": "Native local ToolSelector; steps are not an API-call or monetary budget. Pricing unavailable; use provider billing."}
        write_json(directory / "run.json", report)
        print(f"Results: {directory}", flush=True)
        response = asyncio.run(execute_agent(settings, args.question, recorder))
        write_json(directory / "response.json", response.model_dump(mode="json"))
        (directory / "answer.md").write_text(response.session.formatted_answer, encoding="utf-8")
        (directory / "context.txt").write_text(response.session.context, encoding="utf-8")
        report.update(status=response.status.value, tool_history=response.session.tool_history,
                      token_counts=response.session.token_counts, evidence_count=len(response.session.contexts))
        print(f"[FINISHED] Agent status: {response.status.value}. Review answer and trajectory.")
        return 0 if response.status.value == "success" else 2
    except KeyboardInterrupt:
        report.update(status="interrupted")
        print("[INTERRUPTED] Partial evidence and trajectory are preserved.")
        return 130
    except Exception as exc:
        message = str(exc)
        for key in ("DEEPSEEK_API_KEY", "EMBEDDING_API_KEY"):
            if config.get(key):
                message = message.replace(config[key], "[REDACTED]")
        report.update(status="failed", error_type=type(exc).__name__, error=message[:1000])
        print(f"[FAIL] {type(exc).__name__}: {message[:1000]}")
        return 1
    finally:
        if directory:
            report["seconds"] = round(time.monotonic() - started, 2)
            if recorder and recorder.state:
                report["scope_observation"] = {
                    "loaded_dockeys": sorted(str(k) for k in recorder.state.docs.docs),
                    "coverage": "loaded_docs_and_texts_checked_after_tools; internal_candidate_ranking_unobserved"}
            write_json(directory / "run.json", report)
            try:
                PaperQAAdapter.export_run(directory)
            except Exception as exc:
                # Keep original run artifacts intact if result conversion fails.
                report["bundle_error"] = type(exc).__name__
                write_json(directory / "run.json", report)
                print(f"[BUNDLE FAILED] {type(exc).__name__}; raw results preserved.")


if __name__ == "__main__":
    raise SystemExit(main())
