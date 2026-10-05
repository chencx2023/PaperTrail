"""First PaperQA document QA run. Add --live to permit model API calls.

Uses Docs.aadd/aquery, not the autonomous agent loop. Cached document vectors
are reused; query embedding, evidence summaries and answers still cost tokens.
"""

import argparse
import asyncio
import hashlib
import json
import math
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from check_models import ROOT, load_config
from model_config import build_models


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_vectors(texts, dimensions):
    for text in texts:
        vector = text.embedding
        if (not vector or len(vector) != dimensions
                or any(not math.isfinite(x) for x in vector)
                or not any(x != 0 for x in vector)):
            raise ValueError("Invalid cached or newly generated document vector.")


async def run(args, config, files, models, run_dir, report):
    from paperqa import Docs
    from paperqa.types import Doc, Text

    settings, llm, summary_llm, embedding = models
    docs = Docs()
    for path in files:
        spec = {
            "schema": 1, "filename": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "paperqa": version("paper-qa"), "lmi": version("fhlmi"),
            "embedding_model": config["EMBEDDING_MODEL"],
            "embedding_base_url": config["EMBEDDING_BASE_URL"],
            "dimensions": config["EMBEDDING_DIMENSIONS"],
            "chunk_chars": 3000, "overlap": 200, "multimodal": False,
        }
        cache_id = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        cache = ROOT / "data" / "indexes" / "smoke" / f"{cache_id}.json"
        if cache.exists():
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved["spec"] != spec:
                raise ValueError("Cache metadata mismatch.")
            doc = Doc.model_validate(saved["doc"])
            texts = [Text.model_validate(t) for t in saved["texts"]]
            if not texts:
                raise ValueError("Cached document contains no text.")
            validate_vectors(texts, embedding.ndim)
            added = await docs.aadd_texts(texts, doc, settings=settings, embedding_model=embedding)
            status = "cached" if added else "duplicate"
        else:
            print(f"[IMPORT] {path.name}", flush=True)
            before = len(docs.texts)
            await docs.aadd(
                path, settings=settings, llm_model=llm, embedding_model=embedding,
                docname="P" + spec["sha256"][:10],
                # Deliberately use a file label, not invented bibliographic metadata.
                citation=f"Local PDF: {path.name}",
            )
            texts = docs.texts[before:]
            if not texts:
                raise ValueError("Document import produced no new chunks; check duplicate PDFs or parsing.")
            validate_vectors(texts, embedding.ndim)
            write_json(cache, {"spec": spec, "doc": texts[0].doc.model_dump(mode="json"),
                               "texts": [t.model_dump(mode="json") for t in texts]})
            status = "imported"
        report["documents"].append({**spec, "status": status, "chunks": len(texts)})
        print(f"[{status.upper()}] {path.name}: {len(texts)} chunks", flush=True)
        write_json(run_dir / "run.json", report)

    print(f"[QUERY] {len(docs.docs)} documents; retrieving and summarizing evidence...", flush=True)
    session = await docs.aquery(
        args.question, settings=settings, llm_model=llm,
        summary_llm_model=summary_llm, embedding_model=embedding,
    )
    write_json(run_dir / "session.json", session.model_dump(mode="json"))
    (run_dir / "answer.md").write_text(session.formatted_answer, encoding="utf-8")
    (run_dir / "context.txt").write_text(session.context, encoding="utf-8")
    report.update(status="completed", evidence_count=len(session.contexts),
                  token_counts=session.token_counts,
                  note="Completed means the pipeline returned; manually verify the answer and citations. "
                       "Token counts may omit ingestion/embedding; provider billing is authoritative.")
    print(f"[COMPLETED] {len(session.contexts)} evidence items; review answer.md and session.json.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Permit billable model API calls.")
    parser.add_argument("--all-papers", action="store_true", help="Use all PDFs directly in data/papers.")
    parser.add_argument("--question", default=(
        "According to the Mem0 paper, how are memories extracted and updated? "
        "Describe the main steps and cite the supporting evidence."
    ))
    args = parser.parse_args()
    config = {}
    run_dir = None
    report = {"status": "started", "scope": "Docs QA, not autonomous agent",
              "started_at": datetime.now(timezone.utc).isoformat(),
              "question": args.question, "documents": []}
    started = time.monotonic()
    try:
        config = load_config("all")
        directory = ROOT / "data" / "papers"
        files = sorted(directory.glob("*.pdf")) if args.all_papers else [directory / "Mem0.pdf"]
        if not files or any(not p.is_file() for p in files):
            raise ValueError("No PDFs found, or Mem0.pdf is missing from data/papers.")
        models = build_models(config)
        print(f"[READY] PaperQA {version('paper-qa')}; {len(files)} PDFs; embedding {config['EMBEDDING_DIMENSIONS']} dimensions.")
        if not args.live:
            print("[LOCAL ONLY] Configuration and model objects validated; no model API calls made.")
            print("Add --live to import the selected PDFs and answer one question.")
            return 0
        run_dir = ROOT / "data" / "runs" / ("paperqa-" + uuid.uuid4().hex[:12])
        report["models"] = {"llm": config["DEEPSEEK_MODEL"], "embedding": config["EMBEDDING_MODEL"]}
        report["limits"] = {"evidence_k": models[0].answer.evidence_k,
                            "answer_max_sources": models[0].answer.answer_max_sources, "llm_max_tokens": 1800,
                            "note": "Source limit counts evidence chunks, not unique papers."}
        write_json(run_dir / "run.json", report)
        print(f"Results: {run_dir}", flush=True)
        asyncio.run(asyncio.wait_for(run(args, config, files, models, run_dir, report), timeout=600))
        return 0
    except Exception as exc:
        # API/library errors can echo configuration: redact both credentials.
        message = str(exc)
        for key in ("DEEPSEEK_API_KEY", "EMBEDDING_API_KEY"):
            if config.get(key):
                message = message.replace(config[key], "[REDACTED]")
        report.update(status="failed", error_type=type(exc).__name__, error=message[:800])
        print(f"[FAIL] {type(exc).__name__}: {message[:800]}")
        return 1
    finally:
        if run_dir is not None:
            report["seconds"] = round(time.monotonic() - started, 2)
            write_json(run_dir / "run.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
