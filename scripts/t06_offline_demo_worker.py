"""Process one T06 demo Job with fixed local fixtures; never contact upstreams.

Only accepts a database and data root under this project's ignored .tooling folder.
Use scripts/job_worker.py --live for real services after the T07 acceptance plan.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx

from backend.app.db import open_database
from backend.app.jobs.dispatcher import run_one_job
from backend.app.jobs.process_lock import worker_singleton
from backend.app.search.client import ArxivClient, TransportResponse
from tests.test_collection_index import FakeEmbedding, settings
from tests.test_document_preparation import text_pdf_bytes
from tests.test_qa_execution import offline_answer


class FixedAtomTransport:
    def send(self, _request):
        fixture = PROJECT_ROOT / "tests" / "fixtures" / "arxiv" / "normal_with_versions.xml"
        return TransportResponse(
            status_code=200, body=fixture.read_bytes(),
            headers=(("Content-Type", "application/atom+xml"),),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / ".tooling" / "t06-demo.sqlite")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / ".tooling" / "t06-demo-data")
    parser.add_argument("--fail-pdf-version", default="", help="Optional fixed version to fail in this demo run.")
    parser.add_argument("--watch", action="store_true", help="Keep processing isolated fixture Jobs.")
    args = parser.parse_args(argv)
    allowed = (PROJECT_ROOT / ".tooling").resolve()
    database_path = args.database.resolve()
    data_root = args.data_root.resolve()
    if not database_path.is_relative_to(allowed) or not data_root.is_relative_to(allowed):
        parser.error("The offline demo can only use paths under the project's .tooling directory.")
    if not database_path.is_file():
        parser.error("Initialize the demo database with scripts/api_server.py --init-db first.")

    def pdf(request):
        if args.fail_pdf_version and request.url.path.endswith(args.fail_pdf_version):
            return httpx.Response(404, content=b"demo PDF unavailable")
        return httpx.Response(
            200, headers={"Content-Type": "application/pdf"}, content=text_pdf_bytes(),
        )

    arxiv = ArxivClient(
        transport=FixedAtomTransport(),
        clock=lambda: datetime(2026, 9, 25, tzinfo=timezone.utc),
    )

    async def demo_answer(scope, question):
        execution = await offline_answer(scope, raw_citations=True)
        return replace(execution, question=question)

    with worker_singleton(database_path) as acquired:
        if not acquired:
            print("[BUSY] Another worker owns this demo database.")
            return 2
        engine = open_database(database_path, create_schema=False)
        try:
            with httpx.Client(transport=httpx.MockTransport(pdf)) as pdf_client:
                while True:
                    result = run_one_job(
                        engine, worker_id="offline-demo", data_root=data_root,
                        search_client=arxiv, import_client=pdf_client,
                        index_settings=settings(), index_embedding_model=FakeEmbedding(),
                        answer_settings=settings(), answer_executor=demo_answer,
                    )
                    if result:
                        print(f"[OFFLINE FIXTURE] {result.status}: kind={result.kind} job={result.job_id}", flush=True)
                    if not args.watch:
                        break
                    time.sleep(1)
        finally:
            engine.dispose()
    if result is None:
        print("[OFFLINE FIXTURE] No queued Job.")
        return 0
    print(
        f"[OFFLINE FIXTURE] {result.status}: kind={result.kind} "
        f"job={result.job_id} attempt={result.attempt_id} result={result.result_resource_id}"
    )
    return 0 if result.status == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
