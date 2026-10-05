"""Preview or verify one fixed-version PDF from a saved arXiv search."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ingestion.downloader import stage_pdf_download
from backend.app.ingestion.selection import InvalidDocumentSelection, select_fixed_version
from backend.app.ingestion.storage import publish_staged_pdf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check one saved arXiv version; --live downloads one PDF."
    )
    parser.add_argument("--search-run", required=True, type=Path)
    parser.add_argument("--versioned-id", required=True)
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "data" / "runs"
    )
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)

    try:
        source = select_fixed_version(args.search_run, args.versioned_id)
    except InvalidDocumentSelection as error:
        print(f"[FAIL] Selection: {error}")
        return 1

    print(f"[SELECTED] {source.title} [{source.versioned_id}]")
    print(f"[SNAPSHOT] {source.snapshot_sha256}")
    print(f"[PDF] {source.download_url}")
    if not args.live:
        print("[PREVIEW] No PDF request was sent. Add --live to verify it.")
        return 0

    run_dir = args.output_root / f"t04-pdf-{uuid4().hex[:12]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    responses: list[dict[str, str | int | None]] = []

    def record_response(response: httpx.Response) -> None:
        responses.append(
            {
                "url": str(response.request.url),
                "status_code": response.status_code,
                "location": response.headers.get("Location"),
            }
        )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_id": source.source_run_id,
        "snapshot_sha256": source.snapshot_sha256,
        "versioned_id": source.versioned_id,
        "title": source.title,
        "request_url": source.download_url,
        "responses": responses,
        "status": "running",
    }
    try:
        with httpx.Client(event_hooks={"response": [record_response]}) as client:
            with stage_pdf_download(
                source, client=client, staging_dir=run_dir / "staging"
            ) as staged:
                stored = publish_staged_pdf(staged, store_dir=run_dir / "blobs")
        report.update(
            status="success",
            sha256=stored.sha256,
            byte_count=stored.byte_count,
            page_count=stored.page_count,
            pdf_ref=stored.path.relative_to(run_dir).as_posix(),
            reused=stored.reused,
        )
        print(
            f"[PASS] {stored.page_count} pages, {stored.byte_count} bytes, "
            f"SHA-256 {stored.sha256}"
        )
        exit_code = 0
    except Exception as error:
        report.update(
            status="failed",
            error_kind=getattr(error, "kind", type(error).__name__),
            error_message=str(error),
        )
        print(f"[FAIL] {report['error_kind']}: {error}")
        exit_code = 1

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[REPORT] {run_dir / 'report.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
