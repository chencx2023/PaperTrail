import asyncio
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.api import create_app
from backend.app.collections.service import SelectedSource, set_collection_selection
from backend.app.db import open_database
from backend.app.ingestion.batch import get_import_report, stage_import_job
from backend.app.jobs.claim import claim_next_job
from backend.app.jobs.dispatcher import run_one_job
from backend.app.jobs.import_submission import get_import_request, retry_import_job
from backend.app.jobs.service import get_attempt
from backend.app.jobs.worker import interrupt_expired_attempt, run_one_search
from backend.app.models import (
    Collection, CollectionImportItem, DocumentVersion, Job, JobAttemptResult, JobRequest, SearchRun,
)
from backend.app.search.client import ArxivClient, TransportResponse
from tests.test_document_preparation import text_pdf_bytes


class FixedTransport:
    def send(self, _request):
        fixture = Path(__file__).parent / "fixtures" / "arxiv" / "normal_with_versions.xml"
        return TransportResponse(
            status_code=200,
            body=fixture.read_bytes(),
            headers=(("Content-Type", "application/atom+xml"),),
        )


class ApiCollectionTest(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.database_path = self.root / "papertrail.sqlite"
        self.data_root = self.root / "data"
        self.engine = open_database(self.database_path)
        self.addCleanup(self.engine.dispose)
        self.client = TestClient(create_app(self.database_path, self.data_root))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def _search(self, *, key="search-1", complete=True):
        response = self.client.post(
            "/api/searches",
            json={"question": "如何评估 Agent 记忆？", "query_text": "LLM agent memory"},
            headers={"Idempotency-Key": key},
        )
        self.assertEqual(response.status_code, 202)
        search_id = response.json()["search_id"]
        if complete:
            client = ArxivClient(
                transport=FixedTransport(),
                clock=lambda: datetime(2026, 9, 25, 8, tzinfo=timezone.utc),
            )
            self.assertEqual(
                run_one_search(
                    self.engine, worker_id="test-worker", data_root=self.data_root, client=client
                ).status,
                "succeeded",
            )
        return search_id

    def _body(self, search_id, *versioned_ids):
        return {
            "selections": [
                {"search_id": search_id, "versioned_id": versioned_id}
                for versioned_id in versioned_ids
            ]
        }

    def _post(self, body, *, key="collection-1"):
        return self.client.post(
            "/api/collections", json=body, headers={"Idempotency-Key": key}
        )

    def _patch(self, collection_id, revision, search_id, *versioned_ids):
        return self.client.patch(
            f"/api/collections/{collection_id}/draft",
            json={
                "expected_selection_revision": revision,
                **self._body(search_id, *versioned_ids),
            },
        )

    def _post_import(self, collection_id, revision, *, key="reimport-1"):
        return self.client.post(
            f"/api/collections/{collection_id}/imports",
            json={"expected_selection_revision": revision},
            headers={"Idempotency-Key": key},
        )

    def _counts(self):
        with Session(self.engine) as session:
            return (
                session.scalar(select(func.count()).select_from(Collection)),
                session.scalar(select(func.count()).select_from(Job).where(Job.kind == "import")),
                session.scalar(select(func.count()).select_from(JobRequest)),
            )

    def test_create_atomically_saves_ordered_fixed_request_and_reopens(self):
        search_id = self._search()
        body = self._body(search_id, "2401.00001v3", "2401.00001v1")
        created = self._post(body)
        self.assertEqual(created.status_code, 202)
        result = created.json()
        self.assertEqual(result["selection_revision"], 1)
        self.assertEqual(result["status"], "queued")
        self.assertEqual(self._counts(), (1, 1, 1))

        request = get_import_request(self.engine, result["job_id"])
        self.assertEqual(request.collection_id, result["collection_id"])
        self.assertEqual(request.selection_revision, 1)
        self.assertEqual(request.source_search_ids, (search_id, search_id))
        self.assertEqual(
            [item.versioned_id for item in request.selections],
            ["2401.00001v3", "2401.00001v1"],
        )
        self.assertTrue(all(len(item.snapshot_sha256) == 64 for item in request.selections))
        with TestClient(create_app(self.database_path, self.data_root)) as reopened:
            collection = reopened.get(f"/api/collections/{result['collection_id']}")
        self.assertEqual(collection.status_code, 200)
        self.assertEqual(collection.json()["selection_revision"], 1)
        self.assertEqual(collection.json()["import_job_id"], result["job_id"])
        self.assertEqual(collection.json()["import_job_status"], "queued")
        self.assertIsNone(collection.json()["import_report"])
        self.assertEqual(
            [item["versioned_id"] for item in collection.json()["selections"]],
            ["2401.00001v3", "2401.00001v1"],
        )
        self.assertNotIn(str(self.root), collection.text)

    def test_lost_response_replay_survives_later_draft_edit(self):
        search_id = self._search()
        body = self._body(search_id, "2401.00001v3")
        self._post(body)  # Server committed; caller discards this response.
        with Session(self.engine) as session:
            original = session.scalar(select(Job).where(Job.kind == "import"))
            search = session.get(SearchRun, search_id)
            collection_id = original.resource_id
            run_dir = self.data_root / search.snapshot_ref.removesuffix("/run.json")
            job_id = original.id
        set_collection_selection(
            self.engine,
            collection_id,
            expected_revision=1,
            data_root=self.data_root,
            selections=(SelectedSource(run_dir, "2401.00001v1"),),
        )

        replay = self._post(body)
        self.assertEqual(replay.status_code, 202)
        self.assertEqual(replay.json()["collection_id"], collection_id)
        self.assertEqual(replay.json()["job_id"], job_id)
        self.assertEqual(self._counts(), (1, 1, 1))
        current = self.client.get(f"/api/collections/{collection_id}").json()
        self.assertEqual(current["selection_revision"], 2)
        self.assertIsNone(current["import_job_id"])

    def test_api_created_request_runs_in_existing_worker_without_real_pdf(self):
        search_id = self._search()
        created = self._post(self._body(search_id, "2401.00001v3")).json()
        requested = []

        def unavailable_pdf(request):
            requested.append(request.url.path)
            return httpx.Response(404, content=b"missing")

        with httpx.Client(transport=httpx.MockTransport(unavailable_pdf)) as client:
            executed = run_one_job(
                self.engine,
                worker_id="test-import-worker",
                data_root=self.data_root,
                import_client=client,
            )
        self.assertEqual(executed.kind, "import")
        self.assertEqual(executed.status, "succeeded")
        self.assertEqual(requested, ["/pdf/2401.00001v3"])
        with Session(self.engine) as session:
            binding = session.get(JobAttemptResult, executed.attempt_id)
        self.assertEqual(get_import_report(self.engine, binding.resource_id).status, "failed")
        current = self.client.get(f"/api/collections/{created['collection_id']}").json()
        self.assertEqual(current["import_job_id"], created["job_id"])
        self.assertEqual(current["import_job_status"], "succeeded")
        self.assertEqual(current["import_report"]["import_run_id"], executed.attempt_id)
        self.assertEqual(current["import_report"]["status"], "failed")
        self.assertEqual(current["import_report"]["items"][0]["status"], "download_failed")
        self.assertIsNone(current["import_report"]["items"][0]["document_id"])
        replay = self._post(self._body(search_id, "2401.00001v3"))
        self.assertEqual(replay.status_code, 202)
        self.assertEqual(replay.json()["collection_id"], created["collection_id"])
        self.assertEqual(replay.json()["job_id"], created["job_id"])
        self.assertEqual(replay.json()["status"], "succeeded")
        self.assertEqual(self._counts(), (1, 1, 1))

    def test_partial_report_reopens_without_exposing_raw_provider_error(self):
        search_id = self._search()
        created = self._post(
            self._body(search_id, "2401.00001v1", "2401.00001v3")
        ).json()

        def one_pdf(request):
            if request.url.path.endswith("v1"):
                return httpx.Response(
                    200, headers={"Content-Type": "application/pdf"}, content=text_pdf_bytes()
                )
            raise RuntimeError("secret-provider-detail")

        with httpx.Client(transport=httpx.MockTransport(one_pdf)) as client:
            executed = run_one_job(
                self.engine, worker_id="test-import-worker", data_root=self.data_root,
                import_client=client,
            )
        self.assertEqual(executed.status, "succeeded")
        with TestClient(create_app(self.database_path, self.data_root)) as reopened:
            response = reopened.get(f"/api/collections/{created['collection_id']}")
        self.assertEqual(response.status_code, 200)
        report = response.json()["import_report"]
        self.assertEqual(report["status"], "partial_failed")
        self.assertEqual(report["selection_revision"], 1)
        self.assertEqual([item["versioned_id"] for item in report["items"]], [
            "2401.00001v1", "2401.00001v3"
        ])
        self.assertEqual([item["status"] for item in report["items"]], [
            "succeeded", "download_failed"
        ])
        self.assertTrue(report["items"][0]["document_id"])
        self.assertIsNone(report["items"][1]["document_id"])
        with Session(self.engine) as session:
            document = session.get(DocumentVersion, report["items"][0]["document_id"])
            self.assertTrue(document.title)
            self.assertEqual(document.versioned_source_id, report["items"][0]["versioned_id"])
            self.assertEqual(report["items"][0].get("title"), document.title)
        self.assertIsNone(report["items"][1].get("title"))
        self.assertNotIn("secret-provider-detail", response.text)
        self.assertNotIn(str(self.root), response.text)

        with Session(self.engine) as session:
            search = session.get(SearchRun, search_id)
            run_dir = self.data_root / search.snapshot_ref.removesuffix("/run.json")
        set_collection_selection(
            self.engine, created["collection_id"], expected_revision=1,
            data_root=self.data_root,
            selections=(SelectedSource(run_dir, "2401.00001v1"),),
        )
        edited = self.client.get(f"/api/collections/{created['collection_id']}").json()
        self.assertEqual(edited["selection_revision"], 2)
        self.assertIsNone(edited["import_job_id"])
        self.assertIsNone(edited["import_report"])

    def test_network_permission_failure_explains_startup_without_exposing_raw_error(self):
        search_id = self._search()
        created = self._post(self._body(search_id, "2401.00001v3")).json()

        def denied(request):
            raise httpx.ConnectError(
                f"[WinError 10013] socket access denied; secret-provider-detail {self.root}",
                request=request,
            )

        with httpx.Client(transport=httpx.MockTransport(denied)) as client:
            run_one_job(self.engine, worker_id="test-import-worker", data_root=self.data_root, import_client=client)
        response = self.client.get(f"/api/collections/{created['collection_id']}")
        item = response.json()["import_report"]["items"][0]
        self.assertEqual(item["status"], "download_failed")
        self.assertIn("运行环境禁止联网", item["failure_reason"])
        self.assertIn("重新启动后台", item["failure_reason"])
        self.assertNotIn("secret-provider-detail", response.text)
        self.assertNotIn(str(self.root), response.text)
        self.assertIsNone(item["document_id"])

    def test_foreign_result_binding_is_rejected_instead_of_displayed(self):
        search_id = self._search()
        created = self._post(self._body(search_id, "2401.00001v3")).json()
        with httpx.Client(transport=httpx.MockTransport(
            lambda _request: httpx.Response(404, content=b"missing")
        )) as client:
            executed = run_one_job(
                self.engine, worker_id="test-import-worker", data_root=self.data_root,
                import_client=client,
            )
        with Session(self.engine) as session:
            session.get(JobAttemptResult, executed.attempt_id).resource_id = "another-attempt"
            session.commit()
        response = self.client.get(f"/api/collections/{created['collection_id']}")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], "INVALID_IMPORT_REPORT")

    def test_staged_old_attempt_is_not_a_current_report_during_retry(self):
        search_id = self._search()
        created = self._post(self._body(search_id, "2401.00001v3")).json()
        claimed = claim_next_job(
            self.engine, worker_id="old-worker", supported_kinds=("import",),
            lease_seconds=0.01,
        )
        request = get_import_request(self.engine, created["job_id"])
        with httpx.Client(transport=httpx.MockTransport(
            lambda _request: httpx.Response(404, content=b"missing")
        )) as client:
            staged = asyncio.run(stage_import_job(
                self.engine, request, attempt_id=claimed.attempt_id,
                data_root=self.data_root, client=client,
            ))
        running = self.client.get(f"/api/collections/{created['collection_id']}").json()
        self.assertEqual(running["import_job_status"], "running")
        self.assertIsNone(running["import_report"])
        self.assertEqual(get_import_report(self.engine, staged.run_id).status, "running")

        lease_until = datetime.fromisoformat(
            get_attempt(self.engine, claimed.attempt_id).lease_until.replace("Z", "+00:00")
        )
        self.assertTrue(interrupt_expired_attempt(
            self.engine, claimed.attempt_id, worker_stopped_confirmed=True,
            now=lease_until + timedelta(seconds=1),
        ))
        retry_import_job(self.engine, created["job_id"], expected_attempt_id=claimed.attempt_id)
        queued = self.client.get(f"/api/collections/{created['collection_id']}").json()
        self.assertEqual(queued["import_job_status"], "queued")
        self.assertIsNone(queued["import_report"])
        self.assertEqual(get_import_report(self.engine, staged.run_id).status, "running")

    def test_changed_report_item_is_not_displayed_as_valid(self):
        search_id = self._search()
        created = self._post(self._body(search_id, "2401.00001v3")).json()
        with httpx.Client(transport=httpx.MockTransport(
            lambda _request: httpx.Response(404, content=b"missing")
        )) as client:
            executed = run_one_job(
                self.engine, worker_id="test-import-worker", data_root=self.data_root,
                import_client=client,
            )
        with Session(self.engine) as session:
            item = session.scalar(select(CollectionImportItem).where(
                CollectionImportItem.import_run_id == executed.attempt_id
            ))
            item.versioned_id = "2401.00001v1"
            session.commit()
        response = self.client.get(f"/api/collections/{created['collection_id']}")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], "INVALID_IMPORT_REPORT")

    def test_patch_draft_advances_revision_without_rewriting_queued_import(self):
        search_id = self._search()
        created = self._post(self._body(search_id, "2401.00001v1", "2401.00001v3")).json()
        edited = self._patch(
            created["collection_id"], 1, search_id, "2401.00001v1", "cs/9901001v2"
        )
        self.assertEqual(edited.status_code, 200)
        self.assertEqual(edited.json()["selection_revision"], 2)
        current = self.client.get(f"/api/collections/{created['collection_id']}").json()
        self.assertEqual(current["selection_revision"], 2)
        self.assertEqual(
            [item["versioned_id"] for item in current["selections"]],
            ["2401.00001v1", "cs/9901001v2"],
        )
        self.assertIsNone(current["import_job_id"])
        self.assertIsNone(current["import_report"])
        old_request = get_import_request(self.engine, created["job_id"])
        self.assertEqual(
            [item.versioned_id for item in old_request.selections],
            ["2401.00001v1", "2401.00001v3"],
        )
        self.assertEqual(self._counts(), (1, 1, 1))

    def test_patch_rejects_stale_or_unverified_source_without_editing(self):
        search_id = self._search()
        collection_id = self._post(self._body(search_id, "2401.00001v3")).json()[
            "collection_id"
        ]
        stale = self._patch(collection_id, 0, search_id, "2401.00001v1")
        unknown = self._patch(collection_id, 1, "unknown", "2401.00001v1")
        wrong_version = self._patch(collection_id, 1, search_id, "2401.00001v9")
        queued_search = self._search(key="search-queued", complete=False)
        unfinished = self._patch(collection_id, 1, queued_search, "2401.00001v3")
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(wrong_version.status_code, 400)
        self.assertEqual(unfinished.status_code, 409)
        self.assertEqual(
            self.client.get(f"/api/collections/{collection_id}").json()["selection_revision"], 1
        )
        self.assertEqual(self._counts(), (1, 1, 1))

    def test_two_simultaneous_draft_edits_cannot_both_win(self):
        search_id = self._search()
        collection_id = self._post(self._body(search_id, "2401.00001v3")).json()[
            "collection_id"
        ]
        ready = Barrier(2)

        def edit(versioned_id):
            with TestClient(create_app(self.database_path, self.data_root)) as client:
                ready.wait(timeout=5)
                return client.patch(
                    f"/api/collections/{collection_id}/draft",
                    json={
                        "expected_selection_revision": 1,
                        **self._body(search_id, versioned_id),
                    },
                )

        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = [
                future.result(timeout=15)
                for future in (
                    pool.submit(edit, "2401.00001v1"),
                    pool.submit(edit, "cs/9901001v2"),
                )
            ]
        self.assertEqual(sorted(response.status_code for response in responses), [200, 409])
        current = self.client.get(f"/api/collections/{collection_id}").json()
        self.assertEqual(current["selection_revision"], 2)
        self.assertIsNone(current["import_job_id"])

    def test_reimport_uses_new_draft_while_old_job_keeps_original_list(self):
        search_id = self._search()
        created = self._post(self._body(search_id, "2401.00001v1")).json()
        collection_id = created["collection_id"]
        self.assertEqual(
            self._patch(collection_id, 1, search_id, "2401.00001v3").status_code, 200
        )
        resubmitted = self._post_import(collection_id, 2)
        self.assertEqual(resubmitted.status_code, 202)
        new_job_id = resubmitted.json()["job_id"]
        self.assertNotEqual(new_job_id, created["job_id"])
        self.assertEqual(resubmitted.json()["selection_revision"], 2)
        self.assertEqual(
            [item.versioned_id for item in get_import_request(self.engine, created["job_id"]).selections],
            ["2401.00001v1"],
        )
        self.assertEqual(
            [item.versioned_id for item in get_import_request(self.engine, new_job_id).selections],
            ["2401.00001v3"],
        )
        self.assertEqual(self._counts(), (1, 2, 2))
        requested = []

        def missing_pdf(request):
            requested.append(request.url.path)
            return httpx.Response(404, content=b"missing")

        with httpx.Client(transport=httpx.MockTransport(missing_pdf)) as client:
            first = run_one_job(
                self.engine, worker_id="test-import-worker", data_root=self.data_root,
                import_client=client,
            )
            between = self.client.get(f"/api/collections/{collection_id}").json()
            second = run_one_job(
                self.engine, worker_id="test-import-worker", data_root=self.data_root,
                import_client=client,
            )
        self.assertEqual(first.job_id, created["job_id"])
        self.assertEqual(second.job_id, new_job_id)
        self.assertEqual(requested, ["/pdf/2401.00001v1", "/pdf/2401.00001v3"])
        self.assertEqual(between["import_job_id"], new_job_id)
        self.assertIsNone(between["import_report"])
        current = self.client.get(f"/api/collections/{collection_id}").json()
        self.assertEqual(current["import_job_id"], new_job_id)
        self.assertEqual(current["import_report"]["items"][0]["versioned_id"], "2401.00001v3")

    def test_reimport_lost_response_replay_survives_newer_draft(self):
        search_id = self._search()
        collection_id = self._post(self._body(search_id, "2401.00001v1")).json()[
            "collection_id"
        ]
        self._patch(collection_id, 1, search_id, "2401.00001v3")
        self._post_import(collection_id, 2, key="reimport-lost")  # Caller loses response.
        with Session(self.engine) as session:
            original = session.scalar(select(Job).where(
                Job.kind == "import", Job.idempotency_key == "reimport-lost"
            ))
        self._patch(collection_id, 2, search_id, "2401.00001v1")

        replay = self._post_import(collection_id, 2, key="reimport-lost")
        changed = self._post_import(collection_id, 3, key="reimport-lost")
        self.assertEqual(replay.status_code, 202)
        self.assertEqual(replay.json()["job_id"], original.id)
        self.assertEqual(changed.status_code, 409)
        self.assertEqual(changed.json()["code"], "IDEMPOTENCY_CONFLICT")
        self.assertEqual(self._counts(), (1, 2, 2))
        current = self.client.get(f"/api/collections/{collection_id}").json()
        self.assertEqual(current["selection_revision"], 3)
        self.assertIsNone(current["import_job_id"])

    def test_reimport_requires_current_revision_and_key(self):
        search_id = self._search()
        collection_id = self._post(self._body(search_id, "2401.00001v1")).json()[
            "collection_id"
        ]
        self._patch(collection_id, 1, search_id, "2401.00001v3")
        stale = self._post_import(collection_id, 1, key="stale-reimport")
        missing_key = self.client.post(
            f"/api/collections/{collection_id}/imports",
            json={"expected_selection_revision": 2},
        )
        unknown = self._post_import("unknown", 2, key="unknown-reimport")
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.json()["code"], "STALE_SELECTION")
        self.assertEqual(missing_key.status_code, 400)
        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(self._counts(), (1, 1, 1))

    def test_concurrent_reimport_with_same_key_queues_once(self):
        search_id = self._search()
        collection_id = self._post(self._body(search_id, "2401.00001v1")).json()[
            "collection_id"
        ]
        self._patch(collection_id, 1, search_id, "2401.00001v3")
        ready = Barrier(2)

        def submit():
            with TestClient(create_app(self.database_path, self.data_root)) as client:
                ready.wait(timeout=5)
                return client.post(
                    f"/api/collections/{collection_id}/imports",
                    json={"expected_selection_revision": 2},
                    headers={"Idempotency-Key": "reimport-concurrent"},
                )

        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = [
                future.result(timeout=15)
                for future in (pool.submit(submit), pool.submit(submit))
            ]
        self.assertEqual([response.status_code for response in responses], [202, 202])
        self.assertEqual(responses[0].json(), responses[1].json())
        self.assertEqual(self._counts(), (1, 2, 2))

    def test_same_key_changed_selection_conflicts_without_orphan_collection(self):
        search_id = self._search()
        first = self._post(self._body(search_id, "2401.00001v3"))
        changed = self._post(self._body(search_id, "2401.00001v1"))
        self.assertEqual(first.status_code, 202)
        self.assertEqual(changed.status_code, 409)
        self.assertEqual(changed.json()["code"], "IDEMPOTENCY_CONFLICT")
        self.assertEqual(self._counts(), (1, 1, 1))

    def test_unfinished_unknown_or_damaged_search_creates_no_collection(self):
        search_id = self._search(complete=False)
        queued = self._post(self._body(search_id, "2401.00001v3"), key="queued")
        self.assertEqual(queued.status_code, 409)
        self.assertEqual(self._counts(), (0, 0, 0))

        client = ArxivClient(
            transport=FixedTransport(),
            clock=lambda: datetime(2026, 9, 25, 8, tzinfo=timezone.utc),
        )
        run_one_search(self.engine, worker_id="test-worker", data_root=self.data_root, client=client)
        unknown_version = self._post(self._body(search_id, "2401.00001v9"), key="unknown")
        self.assertEqual(unknown_version.status_code, 400)
        with Session(self.engine) as session:
            search = session.get(SearchRun, search_id)
            response_path = self.data_root / search.snapshot_ref.removesuffix("run.json") / "response.atom"
        response_path.write_bytes(b"corrupt")
        damaged = self._post(self._body(search_id, "2401.00001v3"), key="damaged")
        self.assertEqual(damaged.status_code, 409)
        self.assertEqual(damaged.json()["code"], "INVALID_SNAPSHOT")
        self.assertEqual(self._counts(), (0, 0, 0))

    def test_bad_collection_requests_do_not_create_work(self):
        search_id = self._search()
        missing_key = self.client.post(
            "/api/collections", json=self._body(search_id, "2401.00001v3")
        )
        empty = self._post({"selections": []}, key="empty")
        unknown = self._post(self._body("missing", "2401.00001v3"), key="missing")
        duplicate = self._post(
            self._body(search_id, "2401.00001v3", "2401.00001v3"), key="duplicate"
        )
        self.assertEqual(missing_key.json()["code"], "INVALID_IDEMPOTENCY_KEY")
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(duplicate.status_code, 400)
        self.assertEqual(self._counts(), (0, 0, 0))

    def test_damaged_saved_collection_returns_standard_error(self):
        search_id = self._search()
        collection_id = self._post(self._body(search_id, "2401.00001v3")).json()[
            "collection_id"
        ]
        with Session(self.engine) as session:
            session.get(Collection, collection_id).selected_json = '{"bad": true}'
            session.commit()
        damaged = self.client.get(f"/api/collections/{collection_id}")
        self.assertEqual(damaged.status_code, 409)
        self.assertEqual(damaged.json()["code"], "INVALID_COLLECTION")
        self.assertNotIn(str(self.root), damaged.text)

    def test_concurrent_same_key_creates_one_collection_and_import_job(self):
        search_id = self._search()
        body = self._body(search_id, "2401.00001v3")
        ready = Barrier(2)

        def submit():
            with TestClient(create_app(self.database_path, self.data_root)) as client:
                ready.wait(timeout=5)
                return client.post(
                    "/api/collections",
                    json=body,
                    headers={"Idempotency-Key": "same-concurrent-key"},
                )

        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = [future.result(timeout=15) for future in [pool.submit(submit) for _ in range(2)]]
        self.assertEqual([response.status_code for response in responses], [202, 202])
        self.assertEqual(responses[0].json(), responses[1].json())
        self.assertEqual(self._counts(), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
