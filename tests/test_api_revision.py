"""T06 confirmation binds a frozen revision to the current import report."""

import tempfile
import unittest
import os
import subprocess
import sys
from dataclasses import replace
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier, Event
from unittest.mock import patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.api import create_app
from backend.app.db import open_database
from backend.app.jobs.dispatcher import run_one_job
from backend.app.jobs.claim import claim_next_job
from backend.app.jobs.index_worker import execute_claimed_index
from backend.app.jobs.ownership import LostClaim
from backend.app.jobs.worker import interrupt_expired_attempt
from backend.app.jobs.ownership import finish_active_attempt
from backend.app.collections.indexing import build_collection_index
from backend.app.evidence.persistence import save_answer_run
from backend.app.evidence.service import build_scoped_answer_bundle
from backend.app.qa.scope import prepare_answer_scope
from uuid import uuid4
from backend.app.models import AnswerRun, AnswerSubmission, IndexArtifact, JobAttempt
from backend.app.models import CollectionRevision, Job, JobRequest
from backend.app.search.client import ArxivClient, TransportResponse
from tests.test_document_preparation import text_pdf_bytes
from tests.test_collection_index import FakeEmbedding, settings
from tests.test_qa_execution import offline_answer


class AtomTransport:
    def send(self, _request):
        fixture = Path(__file__).parent / "fixtures" / "arxiv" / "normal_with_versions.xml"
        return TransportResponse(200, fixture.read_bytes(), (("Content-Type", "application/atom+xml"),))


class ApiRevisionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.database_path = self.root / "papertrail.sqlite"
        self.data_root = self.root / "data"
        self.engine = open_database(self.database_path)
        self.addCleanup(self.engine.dispose)
        self.client = TestClient(create_app(self.database_path, self.data_root))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def imported(self, *, partial=False):
        search = self.client.post(
            "/api/searches", headers={"Idempotency-Key": "search"},
            json={"question": "如何评估记忆？", "query_text": "LLM agent memory"},
        ).json()
        arxiv = ArxivClient(transport=AtomTransport(), clock=lambda: datetime(2026, 9, 25, tzinfo=timezone.utc))
        run_one_job(self.engine, worker_id="search", data_root=self.data_root, search_client=arxiv)
        versions = ["2401.00001v1", "2401.00001v3"] if partial else ["2401.00001v1"]
        created = self.client.post(
            "/api/collections", headers={"Idempotency-Key": "collection"},
            json={"selections": [{"search_id": search["search_id"], "versioned_id": v} for v in versions]},
        ).json()

        def pdf(request):
            if partial and request.url.path.endswith("v3"):
                return httpx.Response(404)
            return httpx.Response(200, headers={"Content-Type": "application/pdf"}, content=text_pdf_bytes())

        with httpx.Client(transport=httpx.MockTransport(pdf)) as client:
            run_one_job(self.engine, worker_id="import", data_root=self.data_root, import_client=client)
        state = self.client.get(f"/api/collections/{created['collection_id']}").json()
        return created, state

    def confirm(self, created, state, document_ids, *, key="confirm"):
        report = state["import_report"]
        return self.client.post(
            f"/api/collections/{created['collection_id']}/revisions",
            headers={"Idempotency-Key": key},
            json={"expected_selection_revision": state["selection_revision"],
                  "import_run_id": report["import_run_id"], "document_ids": document_ids},
        )

    def recover_cli(self, job_id, *flags, script="recover_job.py"):
        return subprocess.run(
            [
                sys.executable, f"scripts/{script}",
                "--database", str(self.database_path), "--job-id", job_id,
                *flags,
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    def test_generic_recovery_cli_import_requires_interruption_before_retry(self):
        created, state = self.imported()
        submitted = self.client.post(
            f"/api/collections/{created['collection_id']}/imports",
            headers={"Idempotency-Key": "recover-import"},
            json={"expected_selection_revision": state["selection_revision"]},
        )
        self.assertEqual(submitted.status_code, 202, submitted.text)
        job_id = submitted.json()["job_id"]
        old = claim_next_job(
            self.engine, worker_id="stopped-import",
            supported_kinds=("import",), lease_seconds=0.05,
        )
        self.assertEqual(old.job_id, job_id)

        legacy = self.recover_cli(job_id, script="recover_search_job.py")
        self.assertEqual(legacy.returncode, 1, legacy.stdout)
        self.assertIn("[NOT FOUND] search job", legacy.stdout)

        preview = self.recover_cli(job_id)
        self.assertEqual(preview.returncode, 0, preview.stderr)
        self.assertIn("kind=import", preview.stdout)
        self.assertIn("[PREVIEW]", preview.stdout)
        rejected = self.recover_cli(job_id, "--retry")
        self.assertEqual(rejected.returncode, 1, rejected.stdout)
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}").json()["status"], "running")
        interrupted = self.recover_cli(job_id, "--confirm-worker-stopped")
        self.assertEqual(interrupted.returncode, 0, interrupted.stdout)
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}").json()["status"], "interrupted")
        queued = self.recover_cli(job_id, "--retry")
        self.assertEqual(queued.returncode, 0, queued.stdout)
        self.assertIn("[QUEUED]", queued.stdout)
        current = self.client.get(f"/api/jobs/{job_id}").json()
        self.assertEqual(current["status"], "queued")
        self.assertNotEqual(current["active_attempt_id"], old.attempt_id)

    def test_generic_recovery_cli_requeues_index(self):
        created, state = self.imported()
        document_id = state["import_report"]["items"][0]["document_id"]
        submitted = self.confirm(created, state, [document_id]).json()
        job_id = submitted["job_id"]
        old = claim_next_job(
            self.engine, worker_id="stopped-index",
            supported_kinds=("index",), lease_seconds=0.05,
        )
        self.assertEqual(old.job_id, job_id)

        preview = self.recover_cli(job_id)
        self.assertEqual(preview.returncode, 0, preview.stderr)
        self.assertIn("kind=index", preview.stdout)
        self.assertEqual(self.recover_cli(job_id, "--confirm-worker-stopped").returncode, 0)
        queued = self.recover_cli(job_id, "--retry")
        self.assertEqual(queued.returncode, 0, queued.stdout)
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}").json()["status"], "queued")

    def test_generic_recovery_cli_answer_reports_new_identity(self):
        revision_id, artifact_id = self.ready_revision()
        submitted = self.client.post(
            "/api/answers", headers={"Idempotency-Key": "recover-answer"},
            json={"revision_id": revision_id, "index_artifact_id": artifact_id,
                  "question": "What is studied?"},
        ).json()
        job_id = submitted["job_id"]
        original_answer_id = submitted["answer_run_id"]
        old = claim_next_job(
            self.engine, worker_id="stopped-answer",
            supported_kinds=("answer",), lease_seconds=0.05,
        )
        self.assertEqual(old.job_id, job_id)

        preview = self.recover_cli(job_id)
        self.assertEqual(preview.returncode, 0, preview.stderr)
        self.assertIn("kind=answer", preview.stdout)
        self.assertIn(original_answer_id, preview.stdout)
        self.assertEqual(self.recover_cli(job_id, "--confirm-worker-stopped").returncode, 0)
        queued = self.recover_cli(job_id, "--retry")
        self.assertEqual(queued.returncode, 0, queued.stdout)
        current = self.client.get(f"/api/jobs/{job_id}").json()
        self.assertEqual(current["status"], "queued")
        self.assertNotEqual(current["resource_id"], original_answer_id)
        self.assertIn(f"answer_run_id={current['resource_id']}", queued.stdout)
        self.assertIn(f"previous_answer_run_id={original_answer_id}", queued.stdout)
        self.assertEqual(
            self.client.get(f"/api/answers/{original_answer_id}").json()["status"],
            "interrupted",
        )
        self.assertEqual(
            self.client.get(f"/api/answers/{current['resource_id']}").json()["status"],
            "queued",
        )

    def test_partial_report_explicit_subset_is_atomically_queued_and_replayed(self):
        created, state = self.imported(partial=True)
        document_id = state["import_report"]["items"][0]["document_id"]
        first = self.confirm(created, state, [document_id])
        self.assertEqual(first.status_code, 202, first.text)
        payload = first.json()
        self.assertEqual(payload["status"], "queued")
        self.assertEqual(self.confirm(created, state, [document_id]).json(), payload)
        with Session(self.engine) as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(CollectionRevision)), 1)
            self.assertEqual(session.scalar(select(func.count()).select_from(Job).where(Job.kind == "index")), 1)
            self.assertIsNotNone(session.get(JobRequest, payload["job_id"]))
        read = self.client.get(
            f"/api/collections/{created['collection_id']}/revisions/{payload['revision_id']}"
        )
        self.assertEqual(read.status_code, 200, read.text)
        self.assertEqual(read.json()["members"][0]["document_id"], document_id)
        self.assertEqual(read.json()["index_status"], "queued")

    def test_confirmation_rejects_stale_draft_and_failed_item(self):
        created, state = self.imported(partial=True)
        self.assertEqual(self.confirm(created, state, ["f" * 32]).status_code, 400)
        changed = self.client.patch(
            f"/api/collections/{created['collection_id']}/draft",
            json={"expected_selection_revision": 1,
                  "selections": [{"search_id": state["selections"][0]["search_id"],
                                  "versioned_id": "2401.00001v1"}]},
        )
        self.assertEqual(changed.status_code, 200)
        success = state["import_report"]["items"][0]["document_id"]
        self.assertEqual(self.confirm(created, state, [success]).status_code, 409)

    def test_unknown_revision_resources_return_not_found(self):
        missing_collection = self.client.post(
            "/api/collections/missing/revisions",
            headers={"Idempotency-Key": "missing-confirm"},
            json={"expected_selection_revision": 1, "import_run_id": "missing",
                  "document_ids": ["missing"]},
        )
        self.assertEqual(missing_collection.status_code, 404)
        missing_answer_scope = self.client.post(
            "/api/answers", headers={"Idempotency-Key": "missing-answer"},
            json={"revision_id": "missing", "index_artifact_id": "missing",
                  "question": "What is studied?"},
        )
        self.assertEqual(missing_answer_scope.status_code, 404)

    def test_search_retry_http_requires_current_failed_attempt(self):
        submitted = self.client.post(
            "/api/searches", headers={"Idempotency-Key": "failed-search"},
            json={"question": "如何评估记忆？", "query_text": "LLM agent memory"},
        ).json()

        class FailingArxiv:
            def fetch(self, _plan):
                raise RuntimeError("offline source failure")

        with self.assertRaises(RuntimeError):
            run_one_job(
                self.engine, worker_id="search-fail", data_root=self.data_root,
                search_client=FailingArxiv(),
            )
        failed = self.client.get(f"/api/jobs/{submitted['job_id']}").json()
        retry = self.client.post(
            f"/api/jobs/{submitted['job_id']}/retry",
            json={"expected_attempt_id": failed["active_attempt_id"]},
        )
        self.assertEqual(retry.status_code, 202, retry.text)
        self.assertEqual(self.client.post(
            f"/api/jobs/{submitted['job_id']}/retry",
            json={"expected_attempt_id": failed["active_attempt_id"]},
        ).status_code, 409)

    def test_import_retry_http_reuses_fixed_request(self):
        created, state = self.imported()
        submitted = self.client.post(
            f"/api/collections/{created['collection_id']}/imports",
            headers={"Idempotency-Key": "failed-import"},
            json={"expected_selection_revision": state["selection_revision"]},
        ).json()
        with patch("backend.app.jobs.import_worker.stage_import_job", side_effect=RuntimeError("offline stage failure")):
            with self.assertRaises(RuntimeError):
                run_one_job(self.engine, worker_id="import-fail", data_root=self.data_root)
        failed = self.client.get(f"/api/jobs/{submitted['job_id']}").json()
        retry = self.client.post(
            f"/api/jobs/{submitted['job_id']}/retry",
            json={"expected_attempt_id": failed["active_attempt_id"]},
        )
        self.assertEqual(retry.status_code, 202, retry.text)
        self.assertEqual(self.client.get(f"/api/jobs/{submitted['job_id']}").json()["status"], "queued")

    def test_confirmation_lost_response_replays_original_after_draft_changes(self):
        created, state = self.imported()
        document_id = state["import_report"]["items"][0]["document_id"]
        first = self.confirm(created, state, [document_id]).json()
        changed = self.client.patch(
            f"/api/collections/{created['collection_id']}/draft",
            json={"expected_selection_revision": 1,
                  "selections": [{"search_id": state["selections"][0]["search_id"],
                                  "versioned_id": "2401.00001v3"}]},
        )
        self.assertEqual(changed.status_code, 200)
        replay = self.confirm(created, state, [document_id])
        self.assertEqual(replay.status_code, 202, replay.text)
        self.assertEqual(replay.json()["job_id"], first["job_id"])
        self.assertEqual(replay.json()["revision_id"], first["revision_id"])

    def test_concurrent_same_confirmation_key_creates_one_index_job(self):
        created, state = self.imported()
        document_id = state["import_report"]["items"][0]["document_id"]
        gate = Barrier(2)

        def submit():
            gate.wait()
            with TestClient(create_app(self.database_path, self.data_root)) as client:
                return client.post(
                    f"/api/collections/{created['collection_id']}/revisions",
                    headers={"Idempotency-Key": "simultaneous-confirm"},
                    json={"expected_selection_revision": 1,
                          "import_run_id": state["import_report"]["import_run_id"],
                          "document_ids": [document_id]},
                )

        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(lambda _: submit(), range(2)))
        self.assertEqual([response.status_code for response in responses], [202, 202])
        self.assertEqual(responses[0].json()["job_id"], responses[1].json()["job_id"])
        with Session(self.engine) as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(Job).where(Job.kind == "index")), 1)

    def test_index_worker_publishes_ready_only_with_current_attempt(self):
        created, state = self.imported()
        document_id = state["import_report"]["items"][0]["document_id"]
        submitted = self.confirm(created, state, [document_id]).json()
        executed = run_one_job(
            self.engine, worker_id="index", data_root=self.data_root,
            index_settings=settings(), index_embedding_model=FakeEmbedding(),
        )
        self.assertEqual(executed.kind, "index")
        self.assertEqual(executed.status, "succeeded")
        with TestClient(create_app(self.database_path, self.data_root)) as reopened:
            read = reopened.get(
                f"/api/collections/{created['collection_id']}/revisions/{submitted['revision_id']}"
            ).json()
        self.assertEqual(read["index_status"], "succeeded")
        self.assertEqual(read["index_artifact_id"], executed.result_resource_id)

    def test_wrong_index_binding_is_not_shown_as_ready(self):
        revision_id, artifact_id = self.ready_revision()
        with Session(self.engine) as session:
            job = session.scalar(select(Job).where(Job.kind == "index", Job.resource_id == revision_id))
            attempt = session.get(JobAttempt, job.active_attempt_id)
            attempt.artifact_ref = "indexes/collections/wrong"
            session.commit()
        with Session(self.engine) as session:
            collection_id = session.get(CollectionRevision, revision_id).collection_id
        revision = self.client.get(f"/api/collections/{collection_id}/revisions/{revision_id}")
        self.assertEqual(revision.status_code, 409)
        answer = self.client.post(
            "/api/answers", headers={"Idempotency-Key": "tampered-index"},
            json={"revision_id": revision_id, "index_artifact_id": artifact_id,
                  "question": "What is studied?"},
        )
        self.assertEqual(answer.status_code, 409)

    def test_failed_index_needs_explicit_retry_and_old_attempt_cannot_publish(self):
        created, state = self.imported()
        document_id = state["import_report"]["items"][0]["document_id"]
        submitted = self.confirm(created, state, [document_id]).json()
        with self.assertRaisesRegex(RuntimeError, "offline embedding failure"):
            run_one_job(
                self.engine, worker_id="index-fail", data_root=self.data_root,
                index_settings=settings(), index_embedding_model=FakeEmbedding(fail=True),
            )
        failed = self.client.get(f"/api/jobs/{submitted['job_id']}").json()
        self.assertEqual(failed["status"], "failed")
        self.assertIsNone(self.client.get(
            f"/api/collections/{created['collection_id']}/revisions/{submitted['revision_id']}"
        ).json()["index_artifact_id"])
        retry = self.client.post(
            f"/api/jobs/{submitted['job_id']}/retry",
            json={"expected_attempt_id": failed["active_attempt_id"]},
        )
        self.assertEqual(retry.status_code, 202, retry.text)
        self.assertNotEqual(retry.json()["attempt_id"], failed["active_attempt_id"])
        good = run_one_job(
            self.engine, worker_id="index-ok", data_root=self.data_root,
            index_settings=settings(), index_embedding_model=FakeEmbedding(),
        )
        self.assertEqual(good.status, "succeeded")

    def test_interrupted_old_index_claim_cannot_publish_after_retry(self):
        created, state = self.imported()
        document_id = state["import_report"]["items"][0]["document_id"]
        submitted = self.confirm(created, state, [document_id]).json()
        old = claim_next_job(self.engine, worker_id="old", supported_kinds=("index",))
        self.assertTrue(interrupt_expired_attempt(
            self.engine, old.attempt_id, worker_stopped_confirmed=True,
            now=datetime.now(timezone.utc) + timedelta(minutes=1),
        ))
        retry = self.client.post(
            f"/api/jobs/{submitted['job_id']}/retry",
            json={"expected_attempt_id": old.attempt_id},
        )
        self.assertEqual(retry.status_code, 202, retry.text)
        with self.assertRaises(LostClaim):
            execute_claimed_index(
                self.engine, old, data_root=self.data_root,
                settings=settings(), embedding_model=FakeEmbedding(),
            )
        read = self.client.get(
            f"/api/collections/{created['collection_id']}/revisions/{submitted['revision_id']}"
        ).json()
        self.assertIsNone(read["index_artifact_id"])

    def ready_revision(self):
        created, state = self.imported()
        document_id = state["import_report"]["items"][0]["document_id"]
        submitted = self.confirm(created, state, [document_id]).json()
        run_one_job(
            self.engine, worker_id="index", data_root=self.data_root,
            index_settings=settings(), index_embedding_model=FakeEmbedding(),
        )
        read = self.client.get(
            f"/api/collections/{created['collection_id']}/revisions/{submitted['revision_id']}"
        ).json()
        return submitted["revision_id"], read["index_artifact_id"]

    def test_answer_is_queued_saved_and_reopened_without_model(self):
        revision_id, artifact_id = self.ready_revision()
        body = {"revision_id": revision_id, "index_artifact_id": artifact_id,
                "question": "What is studied?"}
        submitted = self.client.post(
            "/api/answers", json=body, headers={"Idempotency-Key": "answer-1"},
        )
        self.assertEqual(submitted.status_code, 202, submitted.text)
        ids = submitted.json()
        self.assertEqual(self.client.post(
            "/api/answers", json=body, headers={"Idempotency-Key": "answer-1"},
        ).json(), ids)
        waiting = self.client.get(f"/api/answers/{ids['answer_run_id']}").json()
        self.assertEqual(waiting["status"], "queued")
        self.assertIsNone(waiting["answer"])
        done = run_one_job(
            self.engine, worker_id="answer", data_root=self.data_root,
            answer_settings=settings(),
            answer_executor=lambda scope, _question: offline_answer(scope, raw_citations=True),
        )
        self.assertEqual(done.status, "succeeded")
        with TestClient(create_app(self.database_path, self.data_root)) as reopened:
            response = reopened.get(f"/api/answers/{ids['answer_run_id']}")
            self.assertEqual(response.status_code, 200, response.text)
            answer = response.json()["answer"]
            self.assertEqual(answer["answer_outcome"], "reported_sufficient")
            with Session(self.engine) as session:
                source_revision = session.get(CollectionRevision, revision_id)
                self.assertEqual(answer["research_context"], {
                    "collection_id": source_revision.collection_id,
                    "revision_id": revision_id,
                })
            # Titles are read-time metadata, never a rewrite of the frozen answer.
            saved_file = self.data_root / "answers" / f"{ids['answer_run_id']}.json"
            frozen = saved_file.read_bytes()
            with Session(self.engine) as session:
                from backend.app.models import DocumentVersion
                document = session.get(DocumentVersion, answer["allowed_documents"][0]["document_id"])
                self.assertEqual(answer["document_titles"][document.id], document.title)
            reopened.get(f"/api/answers/{ids['answer_run_id']}")
            self.assertEqual(saved_file.read_bytes(), frozen)
            evidence_id = answer["evidence"][0]["evidence_id"]
            evidence = reopened.get(
                f"/api/answers/{ids['answer_run_id']}/evidence/{evidence_id}"
            )
            self.assertEqual(evidence.status_code, 200, evidence.text)
            self.assertEqual(evidence.json()["document_id"], answer["allowed_documents"][0]["document_id"])

    def test_saved_evidence_opens_only_its_verified_fixed_pdf(self):
        revision_id, artifact_id = self.ready_revision()
        submitted = self.client.post(
            "/api/answers", headers={"Idempotency-Key": "answer-pdf"},
            json={"revision_id": revision_id, "index_artifact_id": artifact_id,
                  "question": "What is studied?"},
        ).json()
        run_one_job(
            self.engine, worker_id="answer-pdf", data_root=self.data_root,
            answer_settings=settings(),
            answer_executor=lambda scope, _question: offline_answer(scope, raw_citations=True),
        )
        answer_id = submitted["answer_run_id"]
        answer = self.client.get(f"/api/answers/{answer_id}").json()["answer"]
        evidence = answer["evidence"][0]
        url = f"/api/answers/{answer_id}/evidence/{evidence['evidence_id']}/pdf"

        response = self.client.get(url)
        self.assertEqual(response.status_code, 200, response.text[:200] if response.status_code != 200 else "")
        self.assertEqual(response.headers["content-type"], "application/pdf")
        self.assertIn("inline", response.headers["content-disposition"])
        self.assertEqual(hashlib.sha256(response.content).hexdigest(), answer["allowed_documents"][0]["sha256"])
        downloaded = self.client.get(f"{url}?download=1")
        self.assertEqual(downloaded.status_code, 200)
        self.assertIn("attachment", downloaded.headers["content-disposition"])
        self.assertEqual(downloaded.content, response.content)
        location_url = f"/api/answers/{answer_id}/evidence/{evidence['evidence_id']}/location"
        location = self.client.get(location_url)
        self.assertEqual(location.status_code, 200, location.text)
        self.assertEqual(location.json(), {
            "evidence_id": evidence["evidence_id"],
            "pdf_sha256": answer["allowed_documents"][0]["sha256"],
            "status": "unknown", "matches": [],
        })
        self.assertEqual(self.client.get(
            f"/api/answers/{answer_id}/evidence/other:context/pdf"
        ).status_code, 404)

        from backend.app.models import DocumentVersion
        with Session(self.engine) as session:
            document = session.get(DocumentVersion, evidence["document_id"])
            original_sha = document.sha256
            blob_path = self.data_root / document.blob_ref
            document.sha256 = "0" * 64
            session.commit()
        self.assertEqual(self.client.get(url).status_code, 409)
        self.assertEqual(self.client.get(location_url).status_code, 409)
        with Session(self.engine) as session:
            session.get(DocumentVersion, evidence["document_id"]).sha256 = original_sha
            session.commit()
        blob_path.write_bytes(b"not the saved PDF")
        self.assertEqual(self.client.get(url).status_code, 409)

    def test_saved_context_coverage_keeps_answer_revision_after_draft_edit(self):
        revision_id, artifact_id = self.ready_revision()

        async def answer_with_observed_context(scope, _question):
            execution = await offline_answer(scope, raw_citations=True)
            context_id = execution.contexts[0].upstream_context_id
            # AnswerResponse clears Context.text.text; rebuild the captured input.
            originals = {context.upstream_context_id: context.original_text for context in execution.contexts}
            contexts = [context.model_copy(update={
                "text": context.text.model_copy(update={"text": originals[context.id] or ""}),
            }) for context in execution.response.session.contexts]
            final_context = await scope.settings.context_serializer(
                contexts, _question, None,
            )
            execution.response.session.raw_answer = f"The paper studies agent memory ({context_id})."
            execution.response.session.context = final_context
            snapshots = tuple(
                replace(snapshot, final_context=final_context)
                if snapshot.final_context is not None else snapshot
                for snapshot in execution.snapshots
            )
            return replace(execution, final_context=final_context, snapshots=snapshots)

        submitted = self.client.post(
            "/api/answers", headers={"Idempotency-Key": "answer-context-coverage"},
            json={"revision_id": revision_id, "index_artifact_id": artifact_id,
                  "question": "What is studied?"},
        ).json()
        run_one_job(
            self.engine, worker_id="answer-context-coverage", data_root=self.data_root,
            answer_settings=settings(), answer_executor=answer_with_observed_context,
        )
        answer_id = submitted["answer_run_id"]
        answer_file = self.data_root / "answers" / f"{answer_id}.json"
        original_hash = hashlib.sha256(answer_file.read_bytes()).hexdigest()
        with Session(self.engine) as session:
            collection_id = session.get(CollectionRevision, revision_id).collection_id
        draft = self.client.get(f"/api/collections/{collection_id}").json()
        updated = self.client.patch(
            f"/api/collections/{collection_id}/draft",
            json={"expected_selection_revision": draft["selection_revision"],
                  "selections": [{"search_id": draft["selections"][0]["search_id"],
                                  "versioned_id": "2401.00001v3"}]},
        )
        self.assertEqual(updated.status_code, 200, updated.text)

        with TestClient(create_app(self.database_path, self.data_root)) as reopened:
            response = reopened.get(f"/api/answers/{answer_id}")
        self.assertEqual(response.status_code, 200, response.text)
        answer = response.json()["answer"]
        self.assertIn("saved_context_coverage", answer)
        self.assertEqual(answer["research_context"], {
            "collection_id": collection_id, "revision_id": revision_id,
        })
        self.assertEqual(answer["saved_context_coverage"], {
            "status": "known", "basis": "saved_paperqa_session_context",
            "total_documents": 1,
            "included_documents": [{"document_id": answer["allowed_documents"][0]["document_id"],
                                    "versioned_id": "2401.00001v1"}],
            "missing_documents": [],
        })
        usage = answer["document_usage"]
        self.assertEqual(usage["total_documents"], 1)
        self.assertEqual(usage["context_fragments"], 1)
        self.assertEqual(usage["evidence_documents"], answer["saved_context_coverage"]["included_documents"])
        self.assertEqual(usage["cited_documents"], answer["saved_context_coverage"]["included_documents"])
        self.assertEqual(hashlib.sha256(answer_file.read_bytes()).hexdigest(), original_hash)

    def test_presave_failure_keeps_old_answer_id_after_retry(self):
        revision_id, artifact_id = self.ready_revision()
        submitted = self.client.post(
            "/api/answers", headers={"Idempotency-Key": "answer-fail"},
            json={"revision_id": revision_id, "index_artifact_id": artifact_id,
                  "question": "What is studied?"},
        ).json()

        async def fail(_scope, _question):
            raise RuntimeError("secret-provider-detail")

        with self.assertRaises(RuntimeError):
            run_one_job(
                self.engine, worker_id="answer-fail", data_root=self.data_root,
                answer_settings=settings(), answer_executor=fail,
            )
        old = self.client.get(f"/api/answers/{submitted['answer_run_id']}")
        self.assertEqual(old.status_code, 200, old.text)
        self.assertEqual(old.json()["status"], "failed")
        self.assertIsNone(old.json()["answer"])
        self.assertNotIn("secret-provider-detail", old.text)
        failed_job = self.client.get(f"/api/jobs/{submitted['job_id']}").json()
        retry = self.client.post(
            f"/api/jobs/{submitted['job_id']}/retry",
            json={"expected_attempt_id": failed_job["active_attempt_id"]},
        )
        self.assertEqual(retry.status_code, 202, retry.text)
        new_id = retry.json()["answer_run_id"]
        self.assertNotEqual(new_id, submitted["answer_run_id"])
        self.assertEqual(retry.json()["previous_answer_run_id"], submitted["answer_run_id"])
        self.assertEqual(self.client.get(f"/api/answers/{submitted['answer_run_id']}").json()["status"], "failed")
        self.assertEqual(self.client.get(f"/api/answers/{new_id}").json()["status"], "queued")
        run_one_job(
            self.engine, worker_id="answer-retry", data_root=self.data_root,
            answer_settings=settings(), answer_executor=lambda scope, _question: offline_answer(scope),
        )
        self.assertEqual(self.client.get(f"/api/answers/{new_id}").json()["status"], "succeeded")
        self.assertEqual(self.client.get(f"/api/answers/{submitted['answer_run_id']}").json()["status"], "failed")
        replay = self.client.post(
            "/api/answers", headers={"Idempotency-Key": "answer-fail"},
            json={"revision_id": revision_id, "index_artifact_id": artifact_id,
                  "question": "What is studied?"},
        )
        self.assertEqual(replay.status_code, 202)
        self.assertEqual(replay.json()["answer_run_id"], submitted["answer_run_id"])

    def test_concurrent_same_answer_key_creates_one_identity(self):
        revision_id, artifact_id = self.ready_revision()
        gate = Barrier(2)

        def submit():
            gate.wait()
            with TestClient(create_app(self.database_path, self.data_root)) as client:
                return client.post(
                    "/api/answers", headers={"Idempotency-Key": "simultaneous-answer"},
                    json={"revision_id": revision_id, "index_artifact_id": artifact_id,
                          "question": "What is studied?"},
                )

        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(lambda _: submit(), range(2)))
        self.assertEqual([response.status_code for response in responses], [202, 202])
        self.assertEqual(responses[0].json()["answer_run_id"], responses[1].json()["answer_run_id"])
        with Session(self.engine) as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(Job).where(Job.kind == "answer")), 1)
            self.assertEqual(session.scalar(select(func.count()).select_from(AnswerSubmission)), 1)

    def test_truncated_answer_is_saved_but_not_reported_sufficient(self):
        revision_id, artifact_id = self.ready_revision()
        submitted = self.client.post(
            "/api/answers", headers={"Idempotency-Key": "answer-truncated"},
            json={"revision_id": revision_id, "index_artifact_id": artifact_id,
                  "question": "What is studied?"},
        ).json()

        async def truncated(scope, _question):
            scope.settings.agent.max_timesteps = 2
            return await offline_answer(scope, raw_citations=True)

        done = run_one_job(
            self.engine, worker_id="answer-truncated", data_root=self.data_root,
            answer_settings=settings(), answer_executor=truncated,
        )
        self.assertEqual(done.status, "succeeded")
        answer = self.client.get(f"/api/answers/{submitted['answer_run_id']}").json()["answer"]
        self.assertEqual(answer["answer_outcome"], "truncated")
        self.assertEqual(answer["upstream_status"], "truncated")

    def test_failed_scoped_answer_keeps_reviewable_bundle_and_evidence_namespace(self):
        revision_id, artifact_id = self.ready_revision()
        submitted = self.client.post(
            "/api/answers", headers={"Idempotency-Key": "answer-scope-fail"},
            json={"revision_id": revision_id, "index_artifact_id": artifact_id,
                  "question": "What is studied?"},
        ).json()
        done = run_one_job(
            self.engine, worker_id="scope-fail", data_root=self.data_root,
            answer_settings=settings(),
            answer_executor=lambda scope, _question: offline_answer(scope, foreign_context=True),
        )
        self.assertEqual(done.status, "failed")
        result = self.client.get(f"/api/answers/{submitted['answer_run_id']}").json()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["answer"]["answer_outcome"], "scope_violation")
        evidence_id = result["answer"]["evidence"][0]["evidence_id"]
        self.assertEqual(self.client.get(
            f"/api/answers/{submitted['answer_run_id']}/evidence/{evidence_id}"
        ).status_code, 200)
        self.assertEqual(self.client.get(
            f"/api/answers/{submitted['answer_run_id']}/evidence/foreign:context"
        ).status_code, 404)

    def test_worker_rejects_answer_to_a_different_question(self):
        revision_id, artifact_id = self.ready_revision()
        submitted = self.client.post(
            "/api/answers", headers={"Idempotency-Key": "answer-mismatch"},
            json={"revision_id": revision_id, "index_artifact_id": artifact_id,
                  "question": "另一个问题？"},
        ).json()
        with self.assertRaisesRegex(ValueError, "question differs"):
            run_one_job(
                self.engine, worker_id="mismatch", data_root=self.data_root,
                answer_settings=settings(),
                answer_executor=lambda scope, _question: offline_answer(scope),
            )
        self.assertEqual(self.client.get(
            f"/api/answers/{submitted['answer_run_id']}"
        ).json()["status"], "failed")

    def test_pre_job_t05_answer_remains_read_only_through_api(self):
        import asyncio
        revision_id, artifact_id = self.ready_revision()
        configured = settings()
        scope = asyncio.run(prepare_answer_scope(
            self.engine, revision_id, artifact_id,
            data_root=self.data_root, settings=configured,
        ))
        execution = asyncio.run(offline_answer(scope, raw_citations=True))
        run_id = uuid4().hex
        bundle = build_scoped_answer_bundle(run_id, scope, execution)
        save_answer_run(self.engine, scope, execution, bundle, data_root=self.data_root)
        with TestClient(create_app(self.database_path, self.data_root)) as reopened:
            response = reopened.get(f"/api/answers/{run_id}")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIsNone(response.json()["job_id"])
            self.assertEqual(response.json()["answer"]["run_id"], run_id)
            self.assertEqual(response.json()["answer"]["research_context"]["revision_id"], revision_id)

    def test_late_answer_cannot_commit_after_interrupt_and_retry(self):
        revision_id, artifact_id = self.ready_revision()
        submitted = self.client.post(
            "/api/answers", headers={"Idempotency-Key": "answer-late"},
            json={"revision_id": revision_id, "index_artifact_id": artifact_id,
                  "question": "What is studied?"},
        ).json()
        entered, release = Event(), Event()

        async def slow(scope, _question):
            import asyncio
            entered.set()
            await asyncio.to_thread(release.wait)
            return await offline_answer(scope)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run_one_job, self.engine, worker_id="old-answer",
                                 data_root=self.data_root, answer_settings=settings(),
                                 answer_executor=slow)
            try:
                self.assertTrue(entered.wait(20))
                old = self.client.get(f"/api/jobs/{submitted['job_id']}").json()
                self.assertTrue(interrupt_expired_attempt(
                    self.engine, old["active_attempt_id"], worker_stopped_confirmed=True,
                    now=datetime.now(timezone.utc) + timedelta(minutes=1),
                ))
                retry = self.client.post(
                    f"/api/jobs/{submitted['job_id']}/retry",
                    json={"expected_attempt_id": old["active_attempt_id"]},
                )
                self.assertEqual(retry.status_code, 202, retry.text)
            finally:
                release.set()
            with self.assertRaises(LostClaim):
                future.result(timeout=30)
        old_read = self.client.get(f"/api/answers/{submitted['answer_run_id']}").json()
        self.assertEqual(old_read["status"], "interrupted")
        self.assertIsNone(old_read["answer"])
        with Session(self.engine) as session:
            self.assertIsNone(session.get(AnswerRun, submitted["answer_run_id"]))
            current = session.get(AnswerSubmission, retry.json()["answer_run_id"])
            self.assertIsNone(current.attempt_id)
        run_one_job(
            self.engine, worker_id="new-answer", data_root=self.data_root,
            answer_settings=settings(), answer_executor=lambda scope, _question: offline_answer(scope),
        )
        self.assertEqual(self.client.get(f"/api/answers/{retry.json()['answer_run_id']}").json()["status"], "succeeded")

    def test_late_index_publish_is_fenced_after_files_are_staged(self):
        import asyncio
        created, state = self.imported()
        document_id = state["import_report"]["items"][0]["document_id"]
        submitted = self.confirm(created, state, [document_id]).json()
        old = claim_next_job(self.engine, worker_id="old-index", supported_kinds=("index",))
        staged = asyncio.run(build_collection_index(
            self.engine, submitted["revision_id"], data_root=self.data_root,
            settings=settings(), embedding_model=FakeEmbedding(), defer_ready=True,
            active_claim=(old.job_id, old.attempt_id, old.worker_id),
        ))
        self.assertEqual(staged.status, "building")
        with self.assertRaisesRegex(ValueError, "ready index"):
            asyncio.run(prepare_answer_scope(
                self.engine, submitted["revision_id"], staged.id,
                data_root=self.data_root, settings=settings(),
            ))
        self.assertTrue(interrupt_expired_attempt(
            self.engine, old.attempt_id, worker_stopped_confirmed=True,
            now=datetime.now(timezone.utc) + timedelta(minutes=1),
        ))
        self.assertEqual(self.client.post(
            f"/api/jobs/{submitted['job_id']}/retry",
            json={"expected_attempt_id": old.attempt_id},
        ).status_code, 202)
        with self.assertRaises(LostClaim):
            finish_active_attempt(
                self.engine, job_id=old.job_id, attempt_id=old.attempt_id,
                worker_id=old.worker_id, status="succeeded", finished_at="2026-09-25T00:00:00Z",
                artifact_ref=f"indexes/collections/{staged.config_hash}",
                result_resource_id=staged.id,
            )
        with Session(self.engine) as session:
            self.assertEqual(session.get(IndexArtifact, staged.id).status, "building")
        new = run_one_job(
            self.engine, worker_id="new-index", data_root=self.data_root,
            index_settings=settings(), index_embedding_model=FakeEmbedding(),
        )
        self.assertEqual(new.status, "succeeded")
        scope = asyncio.run(prepare_answer_scope(
            self.engine, submitted["revision_id"], staged.id,
            data_root=self.data_root, settings=settings(),
        ))
        self.assertEqual(scope.index_artifact_id, staged.id)
