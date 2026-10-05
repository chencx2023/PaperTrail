import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.api import create_app
from backend.app import api as api_module
from backend.app.db import open_database
from backend.app.jobs.worker import run_one_search
from backend.app.models import Job, SearchRun
from backend.app.search.client import ArxivClient, ArxivTransportError, TransportResponse


class FixedTransport:
    def __init__(self, fixture_name: str = "normal_with_versions.xml") -> None:
        self.fixture_name = fixture_name
        self.calls = 0

    def send(self, _request):
        self.calls += 1
        fixture = Path(__file__).parent / "fixtures" / "arxiv" / self.fixture_name
        return TransportResponse(
            status_code=200,
            body=fixture.read_bytes(),
            headers=(("Content-Type", "application/atom+xml"),),
        )


def fixed_client(transport: FixedTransport) -> ArxivClient:
    return ArxivClient(
        transport=transport,
        clock=lambda: datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc),
    )


class ApiSearchTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.database_path = self.root / "papertrail.sqlite"
        self.data_root = self.root / "data"
        self.engine = open_database(self.database_path)
        self.addCleanup(self.engine.dispose)
        self.client = TestClient(create_app(self.database_path, self.data_root))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.body = {
            "question": "如何评估 Agent 记忆？",
            "query_text": "LLM agent memory",
        }

    def _post(self, *, key: str = "search-submit-1", body: dict | None = None):
        return self.client.post(
            "/api/searches",
            json=self.body if body is None else body,
            headers={"Idempotency-Key": key},
        )

    def _counts(self) -> tuple[int, int]:
        with Session(self.engine) as session:
            return (
                session.scalar(select(func.count()).select_from(Job)),
                session.scalar(select(func.count()).select_from(SearchRun)),
            )

    def test_lost_response_replay_after_worker_completion_returns_same_ids(self):
        self._post()  # The caller discards the response after the server commits.
        self.assertEqual(self._counts(), (1, 1))
        with Session(self.engine) as session:
            original_job = session.scalar(select(Job))
            original_search = session.scalar(select(SearchRun))

        transport = FixedTransport()
        executed = run_one_search(
            self.engine,
            worker_id="test-worker",
            data_root=self.data_root,
            client=fixed_client(transport),
        )
        self.assertEqual(executed.status, "succeeded")

        recovered = self._post()
        self.assertEqual(recovered.status_code, 202)
        self.assertEqual(recovered.json()["search_id"], original_search.id)
        self.assertEqual(recovered.json()["job_id"], original_job.id)
        self.assertEqual(recovered.json()["status"], "succeeded")
        self.assertEqual(self._counts(), (1, 1))
        self.assertIsNone(
            run_one_search(
                self.engine,
                worker_id="test-worker",
                data_root=self.data_root,
                client=fixed_client(transport),
            )
        )
        self.assertEqual(transport.calls, 1)

    def test_concurrent_same_key_submissions_create_one_job(self):
        ready = Barrier(2)

        def submit_at_same_time():
            with TestClient(create_app(self.database_path, self.data_root)) as client:
                ready.wait(timeout=5)
                return client.post(
                    "/api/searches",
                    json=self.body,
                    headers={"Idempotency-Key": "race-1"},
                )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(submit_at_same_time) for _ in range(2)]
            responses = [future.result(timeout=15) for future in futures]

        self.assertEqual([response.status_code for response in responses], [202, 202])
        self.assertEqual(responses[0].json(), responses[1].json())
        self.assertEqual(self._counts(), (1, 1))

    def test_missing_or_blank_idempotency_key_does_not_create_work(self):
        for headers in ({}, {"Idempotency-Key": "  "}):
            with self.subTest(headers=headers):
                response = self.client.post("/api/searches", json=self.body, headers=headers)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["code"], "INVALID_IDEMPOTENCY_KEY")
        self.assertEqual(self._counts(), (0, 0))

    def test_same_key_with_changed_body_conflicts_without_new_job(self):
        first = self._post()
        changed = self._post(
            body={"question": "另一个问题", "query_text": "LLM agent memory"}
        )
        self.assertEqual(first.status_code, 202)
        self.assertEqual(changed.status_code, 409)
        self.assertEqual(changed.json()["code"], "IDEMPOTENCY_CONFLICT")
        self.assertEqual(self._counts(), (1, 1))

    def test_invalid_query_and_malformed_body_do_not_create_work(self):
        invalid_query = self._post(body={"question": "问题", "query_text": "中文词"})
        missing_field = self._post(body={"question": "问题"})
        self.assertEqual(invalid_query.status_code, 400)
        self.assertEqual(invalid_query.json()["code"], "INVALID_SEARCH_QUERY")
        self.assertEqual(missing_field.status_code, 422)
        self.assertEqual(missing_field.json()["code"], "INVALID_REQUEST")
        self.assertEqual(self._counts(), (0, 0))

    def test_unknown_request_field_is_rejected_before_submitting(self):
        response = self._post(body={**self.body, "source": "another-provider"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "INVALID_REQUEST")
        self.assertEqual(self._counts(), (0, 0))

    def test_job_read_exposes_attempt_state_but_not_raw_provider_error(self):
        submission = self._post()
        self.assertEqual(submission.status_code, 202)
        submitted = submission.json()
        job_id = submitted["job_id"]
        queued = self.client.get(f"/api/jobs/{job_id}")
        self.assertEqual(queued.status_code, 200)
        self.assertEqual(queued.json()["status"], "queued")
        self.assertIsNone(queued.json()["attempt"])

        class ExplodingClient:
            def fetch(self, _plan):
                raise RuntimeError("secret-provider-detail")

        with self.assertRaisesRegex(RuntimeError, "secret-provider-detail"):
            run_one_search(
                self.engine,
                worker_id="test-worker",
                data_root=self.data_root,
                client=ExplodingClient(),
            )
        failed = self.client.get(f"/api/jobs/{job_id}")
        self.assertEqual(failed.status_code, 200)
        self.assertEqual(failed.json()["status"], "failed")
        self.assertEqual(failed.json()["attempt"]["status"], "failed")
        self.assertEqual(failed.json()["attempt"]["error_kind"], "RuntimeError")
        self.assertNotIn("secret-provider-detail", failed.text)

        search = self.client.get(f"/api/searches/{submitted['search_id']}")
        self.assertEqual(search.status_code, 200)
        self.assertEqual(search.json()["status"], "failed")
        self.assertIsNone(search.json()["candidates"])

    def test_failed_search_explains_rate_limit_and_network_without_raw_errors(self):
        for number, expected in enumerate(("rate_limited", "network")):
            class FailingTransport:
                def send(self, _request):
                    if expected == "network":
                        raise ArxivTransportError("secret-network-detail")
                    return TransportResponse(429, b"secret-provider-body", ())
            submitted = self._post(key=f"failed-{number}").json()
            run_one_search(self.engine, worker_id="offline", data_root=self.data_root,
                client=ArxivClient(transport=FailingTransport(), max_attempts=1))
            for _ in range(2):
                read = self.client.get(f"/api/searches/{submitted['search_id']}")
                self.assertEqual(read.status_code, 200)
                self.assertEqual(read.json()["failure"]["kind"], expected)
                self.assertEqual(read.json()["failure"]["http_attempts"], 1)
                self.assertNotIn("secret-", read.text)

    def test_unknown_job_and_missing_database_use_standard_errors(self):
        unknown = self.client.get("/api/jobs/unknown")
        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(unknown.json().get("code"), "NOT_FOUND")

        missing_path = self.root / "missing.sqlite"
        with TestClient(create_app(missing_path, self.data_root)) as missing_client:
            response = missing_client.get("/api/jobs/unknown")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "DATABASE_UNAVAILABLE")
        self.assertFalse(missing_path.exists())

    def test_queued_search_is_distinct_from_reopened_verified_result(self):
        submission = self._post().json()
        search_id = submission["search_id"]
        queued = self.client.get(f"/api/searches/{search_id}")
        self.assertEqual(queued.status_code, 200)
        self.assertEqual(queued.json()["status"], "queued")
        self.assertIsNone(queued.json()["candidates"])
        self.assertIsNone(queued.json()["snapshot_sha256"])

        transport = FixedTransport()
        run_one_search(
            self.engine,
            worker_id="test-worker",
            data_root=self.data_root,
            client=fixed_client(transport),
        )
        self.engine.dispose()
        with TestClient(create_app(self.database_path, self.data_root)) as reopened:
            completed = reopened.get(f"/api/searches/{search_id}")
        self.assertEqual(completed.status_code, 200)
        self.assertEqual(completed.json()["status"], "succeeded")
        self.assertEqual(completed.json()["job_id"], submission["job_id"])
        self.assertEqual(len(completed.json()["snapshot_sha256"]), 64)
        self.assertGreater(len(completed.json()["candidates"]), 0)
        self.assertTrue(completed.json()["candidates"][0]["title"])
        self.assertEqual(transport.calls, 1)

    def test_successful_empty_feed_returns_empty_candidates(self):
        search_id = self._post().json()["search_id"]
        run_one_search(
            self.engine,
            worker_id="test-worker",
            data_root=self.data_root,
            client=fixed_client(FixedTransport("empty.xml")),
        )
        response = self.client.get(f"/api/searches/{search_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "succeeded")
        self.assertEqual(response.json()["candidates"], [])

    def test_worker_completion_between_status_reads_does_not_look_corrupt(self):
        search_id = self._post().json()["search_id"]
        original_get_search = api_module.get_search
        transport = FixedTransport()
        completed = False

        def get_search_then_complete(engine, requested_id):
            nonlocal completed
            search = original_get_search(engine, requested_id)
            if not completed:
                completed = True
                run_one_search(
                    self.engine,
                    worker_id="test-worker",
                    data_root=self.data_root,
                    client=fixed_client(transport),
                )
            return search

        with patch.object(api_module, "get_search", side_effect=get_search_then_complete):
            response = self.client.get(f"/api/searches/{search_id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "queued")
        self.assertIsNone(response.json()["candidates"])
        completed_read = self.client.get(f"/api/searches/{search_id}")
        self.assertEqual(completed_read.json()["status"], "succeeded")
        self.assertEqual(transport.calls, 1)

    def test_corrupted_or_escaped_snapshot_is_not_returned_as_success(self):
        search_id = self._post().json()["search_id"]
        run_one_search(
            self.engine,
            worker_id="test-worker",
            data_root=self.data_root,
            client=fixed_client(FixedTransport()),
        )
        with Session(self.engine) as session:
            search = session.get(SearchRun, search_id)
            snapshot_ref = search.snapshot_ref
        response_path = (self.data_root / snapshot_ref).parent / "response.atom"
        response_path.write_bytes(b"corrupted")

        corrupted = self.client.get(f"/api/searches/{search_id}")
        self.assertEqual(corrupted.status_code, 409)
        self.assertEqual(corrupted.json()["code"], "INVALID_SNAPSHOT")
        self.assertNotIn(str(self.root), corrupted.text)

        with Session(self.engine) as session:
            session.get(SearchRun, search_id).snapshot_ref = "../outside/run.json"
            session.commit()
        escaped = self.client.get(f"/api/searches/{search_id}")
        self.assertEqual(escaped.status_code, 409)
        self.assertEqual(escaped.json()["code"], "INVALID_SNAPSHOT")
        self.assertNotIn(str(self.root), escaped.text)

    def test_unknown_search_has_standard_not_found_error(self):
        response = self.client.get("/api/searches/unknown")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json().get("code"), "NOT_FOUND")

    def test_search_cannot_read_another_jobs_valid_snapshot(self):
        first_id = self._post(key="first").json()["search_id"]
        second_id = self._post(
            key="second",
            body={"question": "另一问题", "query_text": "LLM agent planning"},
        ).json()["search_id"]
        client = fixed_client(FixedTransport())
        run_one_search(
            self.engine, worker_id="test-worker", data_root=self.data_root, client=client
        )
        run_one_search(
            self.engine, worker_id="test-worker", data_root=self.data_root, client=client
        )
        with Session(self.engine) as session:
            first = session.get(SearchRun, first_id)
            second = session.get(SearchRun, second_id)
            first.snapshot_ref = second.snapshot_ref
            session.commit()

        response = self.client.get(f"/api/searches/{first_id}")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], "INVALID_SNAPSHOT")

    def test_job_read_rejects_attempt_belonging_to_another_job(self):
        first_job_id = self._post(key="first").json()["job_id"]
        self._post(
            key="second",
            body={"question": "另一问题", "query_text": "LLM agent planning"},
        )
        client = fixed_client(FixedTransport())
        run_one_search(
            self.engine, worker_id="test-worker", data_root=self.data_root, client=client
        )
        run_one_search(
            self.engine, worker_id="test-worker", data_root=self.data_root, client=client
        )
        with Session(self.engine) as session:
            jobs = session.scalars(select(Job).order_by(Job.created_at, Job.id)).all()
            first = session.get(Job, first_job_id)
            other = next(job for job in jobs if job.id != first_job_id)
            first.active_attempt_id = other.active_attempt_id
            session.commit()

        response = self.client.get(f"/api/jobs/{first_job_id}")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], "INVALID_JOB_STATE")


if __name__ == "__main__":
    unittest.main()
