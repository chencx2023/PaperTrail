import tempfile
import unittest
from pathlib import Path


class SearchJobSubmissionTest(unittest.TestCase):
    def test_submit_returns_queued_ids_that_survive_database_reopen(self):
        from backend.app.db import open_database
        from backend.app.jobs.service import get_job, get_search, submit_search

        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "papertrail.sqlite"
            engine = open_database(database_path)
            submitted = submit_search(
                engine,
                question="如何评估 LLM Agent 的记忆能力？",
                query_text="LLM agent memory",
                idempotency_key="search-click-1",
            )
            self.assertTrue(submitted.search_id)
            self.assertTrue(submitted.job_id)
            self.assertNotEqual(submitted.search_id, submitted.job_id)
            engine.dispose()

            reopened = open_database(database_path)
            search = get_search(reopened, submitted.search_id)
            job = get_job(reopened, submitted.job_id)
            self.assertEqual(search.question, "如何评估 LLM Agent 的记忆能力？")
            self.assertEqual(search.query_text, "LLM agent memory")
            self.assertIsNone(search.snapshot_ref)
            self.assertEqual(job.kind, "search")
            self.assertEqual(job.resource_id, submitted.search_id)
            self.assertEqual(job.status, "queued")
            self.assertIsNone(job.active_attempt_id)
            reopened.dispose()

    def test_same_key_and_request_reuse_one_search_and_job(self):
        from sqlalchemy import func, select
        from sqlalchemy.orm import Session

        from backend.app.db import open_database
        from backend.app.jobs.service import submit_search
        from backend.app.models import Job, SearchRun

        with tempfile.TemporaryDirectory() as directory:
            engine = open_database(Path(directory) / "papertrail.sqlite")
            try:
                request = {
                    "question": "Agent 记忆如何评估？",
                    "query_text": "LLM agent memory",
                    "idempotency_key": "double-click",
                }
                first = submit_search(engine, **request)
                second = submit_search(engine, **request)

                self.assertEqual(second, first)
                with Session(engine) as session:
                    self.assertEqual(session.scalar(select(func.count()).select_from(Job)), 1)
                    self.assertEqual(
                        session.scalar(select(func.count()).select_from(SearchRun)), 1
                    )
            finally:
                engine.dispose()

    def test_same_key_with_different_request_conflicts_without_creating_work(self):
        from sqlalchemy import func, select
        from sqlalchemy.orm import Session

        from backend.app.db import open_database
        from backend.app.jobs.service import submit_search
        from backend.app.models import Job, SearchRun

        with tempfile.TemporaryDirectory() as directory:
            engine = open_database(Path(directory) / "papertrail.sqlite")
            try:
                first = submit_search(
                    engine,
                    question="Agent 记忆如何评估？",
                    query_text="LLM agent memory",
                    idempotency_key="one-key",
                )
                with self.assertRaises(ValueError):
                    submit_search(
                        engine,
                        question="Agent 规划如何评估？",
                        query_text="LLM agent planning",
                        idempotency_key="one-key",
                    )
                with Session(engine) as session:
                    self.assertEqual(session.scalar(select(func.count()).select_from(Job)), 1)
                    self.assertEqual(
                        session.scalar(select(func.count()).select_from(SearchRun)), 1
                    )
                    self.assertEqual(session.get(Job, first.job_id).resource_id, first.search_id)
            finally:
                engine.dispose()

    def test_database_rejects_duplicate_key_when_submissions_race(self):
        from sqlalchemy.exc import IntegrityError
        from sqlalchemy.orm import Session

        from backend.app.db import open_database
        from backend.app.jobs.service import submit_search
        from backend.app.models import Job

        with tempfile.TemporaryDirectory() as directory:
            engine = open_database(Path(directory) / "papertrail.sqlite")
            try:
                first = submit_search(
                    engine,
                    question="Agent 记忆如何评估？",
                    query_text="LLM agent memory",
                    idempotency_key="one-key",
                )
                with Session(engine) as session:
                    original = session.get(Job, first.job_id)
                    session.add(
                        Job(
                            id="other-job",
                            kind="search",
                            resource_id="other-search",
                            status="queued",
                            idempotency_key="one-key",
                            request_hash="different",
                            created_at=original.created_at,
                            updated_at=original.updated_at,
                        )
                    )
                    with self.assertRaises(IntegrityError):
                        session.commit()
            finally:
                engine.dispose()

    def test_blank_idempotency_key_does_not_enqueue(self):
        from sqlalchemy import func, select
        from sqlalchemy.orm import Session

        from backend.app.db import open_database
        from backend.app.jobs.service import submit_search
        from backend.app.models import Job

        with tempfile.TemporaryDirectory() as directory:
            engine = open_database(Path(directory) / "papertrail.sqlite")
            try:
                with self.assertRaises(ValueError):
                    submit_search(
                        engine,
                        question="Agent 记忆如何评估？",
                        query_text="LLM agent memory",
                        idempotency_key="  ",
                    )
                with Session(engine) as session:
                    self.assertEqual(session.scalar(select(func.count()).select_from(Job)), 0)
            finally:
                engine.dispose()

    def test_blank_question_does_not_enqueue(self):
        from sqlalchemy import func, select
        from sqlalchemy.orm import Session

        from backend.app.db import open_database
        from backend.app.jobs.service import submit_search
        from backend.app.models import SearchRun

        with tempfile.TemporaryDirectory() as directory:
            engine = open_database(Path(directory) / "papertrail.sqlite")
            try:
                with self.assertRaises(ValueError):
                    submit_search(
                        engine,
                        question=" \t ",
                        query_text="LLM agent memory",
                        idempotency_key="one-key",
                    )
                with Session(engine) as session:
                    self.assertEqual(
                        session.scalar(select(func.count()).select_from(SearchRun)), 0
                    )
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
