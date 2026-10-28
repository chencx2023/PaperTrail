"""Local HTTP entry point for saved PaperTrail resources and short Job submissions."""

from contextlib import contextmanager, asynccontextmanager
import asyncio
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Header, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, StrictInt
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import CreateTable
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.collections.submission import (
    CollectionUnavailable,
    CollectionSearchNotReady,
    CollectionSearchUnavailable,
    CollectionSnapshotInvalid,
    InvalidCollectionState,
    SearchChoice,
    create_collection_with_import,
    edit_collection_draft,
    get_collection_state,
)
from backend.app.collections.service import StaleSelection, get_collection
from backend.app.collections.report_read import InvalidCurrentImportReport
from backend.app.collections.revision_submission import (
    IndexRetryNotAllowed, InvalidIndexRequest, read_revision_state,
    retry_index_job, submit_revision_index,
)
from backend.app.db import open_database
from backend.app.runtime import read_runtime
from backend.app.history import read_history
from backend.app.models import ResearchRecord, ResearchOperation
from backend.app.research.records import ensure_search_record, backfill_research_records, read_research, list_researches, resolve_research, rename_research
from backend.app.research.workflow import ResearchConflict, prepare_research, answer_research, retry_research
from backend.app.search.suggestions import SuggestionError, generate_suggestion, generate_expansion, read_suggestion, save_adoption
from backend.app.search.query_builder import InvalidQueryPlan, plan_from_data
from backend.app.search.screening import ScreeningError, generate_screening, read_screening, screening_plan
from backend.app.search.failure import read_search_failure
from backend.app.evidence.pdf import (
    InvalidSavedPdf, locate_original_text_pages, read_saved_evidence_pdf,
)
from backend.app.ingestion.selection import InvalidDocumentSelection
from backend.app.jobs.import_submission import (
    ImportRetryNotAllowed, InvalidJobRequest, retry_import_job, submit_import_job,
)
from backend.app.jobs.service import (
    IdempotencyConflict,
    RetryNotAllowed,
    get_attempt,
    get_job,
    get_search,
    get_search_job,
    submit_search,
    retry_search,
)
from backend.app.qa.submission import (
    AnswerNotReady, AnswerResourceMissing, AnswerRetryNotAllowed, InvalidAnswerRequest,
    read_answer_state, retry_answer, submit_answer,
)
from backend.app.search.service import (
    InvalidSearchQuery,
    InvalidSearchSnapshot,
    replay_search_run,
)


class ApiError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int,
        resource_id: str | None = None,
        retryable: bool = False,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        self.resource_id = resource_id
        self.retryable = retryable


class SearchCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    query_text: str
    query_suggestion_id: str | None = None
    query_plan: dict | None = None


class QuerySuggestionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str


class QueryExpansionCreate(QuerySuggestionCreate):
    query_text: str


class ScreeningCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CollectionChoiceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_id: str
    versioned_id: str


class CollectionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selections: list[CollectionChoiceCreate]


class CollectionDraftUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_selection_revision: StrictInt
    selections: list[CollectionChoiceCreate]


class CollectionImportCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_selection_revision: StrictInt


class CollectionRevisionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_selection_revision: StrictInt
    import_run_id: str
    document_ids: list[str]


class RetryCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_attempt_id: str


class AnswerCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision_id: str
    index_artifact_id: str
    question: str


class ResearchPrepareCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    versioned_ids: list[str]
    expected_pool_revision_id: str | None


class ResearchAnswerCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    document_ids: list[str]
    expected_pool_revision_id: str


class ResearchRename(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str


class ResearchRetry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: str


def create_app(database_path: Path, data_root: Path, *, public_access=None) -> FastAPI:
    """Open only an initialized database; constructing the app is side-effect free."""

    db_path = Path(database_path).resolve()
    @asynccontextmanager
    async def lifespan(_app):
        if db_path.is_file():
            engine = open_database(db_path, create_schema=False)
            try:
                with engine.begin() as connection:
                    connection.execute(text("SELECT 1 FROM jobs LIMIT 1"))
                    for model in (ResearchRecord, ResearchOperation):
                        connection.execute(CreateTable(model.__table__, if_not_exists=True))
                backfill_research_records(engine)
            except (SQLAlchemyError, OSError):
                # Keep existing health/error endpoints available for a broken DB.
                pass
            finally:
                engine.dispose()
        yield
    app = FastAPI(lifespan=lifespan)
    app.state.data_root = Path(data_root).resolve()
    suggestion_lock = asyncio.Lock()
    screening_lock = asyncio.Lock()
    if public_access is not None:
        public_access.database_path = db_path
        app.middleware("http")(public_access.dispatch)

    @app.exception_handler(ApiError)
    async def api_error_handler(_request, error: ApiError):
        return JSONResponse(
            status_code=error.status_code,
            content={
                "code": error.code,
                "message": error.message,
                "resource_id": error.resource_id,
                "retryable": error.retryable,
            },
        )

    @app.exception_handler(SQLAlchemyError)
    async def database_error_handler(_request, _error: SQLAlchemyError):
        return JSONResponse(
            status_code=503,
            content={
                "code": "DATABASE_UNAVAILABLE",
                "message": "Database is unavailable.",
                "resource_id": None,
                "retryable": True,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request, _error: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "code": "INVALID_REQUEST",
                "message": "Request body is invalid.",
                "resource_id": None,
                "retryable": False,
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(_request, error: StarletteHTTPException):
        return JSONResponse(
            status_code=error.status_code,
            content={
                "code": "NOT_FOUND" if error.status_code == 404 else "HTTP_ERROR",
                "message": "Resource not found." if error.status_code == 404 else "Request failed.",
                "resource_id": None,
                "retryable": False,
            },
        )

    @contextmanager
    def database():
        if not db_path.is_file():
            raise ApiError(
                "DATABASE_UNAVAILABLE", "Database is unavailable.", 503, retryable=True
            )
        engine = open_database(db_path, create_schema=False)
        try:
            yield engine
        finally:
            engine.dispose()

    @app.get("/api/health")
    def health():
        try:
            with database() as engine, engine.connect() as connection:
                connection.execute(text("SELECT 1 FROM jobs LIMIT 1"))
        except (SQLAlchemyError, OSError) as error:
            raise ApiError(
                "DATABASE_UNAVAILABLE", "Database is unavailable.", 503, retryable=True
            ) from error
        return {"status": "ok"}

    @app.get("/api/runtime")
    def runtime():
        with database() as engine:
            return {**read_runtime(engine, db_path), "shared_workspace": public_access is not None}

    @app.get("/api/history")
    def history(kind: str = "all", limit: int = Query(default=25, ge=1, le=100), cursor: str | None = None):
        try:
            with database() as engine:
                return read_history(engine, kind=kind, limit=limit, cursor=cursor)
        except ValueError as error:
            raise ApiError("INVALID_HISTORY_QUERY", str(error), 400) from error

    @app.get("/api/researches")
    def researches(limit: int = Query(default=12, ge=1, le=100), cursor: str | None = None):
        with database() as engine:
            try: return list_researches(engine, limit=limit, cursor=cursor)
            except ValueError as error: raise ApiError("INVALID_HISTORY_QUERY", str(error), 400) from error

    @app.get("/api/researches/resolve")
    def research_resolve(search: str | None = None, collection: str | None = None,
                         revision: str | None = None, answer: str | None = None):
        with database() as engine:
            return {"research_id": resolve_research(engine, search_id=search, collection_id=collection,
                revision_id=revision, answer_id=answer)}

    @app.get("/api/researches/{research_id}")
    def research_read(research_id: str):
        with database() as engine:
            try: record = read_research(engine, research_id)
            except (ValueError, OSError) as error: raise ApiError("INVALID_RESEARCH_RESULT", "保存的研究范围无法核对。", 409) from error
            if record is None: raise ApiError("NOT_FOUND", "研究记录不存在。", 404)
            return record

    @app.patch("/api/researches/{research_id}")
    def research_rename(research_id: str, body: ResearchRename):
        with database() as engine:
            try:
                rename_research(engine, research_id, body.title)
                return read_research(engine, research_id)
            except LookupError as error: raise ApiError("NOT_FOUND", str(error), 404) from error
            except ValueError as error: raise ApiError("INVALID_NAME", str(error), 400) from error

    def research_submit(id, kind, body, key):
        if not key or not key.strip(): raise ApiError("INVALID_IDEMPOTENCY_KEY", "需要提交标识。", 400)
        try:
            with database() as engine:
                handler = {"prepare": prepare_research, "answer": answer_research, "retry": retry_research}[kind]
                return handler(engine, id, **body.model_dump(), idempotency_key=key, data_root=app.state.data_root)
        except IdempotencyConflict as error: raise ApiError("IDEMPOTENCY_CONFLICT", str(error), 409) from error
        except ResearchConflict as error: raise ApiError("RESEARCH_CONFLICT", str(error), 409) from error
        except LookupError as error: raise ApiError("NOT_FOUND", str(error), 404) from error
        except ValueError as error: raise ApiError("INVALID_RESEARCH_REQUEST", str(error), 400) from error

    @app.post("/api/researches/{research_id}/prepare", status_code=202)
    def research_prepare(research_id: str, body: ResearchPrepareCreate, idempotency_key: str | None = Header(default=None)):
        return research_submit(research_id, "prepare", body, idempotency_key)

    @app.post("/api/researches/{research_id}/answers", status_code=202)
    def research_answer(research_id: str, body: ResearchAnswerCreate, idempotency_key: str | None = Header(default=None)):
        return research_submit(research_id, "answer", body, idempotency_key)

    @app.post("/api/researches/{research_id}/retry", status_code=202)
    def research_retry(research_id: str, body: ResearchRetry, idempotency_key: str | None = Header(default=None)):
        return research_submit(research_id, "retry", body, idempotency_key)

    @app.post("/api/query-suggestions")
    async def post_query_suggestion(body: QuerySuggestionCreate):
        if suggestion_lock.locked():
            raise ApiError("QUERY_ASSISTANT_BUSY", "另一条检索词建议正在生成，请稍后再试。", 429)
        try:
            async with suggestion_lock:
                return await generate_suggestion(body.question, app.state.data_root)
        except SuggestionError as error:
            raise ApiError("QUERY_SUGGESTION_FAILED", str(error), 400) from error

    @app.post("/api/searches", status_code=202)
    def post_search(
        body: SearchCreate, idempotency_key: str | None = Header(default=None)
    ):
        if idempotency_key is None or not idempotency_key.strip():
            raise ApiError(
                "INVALID_IDEMPOTENCY_KEY", "Idempotency-Key is required.", 400
            )
        try:
            suggestion = read_suggestion(app.state.data_root, body.query_suggestion_id) if body.query_suggestion_id else None
            if suggestion and suggestion["question"] != body.question:
                raise SuggestionError("研究问题已经修改，请重新生成建议或手动填写检索词。")
            multi = plan_from_data(body.query_plan) if body.query_plan is not None else None
            if suggestion and ("query_plan" in suggestion) != (multi is not None):
                raise SuggestionError("建议类型与搜索不一致，请重新采用或取消扩展。")
            if multi and suggestion and suggestion["query_text"] != multi.baseline_query_text:
                raise SuggestionError("原检索词已修改，请为当前检索词重新扩展。")
            source = ({key: suggestion[key] for key in ("suggestion_id", "model", "prompt_version", "prompt_sha256", "usage", "elapsed_ms", "cost_usd")}
                      if multi and suggestion else None)
            with database() as engine:
                submitted = submit_search(
                    engine,
                    question=body.question,
                    query_text=body.query_text,
                    idempotency_key=idempotency_key,
                    query_plan=multi.as_data() if multi else None, plan_source=source,
                )
                job = get_job(engine, submitted.job_id)
                research_id = ensure_search_record(engine, submitted.search_id)
            if suggestion:
                save_adoption(app.state.data_root, submitted.search_id, suggestion, body.query_text,
                              multi.as_data() if multi else None)
        except IdempotencyConflict as error:
            raise ApiError(
                "IDEMPOTENCY_CONFLICT", "Key belongs to another request.", 409
            ) from error
        except InvalidSearchQuery as error:
            raise ApiError("INVALID_SEARCH_QUERY", str(error), 400) from error
        except ValueError as error:
            raise ApiError("INVALID_REQUEST", str(error), 400) from error
        return {
            "search_id": submitted.search_id,
            "research_id": research_id,
            "job_id": submitted.job_id,
            "status": job.status,
        }

    @app.post("/api/query-expansions")
    async def post_query_expansion(body: QueryExpansionCreate):
        if suggestion_lock.locked():
            raise ApiError("QUERY_ASSISTANT_BUSY", "另一条检索建议正在生成，请稍后再试。", 429)
        try:
            async with suggestion_lock:
                return await generate_expansion(body.question, body.query_text, app.state.data_root)
        except SuggestionError as error:
            raise ApiError("QUERY_EXPANSION_FAILED", str(error), 400) from error

    @app.get("/api/jobs/{job_id}")
    def read_job(job_id: str):
        with database() as engine:
            job = get_job(engine, job_id)
            if job is None:
                raise ApiError("NOT_FOUND", "Job not found.", 404, job_id)
            attempt = (
                get_attempt(engine, job.active_attempt_id)
                if job.active_attempt_id is not None
                else None
            )
            if attempt is not None and attempt.job_id != job.id:
                raise ApiError(
                    "INVALID_JOB_STATE", "Job attempt is unavailable.", 409, job_id
                )
        attempt_data = None
        if attempt is not None:
            attempt_data = {
                "attempt_id": attempt.id,
                "attempt_no": attempt.attempt_no,
                "worker_id": attempt.worker_id,
                "status": attempt.status,
                "started_at": attempt.started_at,
                "heartbeat_at": attempt.heartbeat_at,
                "lease_until": attempt.lease_until,
                "finished_at": attempt.finished_at,
                "error_kind": attempt.error_kind,
                "error_message": (
                    "Task failed. Check local run records."
                    if attempt.error_kind is not None
                    else None
                ),
            }
        return {
            "job_id": job.id,
            "kind": job.kind,
            "resource_id": job.resource_id,
            "status": job.status,
            "active_attempt_id": job.active_attempt_id,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "attempt": attempt_data,
        }

    @app.post("/api/jobs/{job_id}/retry", status_code=202)
    def post_job_retry(job_id: str, body: RetryCreate):
        with database() as engine:
            job = get_job(engine, job_id)
            if job is None:
                raise ApiError("NOT_FOUND", "Job not found.", 404, job_id)
            try:
                if job.kind == "search":
                    attempt_id = retry_search(
                        engine, job_id, expected_attempt_id=body.expected_attempt_id
                    ).attempt_id
                elif job.kind == "import":
                    attempt_id = retry_import_job(
                        engine, job_id, expected_attempt_id=body.expected_attempt_id
                    ).attempt_id
                elif job.kind == "index":
                    attempt_id = retry_index_job(
                        engine, job_id, expected_attempt_id=body.expected_attempt_id
                    )
                elif job.kind == "answer":
                    retried = retry_answer(
                        engine, job_id, expected_attempt_id=body.expected_attempt_id
                    )
                    backfill_research_records(engine)
                    return {
                        "job_id": retried.job_id,
                        "attempt_id": retried.attempt_id,
                        "answer_run_id": retried.answer_run_id,
                        "previous_answer_run_id": retried.previous_answer_run_id,
                        "status": "queued",
                    }
                else:
                    raise ApiError("UNSUPPORTED_JOB_KIND", "Job kind cannot be retried.", 409, job_id)
            except (RetryNotAllowed, ImportRetryNotAllowed, IndexRetryNotAllowed, AnswerRetryNotAllowed) as error:
                raise ApiError("RETRY_NOT_ALLOWED", str(error), 409, job_id) from error
            except (InvalidJobRequest, InvalidIndexRequest, InvalidAnswerRequest) as error:
                raise ApiError("INVALID_JOB_REQUEST", str(error), 409, job_id) from error
        return {"job_id": job_id, "attempt_id": attempt_id, "status": "queued"}

    @app.post("/api/collections", status_code=202)
    def post_collection(
        body: CollectionCreate, idempotency_key: str | None = Header(default=None)
    ):
        if idempotency_key is None or not idempotency_key.strip():
            raise ApiError("INVALID_IDEMPOTENCY_KEY", "Idempotency-Key is required.", 400)
        choices = tuple(
            SearchChoice(item.search_id, item.versioned_id) for item in body.selections
        )
        try:
            with database() as engine:
                submitted = create_collection_with_import(
                    engine,
                    data_root=app.state.data_root,
                    choices=choices,
                    idempotency_key=idempotency_key,
                )
                job = get_job(engine, submitted.job_id)
                backfill_research_records(engine)
        except IdempotencyConflict as error:
            raise ApiError("IDEMPOTENCY_CONFLICT", "Key belongs to another request.", 409) from error
        except CollectionSearchUnavailable as error:
            raise ApiError("NOT_FOUND", "Search not found.", 404) from error
        except CollectionSearchNotReady as error:
            raise ApiError("SEARCH_NOT_READY", str(error), 409) from error
        except CollectionSnapshotInvalid as error:
            raise ApiError("INVALID_SNAPSHOT", str(error), 409) from error
        except InvalidJobRequest as error:
            raise ApiError("INVALID_IMPORT_REQUEST", str(error), 409) from error
        except InvalidDocumentSelection as error:
            raise ApiError("INVALID_SELECTION", str(error), 400) from error
        except ValueError as error:
            raise ApiError("INVALID_REQUEST", str(error), 400) from error
        return {
            "collection_id": submitted.collection_id,
            "job_id": submitted.job_id,
            "selection_revision": 1,
            "status": job.status,
        }

    @app.get("/api/collections/{collection_id}")
    def read_collection(collection_id: str):
        try:
            with database() as engine:
                collection = get_collection_state(engine, collection_id)
        except InvalidCurrentImportReport as error:
            raise ApiError(
                "INVALID_IMPORT_REPORT", "Current import report is unavailable.", 409,
                collection_id,
            ) from error
        except InvalidCollectionState as error:
            raise ApiError(
                "INVALID_COLLECTION", "Saved collection is unavailable.", 409, collection_id
            ) from error
        except InvalidJobRequest as error:
            raise ApiError(
                "INVALID_IMPORT_REQUEST", "Saved import request is unavailable.", 409, collection_id
            ) from error
        if collection is None:
            raise ApiError("NOT_FOUND", "Collection not found.", 404, collection_id)
        return asdict(collection)

    @app.patch("/api/collections/{collection_id}/draft")
    def patch_collection_draft(collection_id: str, body: CollectionDraftUpdate):
        choices = tuple(
            SearchChoice(item.search_id, item.versioned_id) for item in body.selections
        )
        try:
            with database() as engine:
                revision = edit_collection_draft(
                    engine,
                    collection_id,
                    expected_revision=body.expected_selection_revision,
                    data_root=app.state.data_root,
                    choices=choices,
                )
        except CollectionUnavailable as error:
            raise ApiError("NOT_FOUND", "Collection not found.", 404, collection_id) from error
        except StaleSelection as error:
            raise ApiError("STALE_SELECTION", "Collection selection changed.", 409, collection_id) from error
        except CollectionSearchUnavailable as error:
            raise ApiError("NOT_FOUND", "Search not found.", 404) from error
        except CollectionSearchNotReady as error:
            raise ApiError("SEARCH_NOT_READY", str(error), 409) from error
        except CollectionSnapshotInvalid as error:
            raise ApiError("INVALID_SNAPSHOT", str(error), 409) from error
        except InvalidDocumentSelection as error:
            raise ApiError("INVALID_SELECTION", str(error), 400) from error
        except ValueError as error:
            raise ApiError("INVALID_REQUEST", str(error), 400) from error
        return {"collection_id": collection_id, "selection_revision": revision}

    @app.post("/api/collections/{collection_id}/imports", status_code=202)
    def post_collection_import(
        collection_id: str,
        body: CollectionImportCreate,
        idempotency_key: str | None = Header(default=None),
    ):
        if idempotency_key is None or not idempotency_key.strip():
            raise ApiError("INVALID_IDEMPOTENCY_KEY", "Idempotency-Key is required.", 400)
        try:
            with database() as engine:
                try:
                    submitted = submit_import_job(
                        engine,
                        collection_id,
                        expected_selection_revision=body.expected_selection_revision,
                        idempotency_key=idempotency_key,
                    )
                except StaleSelection as error:
                    if get_collection(engine, collection_id) is None:
                        raise ApiError(
                            "NOT_FOUND", "Collection not found.", 404, collection_id
                        ) from error
                    raise ApiError(
                        "STALE_SELECTION", "Collection selection changed.", 409,
                        collection_id,
                    ) from error
                job = get_job(engine, submitted.job_id)
        except IdempotencyConflict as error:
            raise ApiError("IDEMPOTENCY_CONFLICT", "Key belongs to another request.", 409) from error
        except InvalidJobRequest as error:
            raise ApiError(
                "INVALID_IMPORT_REQUEST", "Saved import request is unavailable.", 409,
                collection_id,
            ) from error
        except ValueError as error:
            raise ApiError("INVALID_REQUEST", str(error), 400) from error
        return {
            "collection_id": submitted.collection_id,
            "job_id": submitted.job_id,
            "selection_revision": body.expected_selection_revision,
            "status": job.status,
        }

    @app.post("/api/collections/{collection_id}/revisions", status_code=202)
    def post_collection_revision(
        collection_id: str, body: CollectionRevisionCreate,
        idempotency_key: str | None = Header(default=None),
    ):
        if idempotency_key is None or not idempotency_key.strip():
            raise ApiError("INVALID_IDEMPOTENCY_KEY", "Idempotency-Key is required.", 400)
        try:
            with database() as engine:
                result = submit_revision_index(
                    engine, collection_id,
                    expected_selection_revision=body.expected_selection_revision,
                    import_run_id=body.import_run_id,
                    document_ids=tuple(body.document_ids),
                    idempotency_key=idempotency_key, data_root=app.state.data_root,
                )
                job = get_job(engine, result.job_id)
                backfill_research_records(engine)
        except IdempotencyConflict as error:
            raise ApiError("IDEMPOTENCY_CONFLICT", str(error), 409) from error
        except StaleSelection as error:
            with database() as engine:
                if get_collection(engine, collection_id) is None:
                    raise ApiError("NOT_FOUND", "Collection not found.", 404, collection_id) from error
            raise ApiError("STALE_IMPORT_REPORT", str(error), 409, collection_id) from error
        except (InvalidCurrentImportReport, InvalidJobRequest) as error:
            raise ApiError("STALE_IMPORT_REPORT", str(error), 409, collection_id) from error
        except ValueError as error:
            raise ApiError("INVALID_CONFIRMATION", str(error), 400, collection_id) from error
        return {
            "collection_id": result.collection_id,
            "revision_id": result.revision_id,
            "job_id": result.job_id,
            "status": job.status,
        }

    @app.get("/api/collections/{collection_id}/revisions/{revision_id}")
    def read_collection_revision(collection_id: str, revision_id: str):
        try:
            with database() as engine:
                revision = read_revision_state(engine, collection_id, revision_id)
        except InvalidIndexRequest as error:
            raise ApiError("INVALID_INDEX_RESULT", str(error), 409, revision_id) from error
        if revision is None:
            raise ApiError("NOT_FOUND", "Revision not found.", 404, revision_id)
        return revision

    @app.post("/api/answers", status_code=202)
    def post_answer(body: AnswerCreate, idempotency_key: str | None = Header(default=None)):
        if idempotency_key is None or not idempotency_key.strip():
            raise ApiError("INVALID_IDEMPOTENCY_KEY", "Idempotency-Key is required.", 400)
        try:
            with database() as engine:
                submitted = submit_answer(
                    engine, revision_id=body.revision_id,
                    index_artifact_id=body.index_artifact_id,
                    question=body.question, idempotency_key=idempotency_key,
                )
                answer = read_answer_state(
                    engine, submitted.answer_run_id, data_root=app.state.data_root,
                )
                backfill_research_records(engine)
        except IdempotencyConflict as error:
            raise ApiError("IDEMPOTENCY_CONFLICT", str(error), 409) from error
        except AnswerResourceMissing as error:
            raise ApiError("NOT_FOUND", str(error), 404, body.revision_id) from error
        except AnswerNotReady as error:
            raise ApiError("INDEX_NOT_READY", str(error), 409, body.revision_id) from error
        except InvalidAnswerRequest as error:
            raise ApiError("INVALID_ANSWER_REQUEST", str(error), 409) from error
        except ValueError as error:
            raise ApiError("INVALID_QUESTION", str(error), 400) from error
        return {
            "answer_run_id": submitted.answer_run_id,
            "job_id": submitted.job_id,
            "status": answer["status"],
        }

    @app.get("/api/answers/{answer_run_id}")
    def read_answer(answer_run_id: str):
        try:
            with database() as engine:
                result = read_answer_state(
                    engine, answer_run_id, data_root=app.state.data_root,
                )
        except (InvalidAnswerRequest, ValueError, OSError) as error:
            raise ApiError("INVALID_ANSWER_RESULT", "Saved answer is unavailable.", 409, answer_run_id) from error
        if result is None:
            raise ApiError("NOT_FOUND", "Answer not found.", 404, answer_run_id)
        return result

    @app.get("/api/answers/{answer_run_id}/evidence/{evidence_id}")
    def read_answer_evidence(answer_run_id: str, evidence_id: str):
        try:
            with database() as engine:
                result = read_answer_state(
                    engine, answer_run_id, data_root=app.state.data_root,
                )
        except (InvalidAnswerRequest, ValueError, OSError) as error:
            raise ApiError("INVALID_ANSWER_RESULT", "Saved answer is unavailable.", 409, answer_run_id) from error
        if result is None:
            raise ApiError("NOT_FOUND", "Answer not found.", 404, answer_run_id)
        if result["answer"] is None:
            raise ApiError("NOT_FOUND", "Evidence not found.", 404, evidence_id)
        for item in result["answer"]["evidence"]:
            if item["evidence_id"] == evidence_id:
                return item
        raise ApiError("NOT_FOUND", "Evidence not found.", 404, evidence_id)

    @app.get("/api/answers/{answer_run_id}/evidence/{evidence_id}/pdf")
    def read_answer_evidence_pdf(answer_run_id: str, evidence_id: str, download: bool = False):
        try:
            with database() as engine:
                result = read_answer_state(
                    engine, answer_run_id, data_root=app.state.data_root,
                )
                if result is None:
                    raise ApiError("NOT_FOUND", "Answer not found.", 404, answer_run_id)
                if result["answer"] is None:
                    raise ApiError("NOT_FOUND", "Evidence not found.", 404, evidence_id)
                saved_pdf = read_saved_evidence_pdf(
                    engine, result["answer"], evidence_id, data_root=app.state.data_root,
                )
        except (InvalidAnswerRequest, InvalidSavedPdf, ValueError, OSError) as error:
            raise ApiError("INVALID_PDF", "Saved PDF is unavailable.", 409, evidence_id) from error
        if saved_pdf is None:
            raise ApiError("NOT_FOUND", "Evidence not found.", 404, evidence_id)
        body, digest = saved_pdf
        disposition = "attachment" if download else "inline"
        return Response(content=body, media_type="application/pdf", headers={
            "Content-Disposition": f'{disposition}; filename="{digest}.pdf"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        })

    @app.get("/api/answers/{answer_run_id}/evidence/{evidence_id}/location")
    def read_answer_evidence_location(answer_run_id: str, evidence_id: str):
        try:
            with database() as engine:
                result = read_answer_state(
                    engine, answer_run_id, data_root=app.state.data_root,
                )
                if result is None:
                    raise ApiError("NOT_FOUND", "Answer not found.", 404, answer_run_id)
                if result["answer"] is None:
                    raise ApiError("NOT_FOUND", "Evidence not found.", 404, evidence_id)
                saved_pdf = read_saved_evidence_pdf(
                    engine, result["answer"], evidence_id, data_root=app.state.data_root,
                )
        except (InvalidAnswerRequest, InvalidSavedPdf, ValueError, OSError) as error:
            raise ApiError("INVALID_PDF", "Saved PDF is unavailable.", 409, evidence_id) from error
        if saved_pdf is None:
            raise ApiError("NOT_FOUND", "Evidence not found.", 404, evidence_id)
        body, digest = saved_pdf
        evidence = next(item for item in result["answer"]["evidence"]
                        if item["evidence_id"] == evidence_id)
        return {"evidence_id": evidence_id, "pdf_sha256": digest,
                **locate_original_text_pages(body, evidence["original_text"])}

    @app.get("/api/searches/{search_id}")
    def read_search(search_id: str):
        with database() as engine:
            job = get_search_job(engine, search_id)
            try:
                search = get_search(engine, search_id)
            except InvalidQueryPlan as error:
                raise ApiError("INVALID_SNAPSHOT", "Saved search plan is unavailable.", 409, search_id) from error
            if search is None or job is None:
                raise ApiError("NOT_FOUND", "Search not found.", 404, search_id)
            attempt = (
                get_attempt(engine, job.active_attempt_id)
                if job.active_attempt_id is not None
                else None
            )

        result = {
            "search_id": search.id,
            "job_id": job.id,
            "status": job.status,
            "question": search.question,
            "query_text": search.query_text,
            "snapshot_sha256": None,
            "candidates": None,
            "query_plan": search.query_plan,
            "fusion": None,
            "provenance": None,
            "failure": None,
        }
        if job.status != "succeeded":
            expected_ref = f"runs/search-{attempt.id}/run.json" if attempt else None
            if (job.status == "failed" and attempt is not None
                    and attempt.job_id == job.id and attempt.status == "failed"
                    and search.snapshot_ref == expected_ref
                    and attempt.artifact_ref == expected_ref):
                snapshot_path = (app.state.data_root / expected_ref).resolve()
                if snapshot_path.is_relative_to(app.state.data_root):
                    result["failure"] = read_search_failure(snapshot_path.parent,
                        expected_run_id=f"search-{attempt.id}", query_text=search.query_text,
                        query_plan=search.query_plan)
            return result

        expected_ref = (
            f"runs/search-{attempt.id}/run.json" if attempt is not None else None
        )
        if (
            search.snapshot_ref is None
            or search.snapshot_ref != expected_ref
            or attempt.job_id != job.id
            or attempt.status != "succeeded"
            or attempt.artifact_ref != search.snapshot_ref
        ):
            raise ApiError(
                "INVALID_SNAPSHOT", "Saved search result is unavailable.", 409, search_id
            )
        snapshot_path = (app.state.data_root / search.snapshot_ref).resolve()
        if (
            snapshot_path.name != "run.json"
            or not snapshot_path.is_relative_to(app.state.data_root)
        ):
            raise ApiError(
                "INVALID_SNAPSHOT", "Saved search result is unavailable.", 409, search_id
            )
        try:
            replayed = replay_search_run(snapshot_path.parent)
        except InvalidSearchSnapshot as error:
            raise ApiError(
                "INVALID_SNAPSHOT", "Saved search result is unavailable.", 409, search_id
            ) from error
        if (
            not replayed.candidates_match
            or replayed.source_run_id != f"search-{attempt.id}"
            or (replayed.provenance["original_plan"] if replayed.provenance else
                replayed.fusion["plan"] if replayed.fusion else None) != search.query_plan
        ):
            raise ApiError(
                "INVALID_SNAPSHOT", "Saved search result is unavailable.", 409, search_id
            )
        result["snapshot_sha256"] = replayed.snapshot_sha256
        result["candidates"] = [asdict(item) for item in replayed.feed.candidates]
        result["fusion"] = replayed.fusion
        result["provenance"] = replayed.provenance
        return result

    @app.get("/api/searches/{search_id}/screening")
    def read_search_screening(search_id: str):
        search = read_search(search_id)
        if search["status"] != "succeeded" or not search["candidates"]:
            return {"screening": None}
        try:
            saved = read_screening(search, app.state.data_root)
            return {"screening": saved, "screening_plan": screening_plan(search, saved)}
        except ScreeningError as error:
            raise ApiError("SCREENING_UNAVAILABLE", str(error), error.status_code, search_id) from None

    @app.post("/api/searches/{search_id}/screening")
    async def post_search_screening(search_id: str, _body: ScreeningCreate):
        if screening_lock.locked():
            raise ApiError("SCREENING_BUSY", "另一批摘要正在筛选，请稍后再试。", 429, search_id)
        async with screening_lock:
            search = read_search(search_id)
            try:
                return await generate_screening(search, app.state.data_root)
            except ScreeningError as error:
                raise ApiError("SCREENING_FAILED", str(error), error.status_code, search_id) from None

    frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if (frontend_dist / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="frontend-assets")

        @app.get("/", include_in_schema=False)
        def frontend_index():
            return FileResponse(frontend_dist / "index.html")

    return app
