"""Database records for search work and fixed-version document identity."""

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SearchRun(Base):
    __tablename__ = "search_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_query_text: Mapped[str] = mapped_column(Text, nullable=False)
    search_query: Mapped[str] = mapped_column(Text, nullable=False)
    query_contract_version: Mapped[int] = mapped_column(nullable=False)
    snapshot_ref: Mapped[str | None] = mapped_column(Text)
    retrieved_at: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)


class Paper(Base):
    __tablename__ = "papers"
    __table_args__ = (UniqueConstraint("source", "source_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    source_id: Mapped[str] = mapped_column(String(80), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors_json: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[str] = mapped_column(String(32), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(32), nullable=False)


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("paper_id", "versioned_source_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False)
    versioned_source_id: Mapped[str] = mapped_column(String(80), nullable=False)
    source_run_id: Mapped[str] = mapped_column(String(32), nullable=False)
    snapshot_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors_json: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[str] = mapped_column(String(32), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(32), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    blob_ref: Mapped[str] = mapped_column(Text, nullable=False)
    byte_count: Mapped[int] = mapped_column(nullable=False)
    page_count: Mapped[int] = mapped_column(nullable=False)
    download_status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)


class DocumentPreparation(Base):
    __tablename__ = "document_preparations"
    __table_args__ = (UniqueConstraint("document_id", "parser_config_hash"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("document_versions.id"), nullable=False)
    parser_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    parser_config_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    chunk_count: Mapped[int | None] = mapped_column()
    error_kind: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    checked_at: Mapped[str] = mapped_column(String(32), nullable=False)


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    selection_revision: Mapped[int] = mapped_column(nullable=False)
    selected_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(32), nullable=False)


class CollectionImportRun(Base):
    __tablename__ = "collection_import_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id"), nullable=False)
    selection_revision: Mapped[int] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)
    finished_at: Mapped[str | None] = mapped_column(String(32))


class CollectionImportItem(Base):
    __tablename__ = "collection_import_items"
    __table_args__ = (UniqueConstraint("import_run_id", "position"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    import_run_id: Mapped[str] = mapped_column(ForeignKey("collection_import_runs.id"), nullable=False)
    position: Mapped[int] = mapped_column(nullable=False)
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)
    versioned_id: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("document_versions.id"))
    preparation_id: Mapped[str | None] = mapped_column(ForeignKey("document_preparations.id"))
    reused_pdf: Mapped[bool] = mapped_column(nullable=False)
    error_kind: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)


class CollectionRevision(Base):
    __tablename__ = "collection_revisions"
    __table_args__ = (
        UniqueConstraint("collection_id", "revision_no"),
        UniqueConstraint("collection_id", "import_run_id", "snapshot_hash"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id"), nullable=False)
    import_run_id: Mapped[str] = mapped_column(ForeignKey("collection_import_runs.id"), nullable=False)
    revision_no: Mapped[int] = mapped_column(nullable=False)
    selection_revision: Mapped[int] = mapped_column(nullable=False)
    snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    confirmed_at: Mapped[str] = mapped_column(String(32), nullable=False)


class CollectionRevisionDocument(Base):
    __tablename__ = "collection_revision_documents"
    __table_args__ = (
        UniqueConstraint("revision_id", "position"),
        UniqueConstraint("revision_id", "document_id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    revision_id: Mapped[str] = mapped_column(ForeignKey("collection_revisions.id"), nullable=False)
    position: Mapped[int] = mapped_column(nullable=False)
    document_id: Mapped[str] = mapped_column(ForeignKey("document_versions.id"), nullable=False)
    versioned_id: Mapped[str] = mapped_column(String(80), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    preparation_id: Mapped[str] = mapped_column(ForeignKey("document_preparations.id"), nullable=False)
    parser_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class IndexArtifact(Base):
    __tablename__ = "index_artifacts"
    __table_args__ = (UniqueConstraint("revision_id", "config_hash"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    revision_id: Mapped[str] = mapped_column(ForeignKey("collection_revisions.id"), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_ref: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error_kind: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(32), nullable=False)


class AnswerRun(Base):
    __tablename__ = "answer_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    job_attempt_id: Mapped[str | None] = mapped_column(ForeignKey("job_attempts.id"))
    collection_revision_id: Mapped[str] = mapped_column(
        ForeignKey("collection_revisions.id"), nullable=False
    )
    index_artifact_id: Mapped[str] = mapped_column(
        ForeignKey("index_artifacts.id"), nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    execution_status: Mapped[str] = mapped_column(String(16), nullable=False)
    answer_outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    upstream_status: Mapped[str] = mapped_column(String(32), nullable=False)
    termination_reason: Mapped[str] = mapped_column(String(32), nullable=False)
    artifact_ref: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[str] = mapped_column(String(32), nullable=False)
    finished_at: Mapped[str] = mapped_column(String(32), nullable=False)
    saved_at: Mapped[str] = mapped_column(String(32), nullable=False)


class AnswerSubmission(Base):
    __tablename__ = "answer_submissions"
    __table_args__ = (UniqueConstraint("attempt_id"),)

    answer_run_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), nullable=False)
    attempt_id: Mapped[str | None] = mapped_column(ForeignKey("job_attempts.id"))
    previous_answer_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("answer_submissions.answer_run_id")
    )
    submitted_at: Mapped[str] = mapped_column(String(32), nullable=False)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (UniqueConstraint("kind", "idempotency_key"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(Text, nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    active_attempt_id: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(32), nullable=False)


class JobRequest(Base):
    __tablename__ = "job_requests"

    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), primary_key=True)
    format_version: Mapped[int] = mapped_column(nullable=False)
    request_json: Mapped[str] = mapped_column(Text, nullable=False)


class JobAttempt(Base):
    __tablename__ = "job_attempts"
    __table_args__ = (UniqueConstraint("job_id", "attempt_no"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), nullable=False)
    attempt_no: Mapped[int] = mapped_column(nullable=False)
    worker_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    started_at: Mapped[str] = mapped_column(String(32), nullable=False)
    heartbeat_at: Mapped[str | None] = mapped_column(String(32))
    lease_until: Mapped[str | None] = mapped_column(String(32))
    finished_at: Mapped[str | None] = mapped_column(String(32))
    artifact_ref: Mapped[str | None] = mapped_column(Text)
    error_kind: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)


class JobAttemptResult(Base):
    __tablename__ = "job_attempt_results"

    attempt_id: Mapped[str] = mapped_column(
        ForeignKey("job_attempts.id"), primary_key=True
    )
    resource_id: Mapped[str] = mapped_column(String(32), nullable=False)
    artifact_ref: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)


class ResearchRecord(Base):
    """Display metadata and immutable links; original artifacts stay untouched."""
    __tablename__ = "research_records"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    origin_key: Mapped[str | None] = mapped_column(Text, unique=True)
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("research_records.id"))
    title: Mapped[str] = mapped_column(Text, nullable=False)
    title_is_custom: Mapped[bool] = mapped_column(nullable=False, default=False)
    search_id: Mapped[str | None] = mapped_column(ForeignKey("search_runs.id"), index=True)
    collection_id: Mapped[str | None] = mapped_column(ForeignKey("collections.id"))
    pool_revision_id: Mapped[str | None] = mapped_column(ForeignKey("collection_revisions.id"))
    scope_revision_id: Mapped[str | None] = mapped_column(ForeignKey("collection_revisions.id"))
    answer_run_id: Mapped[str | None] = mapped_column(String(32), index=True)
    question: Mapped[str | None] = mapped_column(Text)
    operation_id: Mapped[str | None] = mapped_column(String(32))
    prepare_operation_id: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(32), nullable=False)


class ResearchOperation(Base):
    """An explicitly submitted intent, resumed only by the local worker."""
    __tablename__ = "research_operations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    research_id: Mapped[str] = mapped_column(ForeignKey("research_records.id"), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    request_json: Mapped[str] = mapped_column(Text, nullable=False)
    report_json: Mapped[str | None] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    phase: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    collection_id: Mapped[str | None] = mapped_column(String(32))
    selection_revision: Mapped[int | None] = mapped_column()
    revision_id: Mapped[str | None] = mapped_column(String(32))
    job_id: Mapped[str | None] = mapped_column(String(32))
    error_kind: Mapped[str | None] = mapped_column(String(64))
    message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(32), nullable=False)
