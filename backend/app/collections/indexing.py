"""Build one isolated native PaperQA index for an immutable collection revision."""

import asyncio
from dataclasses import dataclass
from importlib.metadata import version
import gc
import json
import hashlib
import math
import os
from pathlib import Path
import shutil
from urllib.parse import urlsplit
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from backend.app.collections.service import get_collection_revision, utcnow
from backend.app.ingestion.preparation import TextPreparationConfig, prepare_document_version
from backend.app.ingestion.service import InvalidDocumentBlob, _checked_blob_ref
from backend.app.ingestion.storage import StoredPdf, _hash_file
from backend.app.models import DocumentVersion, IndexArtifact, Job, JobAttempt
from backend.app.jobs.ownership import LostClaim

from paperqa import Docs, Settings
from paperqa.agents.search import SearchIndex


@dataclass(frozen=True, slots=True)
class IndexArtifactRecord:
    id: str
    revision_id: str
    config_hash: str
    config_json: str
    artifact_ref: str | None
    status: str
    error_kind: str | None
    error_message: str | None


def _embedding_spec(settings: Settings) -> dict:
    config = settings.embedding_config or {}
    if set(config) != {"dimensions", "batch_size", "kwargs"}:
        raise ValueError("Unsupported embedding configuration for a reproducible index.")
    kwargs = config["kwargs"]
    if set(kwargs) - {
        "api_base", "api_key", "timeout", "num_retries", "encoding_format",
        "allowed_openai_params", "client",
    }:
        raise ValueError("Unrecognized embedding options could change index content.")
    base_url = kwargs.get("api_base")
    parsed = urlsplit(base_url or "")
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Embedding base URL must be HTTPS and contain no credentials or query.")
    dimensions = config["dimensions"]
    if not isinstance(dimensions, int) or dimensions < 1:
        raise ValueError("Embedding dimensions must be positive.")
    return {
        "model": settings.embedding,
        "base_url": base_url,
        "dimensions": dimensions,
        "batch_size": config["batch_size"],
        "encoding_format": kwargs.get("encoding_format", "float"),
        "allowed_openai_params": sorted(kwargs.get("allowed_openai_params", [])),
        "lmi": version("fhlmi"),
    }


def _preparation_config(settings: Settings) -> TextPreparationConfig:
    parsing = settings.parsing
    if (
        parsing.use_doc_details
        or parsing.disable_doc_valid_check
        or parsing.defer_embedding
        or parsing.doc_filters
        or parsing.should_parse_and_enrich_media != (False, False)
        or set(parsing.reader_config) != {"chunk_chars", "overlap"}
    ):
        raise ValueError("Only the checked text-only PaperQA parsing configuration is supported.")
    return TextPreparationConfig(
        chunk_chars=parsing.reader_config["chunk_chars"],
        overlap=parsing.reader_config["overlap"],
        page_size_limit=parsing.page_size_limit,
    )


def _citation(*, title: str, authors: list[str], year: str, versioned_id: str) -> str:
    byline = ", ".join(authors) if authors else "Authors unavailable"
    return f"{byline}. {title}. arXiv:{versioned_id} ({year})."


def _identity(engine: Engine, revision_id: str, settings: Settings) -> tuple[dict, str, str, list[dict]]:
    revision = get_collection_revision(engine, revision_id)
    if revision is None or not revision.members:
        raise LookupError("A nonempty confirmed collection revision is required.")
    prep_config = _preparation_config(settings)
    documents = []
    with Session(engine) as session:
        for member in revision.members:
            document = session.get(DocumentVersion, member.document_id)
            if (
                document is None
                or document.sha256 != member.sha256
                or document.versioned_source_id != member.versioned_id
            ):
                raise ValueError("Frozen member no longer matches its document version.")
            authors = json.loads(document.authors_json)
            year = document.published_at[:4]
            documents.append({
                "document_id": document.id,
                "versioned_id": document.versioned_source_id,
                "sha256": document.sha256,
                "byte_count": document.byte_count,
                "page_count": document.page_count,
                "blob_ref": document.blob_ref,
                "file_location": f"{document.id}.pdf",
                "title": document.title,
                "authors": authors,
                "year": year,
                "citation": _citation(
                    title=document.title, authors=authors, year=year,
                    versioned_id=document.versioned_source_id,
                ),
                "preparation_id": member.preparation_id,
                "parser_config_hash": member.parser_config_hash,
            })
    parser_root = settings.parsing.parse_pdf.__module__.split(".", 1)[0]
    parser_package = {
        "paperqa_pypdf": "paper-qa-pypdf",
        "paperqa_pymupdf": "paper-qa-pymupdf",
    }.get(parser_root)
    if parser_package is None:
        raise ValueError("Unsupported PaperQA PDF parser for this index.")
    spec = {
        "schema": 1,
        "revision_id": revision.id,
        "revision_snapshot_hash": revision.snapshot_hash,
        "paperqa": version("paper-qa"),
        "preparation": {
            "chunk_chars": prep_config.chunk_chars,
            "overlap": prep_config.overlap,
            "page_size_limit": prep_config.page_size_limit,
            "pdf_parser": f"{settings.parsing.parse_pdf.__module__}.{settings.parsing.parse_pdf.__qualname__}",
            "parser_package": parser_package,
            "parser_version": version(parser_package),
            "pypdf": version("pypdf"),
            "media": "off",
            "use_doc_details": False,
        },
        "embedding": _embedding_spec(settings),
        "metadata_policy": "arxiv-version-citation-v2",
        "native_index_fields": [*SearchIndex.REQUIRED_FIELDS, "title", "year"],
        "documents": sorted(documents, key=lambda doc: (doc["versioned_id"], doc["sha256"])),
    }
    spec_json = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    spec_hash = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()
    return spec, spec_json, spec_hash, documents


def get_index_artifact(
    engine: Engine, revision_id: str, *, settings: Settings
) -> IndexArtifactRecord | None:
    _spec, _json, config_hash, _documents = _identity(engine, revision_id, settings)
    with Session(engine) as session:
        artifact = session.scalar(
            select(IndexArtifact).where(
                IndexArtifact.revision_id == revision_id,
                IndexArtifact.config_hash == config_hash,
            )
        )
        return None if artifact is None else _record(artifact)


async def build_collection_index(
    engine: Engine,
    revision_id: str,
    *,
    data_root: Path,
    settings: Settings,
    embedding_model,
    defer_ready: bool = False,
    active_claim: tuple[str, str, str] | None = None,
) -> IndexArtifactRecord:
    if defer_ready != (active_claim is not None):
        raise ValueError("Deferred index publication requires an active Job claim.")
    spec, spec_json, config_hash, documents = _identity(engine, revision_id, settings)
    if getattr(embedding_model, "ndim", None) != spec["embedding"]["dimensions"]:
        raise ValueError("Embedding model dimensions differ from the index configuration.")
    final_parent = Path(data_root) / "indexes" / "collections"
    final_parent.mkdir(parents=True, exist_ok=True)
    final_dir = final_parent / config_hash
    reused_ready = False
    with Session(engine) as session:
        if active_claim is not None:
            job_id, attempt_id, worker_id = active_claim
            job = session.get(Job, job_id)
            attempt = session.get(JobAttempt, attempt_id)
            if (
                job is None or job.kind != "index" or job.resource_id != revision_id
                or job.status != "running" or job.active_attempt_id != attempt_id
                or attempt is None or attempt.job_id != job_id
                or attempt.worker_id != worker_id or attempt.status != "running"
            ):
                raise LostClaim("Index attempt is no longer the active worker.")
        artifact = session.scalar(
            select(IndexArtifact).where(
                IndexArtifact.revision_id == revision_id,
                IndexArtifact.config_hash == config_hash,
            )
        )
        if artifact is not None and artifact.status == "building" and not defer_ready:
            raise RuntimeError("Index build is already in progress; inspect it before retrying.")
        if artifact is not None and artifact.status == "ready" and defer_ready:
            reused_ready = True
        if artifact is None:
            artifact = IndexArtifact(
                id=uuid4().hex,
                revision_id=revision_id,
                config_hash=config_hash,
                config_json=spec_json,
                status="building",
                created_at=utcnow(),
                updated_at=utcnow(),
            )
            session.add(artifact)
        elif not reused_ready:
            artifact.status = "building"
            artifact.artifact_ref = None
            artifact.error_kind = None
            artifact.error_message = None
            artifact.updated_at = utcnow()
        session.commit()
        artifact_id = artifact.id
        ready_record = _record(artifact) if reused_ready else None

    stage_parent: Path | None = None
    try:
        if reused_ready:
            await _verify_index(final_parent, config_hash, spec, data_root)
            return ready_record
        await _check_inputs(engine, revision_id, documents, data_root, settings)
        if final_dir.exists():
            await _verify_index(final_parent, config_hash, spec, data_root)
        else:
            stage_parent = final_parent / f".stage-{uuid4().hex}"
            stage_dir = stage_parent / config_hash
            scope = stage_dir / "scope"
            scope.mkdir(parents=True, exist_ok=False)
            index = SearchIndex(
                fields=spec["native_index_fields"],
                index_name=config_hash,
                index_directory=str(stage_parent),
            )
            reusable = await _reusable_document_vectors(engine, spec, data_root)
            for document in documents:
                source = Path(data_root) / document["blob_ref"]
                path = scope / document["file_location"]
                os.link(source, path)
                docs = reusable.get(document["document_id"])
                if docs is None:
                    docs = Docs()
                    added = await docs.aadd(
                        path,
                        citation=document["citation"],
                        docname="P" + document["document_id"][:10],
                        dockey=document["versioned_id"],
                        title=document["title"],
                        authors=document["authors"],
                        settings=settings,
                        embedding_model=embedding_model,
                    )
                    if not added:
                        raise ValueError("PaperQA did not add the selected PDF to the index.")
                if not docs.texts:
                    raise ValueError("PaperQA did not add the selected PDF to the index.")
                await index.add_document(
                    {
                        "title": document["title"],
                        "year": document["year"],
                        "file_location": document["file_location"],
                        # Native storage keys serialized Docs by the indexed body.
                        # Different fixed versions must not collide on equal text.
                        "body": f"arXiv:{document['versioned_id']}\n" + "\n".join(text.text for text in docs.texts),
                    },
                    document=docs,
                    lock_acquisition_max_retries=3,
                )
            await index.save_index()
            (stage_dir / "manifest.json").write_text(
                json.dumps({"config_hash": config_hash, "spec": spec}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            await _verify_index(stage_parent, config_hash, spec, data_root, opened=index)
            # Tantivy keeps Windows directory handles while SearchIndex is alive.
            del index
            gc.collect()
            # Both source and destination are checked before moving a whole directory.
            parent = final_parent.resolve(strict=True)
            if not stage_dir.resolve(strict=True).is_relative_to(parent) or not final_dir.resolve().is_relative_to(parent):
                raise ValueError("Index publication path escaped its data root.")
            if final_dir.exists():
                raise ValueError("An index appeared at the publication path during build.")
            stage_dir.rename(final_dir)
            await _verify_index(final_parent, config_hash, spec, data_root)
        if defer_ready:
            with Session(engine) as session:
                return _record(session.get(IndexArtifact, artifact_id))
        with Session(engine) as session:
            artifact = session.get(IndexArtifact, artifact_id)
            artifact.status = "ready"
            artifact.artifact_ref = final_dir.relative_to(Path(data_root)).as_posix()
            artifact.updated_at = utcnow()
            session.commit()
            return _record(artifact)
    except Exception as error:
        if not defer_ready:
            with Session(engine) as session:
                artifact = session.get(IndexArtifact, artifact_id)
                artifact.status = "failed"
                artifact.artifact_ref = None
                artifact.error_kind = type(error).__name__
                artifact.error_message = str(error)[:800]
                artifact.updated_at = utcnow()
                session.commit()
        raise
    finally:
        if stage_parent is not None and stage_parent.exists():
            parent = final_parent.resolve(strict=True)
            target = stage_parent.resolve(strict=True)
            if target.is_relative_to(parent) and target.name.startswith(".stage-"):
                try:
                    shutil.rmtree(target)
                except OSError:
                    # The ignored staging directory is never considered a ready cache.
                    pass


async def _reusable_document_vectors(engine: Engine, spec: dict, data_root: Path) -> dict[str, Docs]:
    """Copy validated per-document Docs into a new strictly scoped native index."""
    desired = {doc['document_id']: doc for doc in spec['documents']}
    reusable = {}
    with Session(engine) as session:
        records = [_record(row) for row in session.scalars(select(IndexArtifact).where(IndexArtifact.status == 'ready')
            .order_by(IndexArtifact.updated_at.desc(), IndexArtifact.id.desc()))]
    parent = Path(data_root) / 'indexes' / 'collections'
    for artifact in records:
        if len(reusable) == len(desired): break
        try:
            cached = json.loads(artifact.config_json)
            if artifact.artifact_ref != f'indexes/collections/{artifact.config_hash}' or hashlib.sha256(artifact.config_json.encode()).hexdigest() != artifact.config_hash:
                continue
            # v1/v2 only differ in native search-body identity, not chunk vectors.
            if cached.get('metadata_policy') not in ('arxiv-version-citation-v1', 'arxiv-version-citation-v2'):
                continue
            if any(cached.get(key) != spec.get(key) for key in ('schema', 'paperqa', 'preparation', 'embedding', 'native_index_fields')):
                continue
            candidates = [doc for doc in cached['documents'] if doc['document_id'] not in reusable and desired.get(doc['document_id']) == doc]
            if not candidates: continue
            directory = (parent / artifact.config_hash).resolve(strict=True)
            if not directory.is_relative_to(parent.resolve(strict=True)): continue
            await _verify_index(parent, artifact.config_hash, cached, data_root)
            opened = SearchIndex(fields=cached['native_index_fields'], index_name=artifact.config_hash, index_directory=str(parent))
            for document in candidates:
                reusable[document['document_id']] = await opened.get_saved_object(document['file_location'])
            del opened
        except (OSError, ValueError, KeyError, TypeError):
            # A cache miss rebuilds selected documents; never relax scope checks.
            continue
    return reusable


async def _check_inputs(
    engine: Engine, revision_id: str, documents: list[dict], data_root: Path, settings: Settings
) -> None:
    config = _preparation_config(settings)
    revision = get_collection_revision(engine, revision_id)
    by_id = {member.document_id: member for member in revision.members}
    for document in documents:
        member = by_id[document["document_id"]]
        stored = StoredPdf(
            versioned_id=document["versioned_id"],
            path=Path(data_root) / document["blob_ref"],
            sha256=document["sha256"],
            byte_count=document["byte_count"],
            page_count=document["page_count"],
            reused=True,
        )
        if _checked_blob_ref(stored, data_root) != document["blob_ref"]:
            raise InvalidDocumentBlob("Index input PDF is not canonical.")
        prepared = await asyncio.to_thread(
            lambda document_id=document["document_id"]: asyncio.run(
                prepare_document_version(engine, document_id, data_root=data_root, config=config)
            )
        )
        if (
            prepared.id != member.preparation_id
            or prepared.parser_config_hash != member.parser_config_hash
            or prepared.status != "succeeded"
            or json.loads(prepared.parser_config_json)["pdf_parser"]
            != f"{settings.parsing.parse_pdf.__module__}.{settings.parsing.parse_pdf.__qualname__}"
        ):
            raise ValueError("Frozen preparation does not match the current PaperQA parser.")


async def _verify_index(
    index_parent: Path,
    index_name: str,
    spec: dict,
    data_root: Path,
    *,
    opened: SearchIndex | None = None,
) -> None:
    directory = index_parent / index_name
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Index manifest is missing or invalid.") from error
    if manifest != {"config_hash": index_name, "spec": spec}:
        raise ValueError("Index manifest does not match the requested revision and configuration.")
    expected = {document["file_location"] for document in spec["documents"]}
    scope = directory / "scope"
    if {path.name for path in scope.glob("*.pdf")} != expected:
        raise ValueError("Index scope directory differs from the frozen members.")
    for document in spec["documents"]:
        path = scope / document["file_location"]
        digest, size = _hash_file(path)
        if digest != document["sha256"] or size != document["byte_count"]:
            raise ValueError("Index scoped PDF bytes changed.")
    if opened is None:
        opened = SearchIndex(
            fields=spec["native_index_fields"],
            index_name=index_name,
            index_directory=str(index_parent),
        )
    if set(await opened.index_files) != expected or await opened.count != len(expected):
        raise ValueError("Native PaperQA index has the wrong document scope.")
    for document in spec["documents"]:
        saved = await opened.get_saved_object(document["file_location"])
        if (
            not isinstance(saved, Docs)
            or set(saved.docs) != {document["versioned_id"]}
            or not saved.texts
        ):
            raise ValueError("Native PaperQA index cannot reload its selected document.")
        for text in saved.texts:
            vector = text.embedding
            if (
                text.doc.dockey != document["versioned_id"]
                or vector is None
                or len(vector) != spec["embedding"]["dimensions"]
                or not any(value != 0 for value in vector)
                or any(not math.isfinite(value) for value in vector)
            ):
                raise ValueError("Native PaperQA index contains an invalid document vector.")


def settings_for_artifact(settings: Settings, artifact: IndexArtifactRecord, *, data_root: Path) -> Settings:
    if artifact.status != "ready" or artifact.artifact_ref != f"indexes/collections/{artifact.config_hash}":
        raise ValueError("Only a ready, canonical index artifact can be opened.")
    root = Path(data_root).resolve(strict=True)
    directory = (root / artifact.artifact_ref).resolve(strict=True)
    if not directory.is_relative_to(root):
        raise ValueError("Index artifact escaped the data root.")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("config_hash") != artifact.config_hash
        or json.dumps(manifest.get("spec"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        != artifact.config_json
        or hashlib.sha256(artifact.config_json.encode("utf-8")).hexdigest() != artifact.config_hash
        or manifest["spec"]["embedding"] != _embedding_spec(settings)
    ):
        raise ValueError("Index artifact manifest or embedding settings changed.")
    parsing = _preparation_config(settings)
    saved_parsing = manifest["spec"]["preparation"]
    parser_root = settings.parsing.parse_pdf.__module__.split(".", 1)[0]
    parser_package = {
        "paperqa_pypdf": "paper-qa-pypdf",
        "paperqa_pymupdf": "paper-qa-pymupdf",
    }.get(parser_root)
    if (
        saved_parsing["chunk_chars"] != parsing.chunk_chars
        or saved_parsing["overlap"] != parsing.overlap
        or saved_parsing["page_size_limit"] != parsing.page_size_limit
        or saved_parsing["pdf_parser"]
        != f"{settings.parsing.parse_pdf.__module__}.{settings.parsing.parse_pdf.__qualname__}"
        or parser_package is None
        or saved_parsing["parser_package"] != parser_package
        or saved_parsing["parser_version"] != version(parser_package)
        or saved_parsing["pypdf"] != version("pypdf")
        or manifest["spec"]["paperqa"] != version("paper-qa")
    ):
        raise ValueError("Index artifact parser settings changed.")
    scope = directory / "scope"
    expected = {document["file_location"] for document in manifest["spec"]["documents"]}
    if {path.name for path in scope.glob("*.pdf")} != expected:
        raise ValueError("Index artifact PDF scope changed.")
    for document in manifest["spec"]["documents"]:
        digest, size = _hash_file(scope / document["file_location"])
        if digest != document["sha256"] or size != document["byte_count"]:
            raise ValueError("Index artifact PDF bytes changed.")
    configured = settings.model_copy(deep=True)
    configured.agent.index.name = artifact.config_hash
    configured.agent.index.index_directory = str(directory.parent)
    configured.agent.index.paper_directory = str(directory / "scope")
    configured.agent.index.recurse_subdirectories = False
    configured.agent.index.sync_with_paper_directory = False
    configured.agent.rebuild_index = False
    return configured


def _record(artifact: IndexArtifact) -> IndexArtifactRecord:
    return IndexArtifactRecord(
        id=artifact.id,
        revision_id=artifact.revision_id,
        config_hash=artifact.config_hash,
        config_json=artifact.config_json,
        artifact_ref=artifact.artifact_ref,
        status=artifact.status,
        error_kind=artifact.error_kind,
        error_message=artifact.error_message,
    )
