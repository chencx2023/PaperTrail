"""Read, rename and incrementally associate legacy immutable research results."""
import base64
import hashlib
import json
from pathlib import PurePosixPath
from uuid import uuid4
from sqlalchemy import select, update, or_, and_
from sqlalchemy.orm import Session
from backend.app.models import (
    ResearchRecord, ResearchOperation, SearchRun, Collection, CollectionRevision,
    CollectionRevisionDocument, CollectionImportItem, DocumentVersion,
    AnswerRun, AnswerSubmission, Job, JobRequest,
)
from backend.app.collections.service import utcnow
from backend.app.collections.revision_submission import read_revision_state


def new_record(session, *, search_id=None, parent=None, origin_key=None, created_at=None):
    search = session.get(SearchRun, search_id) if search_id else None
    now = created_at or utcnow()
    record = ResearchRecord(id=uuid4().hex, origin_key=origin_key,
        parent_id=parent.id if parent else None, search_id=search_id,
        title=search.question if search else '论文研究', title_is_custom=False,
        collection_id=parent.collection_id if parent else None,
        pool_revision_id=parent.pool_revision_id if parent else None,
        prepare_operation_id=parent.prepare_operation_id if parent else None,
        created_at=now, updated_at=now)
    session.add(record)
    session.flush()
    return record


def ensure_search_record(engine, search_id):
    with Session(engine) as session:
        session.execute(update(ResearchRecord).where(ResearchRecord.id == '').values(updated_at=ResearchRecord.updated_at))
        record = session.scalar(select(ResearchRecord).where(ResearchRecord.origin_key == f'search:{search_id}'))
        if record is None:
            search = session.get(SearchRun, search_id)
            if search is None: raise LookupError('检索记录不存在。')
            record = new_record(session, search_id=search_id, origin_key=f'search:{search_id}', created_at=search.created_at)
            session.commit()
        return record.id


def _search_for_revision(session, revision):
    # Use the exact source references of this frozen revision, never today's draft.
    ids = session.scalars(select(CollectionRevisionDocument.document_id).where(CollectionRevisionDocument.revision_id == revision.id)).all()
    sources = set(session.scalars(select(CollectionImportItem.source_ref).where(
        CollectionImportItem.import_run_id == revision.import_run_id,
        CollectionImportItem.document_id.in_(ids), CollectionImportItem.status == 'succeeded')).all())
    if len(sources) != 1: return None
    source = next(iter(sources))
    matches = [r.id for r in session.scalars(select(SearchRun).where(SearchRun.snapshot_ref.is_not(None)))
               if PurePosixPath(r.snapshot_ref).parent.as_posix() == source]
    return matches[0] if len(matches) == 1 else None


def _pool_contains_revision(session, pool_id, scope):
    pool = session.get(CollectionRevision, pool_id) if pool_id else None
    if not pool or pool.collection_id != scope.collection_id or pool.import_run_id != scope.import_run_id: return False
    def identities(id):
        return set(session.execute(select(CollectionRevisionDocument.document_id,
            CollectionRevisionDocument.versioned_id, CollectionRevisionDocument.sha256,
            CollectionRevisionDocument.preparation_id, CollectionRevisionDocument.parser_config_hash)
            .where(CollectionRevisionDocument.revision_id == id)).all())
    members = identities(scope.id)
    return bool(members) and members.issubset(identities(pool.id))


def _pending_managed_answer(session, job, payload):
    """The durable chain, rather than legacy migration, owns this enqueue gap."""
    if not job: return False
    prefix, _, suffix = job.idempotency_key.partition(':')
    operation_id, _, kind = suffix.partition(':')
    if prefix != 'research' or kind != 'answer': return False
    operation = session.get(ResearchOperation, operation_id)
    record = session.get(ResearchRecord, operation.research_id) if operation else None
    if not operation or operation.kind != 'answer' or operation.status != 'pending' or not record or record.operation_id != operation.id: return False
    if hashlib.sha256(operation.request_json.encode()).hexdigest() != operation.request_hash: return False
    request = json.loads(operation.request_json)
    return payload.get('revision_id') == operation.revision_id and payload.get('question') == request.get('question')


def backfill_research_records(engine):
    """Add metadata at startup. No artifact changes and no jobs are submitted."""
    with Session(engine) as session:
        session.execute(update(ResearchRecord).where(ResearchRecord.id == '').values(updated_at=ResearchRecord.updated_at))
        for search in session.scalars(select(SearchRun).order_by(SearchRun.created_at, SearchRun.id)):
            if not session.scalar(select(ResearchRecord.id).where(ResearchRecord.origin_key == f'search:{search.id}')):
                new_record(session, search_id=search.id, origin_key=f'search:{search.id}', created_at=search.created_at)
        saved = {r.id: r for r in session.scalars(select(AnswerRun))}
        submissions = {r.answer_run_id: r for r in session.scalars(select(AnswerSubmission))}
        for answer_id in sorted(set(saved) | set(submissions), key=lambda id:
            saved[id].saved_at if id in saved else submissions[id].submitted_at):
            if session.scalar(select(ResearchRecord.id).where(ResearchRecord.answer_run_id == answer_id)): continue
            answer = saved.get(answer_id)
            submission = submissions.get(answer_id)
            request = session.get(JobRequest, submission.job_id) if submission else None
            job = session.get(Job, submission.job_id) if submission else None
            try:
                payload = json.loads(request.request_json) if request and job and hashlib.sha256(request.request_json.encode()).hexdigest() == job.request_hash else {}
                if not isinstance(payload, dict): payload = {}
            except (ValueError, TypeError): payload = {}
            if _pending_managed_answer(session, job, payload): continue
            revision_id = answer.collection_revision_id if answer else payload.get('revision_id')
            revision = session.get(CollectionRevision, revision_id) if revision_id else None
            search_id = _search_for_revision(session, revision) if revision else None
            base = session.scalar(select(ResearchRecord).where(ResearchRecord.origin_key == f'search:{search_id}')) if search_id else None
            record = base if base is not None and not base.answer_run_id and not base.question else new_record(
                session, search_id=search_id, parent=base, origin_key=f'answer:{answer_id}',
                created_at=answer.started_at if answer else submission.submitted_at)
            record.question = answer.question if answer else payload.get('question')
            if not record.title_is_custom: record.title = record.question or record.title
            record.answer_run_id = answer_id
            if revision:
                record.collection_id = revision.collection_id
                if not _pool_contains_revision(session, record.pool_revision_id, revision): record.pool_revision_id = revision.id
                record.scope_revision_id = revision.id
        # Keep collections without an associated answer, including failed imports.
        for collection in session.scalars(select(Collection).order_by(Collection.created_at, Collection.id)):
            if session.scalar(select(ResearchRecord.id).where(ResearchRecord.collection_id == collection.id)): continue
            revision = session.scalar(select(CollectionRevision).where(CollectionRevision.collection_id == collection.id)
                .order_by(CollectionRevision.confirmed_at.desc(), CollectionRevision.id.desc()).limit(1))
            search_id = _search_for_revision(session, revision) if revision else None
            if search_id is None:
                try: selections = json.loads(collection.selected_json)
                except (ValueError, TypeError): selections = []
                sources = {item.get('source_ref') for item in selections if isinstance(item, dict)} if isinstance(selections, list) else set()
                if len(sources) == 1 and next(iter(sources)):
                    matches = [r.id for r in session.scalars(select(SearchRun).where(SearchRun.snapshot_ref.is_not(None)))
                        if PurePosixPath(r.snapshot_ref).parent.as_posix() == next(iter(sources))]
                    if len(matches) == 1: search_id = matches[0]
            base = session.scalar(select(ResearchRecord).where(ResearchRecord.origin_key == f'search:{search_id}')) if search_id else None
            record = base if base is not None and not base.collection_id and not base.question else new_record(
                session, search_id=search_id, parent=base, origin_key=f'collection:{collection.id}', created_at=collection.created_at)
            record.collection_id = collection.id
            record.pool_revision_id = revision.id if revision else None
        # Each older frozen revision remains addressable, even if a newer one of
        # the same collection already belongs to a different saved answer.
        for revision in session.scalars(select(CollectionRevision).order_by(CollectionRevision.confirmed_at, CollectionRevision.id)):
            if session.scalar(select(ResearchRecord.id).where(or_(ResearchRecord.pool_revision_id == revision.id, ResearchRecord.scope_revision_id == revision.id))): continue
            if session.scalar(select(ResearchOperation.id).join(ResearchRecord, ResearchRecord.operation_id == ResearchOperation.id).where(ResearchRecord.collection_id == revision.collection_id, ResearchOperation.status == 'pending')): continue
            search_id = _search_for_revision(session, revision)
            base = session.scalar(select(ResearchRecord).where(ResearchRecord.origin_key == f'search:{search_id}')) if search_id else None
            record = new_record(session, search_id=search_id, parent=base, origin_key=f'revision:{revision.id}', created_at=revision.confirmed_at)
            record.collection_id = revision.collection_id; record.pool_revision_id = revision.id
        session.commit()


def resolve_research(engine, *, search_id=None, collection_id=None, revision_id=None, answer_id=None):
    with Session(engine) as session:
        if answer_id: condition = ResearchRecord.answer_run_id == answer_id
        elif revision_id: condition = or_(ResearchRecord.scope_revision_id == revision_id, ResearchRecord.pool_revision_id == revision_id)
        elif collection_id: condition = ResearchRecord.collection_id == collection_id
        elif search_id: condition = ResearchRecord.origin_key == f'search:{search_id}'
        else: return None
        return session.scalar(select(ResearchRecord.id).where(condition)
            .order_by(ResearchRecord.created_at.desc(), ResearchRecord.id.desc()).limit(1))


def rename_research(engine, id, title):
    title = title.strip()
    if not title or len(title) > 200: raise ValueError('名称需为 1–200 个字符。')
    with Session(engine) as session:
        record = session.get(ResearchRecord, id)
        if record is None: raise LookupError('研究记录不存在。')
        record.title = title
        record.title_is_custom = True
        record.updated_at = utcnow()
        session.commit()


def read_research(engine, id):
    with Session(engine) as session:
        record = session.get(ResearchRecord, id)
        if record is None: return None
        operation = session.get(ResearchOperation, record.operation_id) if record.operation_id else None
        prepare = session.get(ResearchOperation, record.prepare_operation_id) if record.prepare_operation_id else None
        members = session.scalars(select(CollectionRevisionDocument).where(
            CollectionRevisionDocument.revision_id == record.pool_revision_id).order_by(CollectionRevisionDocument.position)).all() if record.pool_revision_id else []
        papers = []
        for member in members:
            doc = session.get(DocumentVersion, member.document_id)
            if doc is None or doc.versioned_source_id != member.versioned_id or doc.sha256 != member.sha256:
                raise ValueError('保存的论文范围无法核对。')
            papers.append({'document_id': doc.id, 'versioned_id': member.versioned_id,
                'title': doc.title, 'authors': json.loads(doc.authors_json)})
        selected_versions = json.loads(prepare.request_json).get('versioned_ids', []) if prepare else [m.versioned_id for m in members]
        if not selected_versions and record.collection_id and not record.question:
            collection = session.get(Collection, record.collection_id)
            selected_versions = [x['versioned_id'] for x in json.loads(collection.selected_json)] if collection else []
        operation_data = None if operation is None else {
            'operation_id': operation.id, 'kind': operation.kind, 'phase': operation.phase,
            'status': operation.status, 'job_id': operation.job_id, 'message': operation.message,
            'document_ids': json.loads(operation.request_json).get('document_ids', [])}
        report = json.loads(prepare.report_json) if prepare and prepare.report_json else None
        pool = read_revision_state(engine, record.collection_id, record.pool_revision_id) if record.pool_revision_id else None
        scope = session.scalars(select(CollectionRevisionDocument.document_id).where(
            CollectionRevisionDocument.revision_id == record.scope_revision_id).order_by(CollectionRevisionDocument.position)).all() if record.scope_revision_id else []
        return {'research_id': record.id, 'parent_id': record.parent_id,
            'title': record.title, 'title_is_custom': record.title_is_custom,
            'search_id': record.search_id, 'collection_id': record.collection_id,
            'pool_revision_id': record.pool_revision_id, 'scope_revision_id': record.scope_revision_id,
            'answer_run_id': record.answer_run_id, 'question': record.question,
            'papers': papers, 'selected_versions': selected_versions,
            'answer_document_ids': scope or (operation_data['document_ids'] if operation_data and operation.kind == 'answer' else []),
            'preparation': pool, 'operation': operation_data, 'import_report': report,
            'created_at': record.created_at, 'updated_at': record.updated_at}


def list_researches(engine, *, limit=12, cursor=None):
    if not 1 <= limit <= 100: raise ValueError('无效分页大小。')
    query = select(ResearchRecord).order_by(ResearchRecord.created_at.desc(), ResearchRecord.id.desc())
    if cursor:
        try:
            date, id = json.loads(base64.urlsafe_b64decode(cursor + '=' * (-len(cursor) % 4)))
            if not isinstance(date, str) or not isinstance(id, str) or len(id) != 32: raise ValueError()
        except Exception as error: raise ValueError('无效历史分页。') from error
        query = query.where(or_(ResearchRecord.created_at < date,
            and_(ResearchRecord.created_at == date, ResearchRecord.id < id)))
    with Session(engine) as session:
        rows = session.scalars(query.limit(limit + 1)).all()
        items = []
        for row in rows[:limit]:
            op = session.get(ResearchOperation, row.operation_id) if row.operation_id else None
            job = session.scalar(select(Job).where(Job.kind == 'answer', Job.resource_id == row.answer_run_id)) if row.answer_run_id else None
            status = job.status if job else op.status if op else 'saved'
            if not job and not op and not row.answer_run_id:
                source = session.scalar(select(Job).where(Job.kind == 'index' if row.pool_revision_id else Job.kind == 'import' if row.collection_id else Job.kind == 'search', Job.resource_id == (row.pool_revision_id or row.collection_id or row.search_id)).order_by(Job.created_at.desc(), Job.id.desc()).limit(1))
                if source: status = source.status
            if row.answer_run_id and job is None:
                answer = session.get(AnswerRun, row.answer_run_id)
                status = answer.execution_status if answer else 'saved'
            items.append({'research_id': row.id, 'title': row.title, 'status': status,
                'question': row.question, 'created_at': row.created_at,
                'href': f'/?research={row.id}&view={"qa" if row.pool_revision_id or row.answer_run_id or op else "search"}'})
        next_cursor = base64.urlsafe_b64encode(json.dumps([rows[limit-1].created_at, rows[limit-1].id]).encode()).decode().rstrip('=') if len(rows) > limit else None
        return {'items': items, 'next_cursor': next_cursor}
