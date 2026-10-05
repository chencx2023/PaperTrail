"""Small durable import/index/answer chain. Only explicit intents are advanced."""
from dataclasses import asdict
import hashlib
import json
from uuid import uuid4
from sqlalchemy import select, update, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from backend.app.models import ResearchRecord, ResearchOperation, CollectionRevision, CollectionRevisionDocument, Job, JobRequest, JobAttempt
from backend.app.collections.service import utcnow, StaleSelection, _revision_record
from backend.app.collections.submission import create_collection_with_import, get_collection_state, SearchChoice, _resolve_choices
from backend.app.collections.revision_submission import submit_revision_index, read_revision_state
from backend.app.jobs.service import IdempotencyConflict, get_job
from backend.app.qa.submission import submit_answer
from backend.app.research.records import new_record, read_research


class ResearchConflict(ValueError):
    pass


def _encoded(payload):
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _submit(engine, id, *, kind, body, idempotency_key, data_root):
    if not idempotency_key or not idempotency_key.strip(): raise ValueError('需要提交标识。')
    request = _encoded({'research_id': id, 'kind': kind, **body})
    digest = hashlib.sha256(request.encode()).hexdigest()
    with Session(engine) as session:
        existing = session.scalar(select(ResearchOperation).where(ResearchOperation.idempotency_key == idempotency_key))
        if existing:
            if existing.request_hash != digest or existing.request_json != request:
                raise IdempotencyConflict('提交标识已用于另一项操作。')
            return read_research(engine, existing.research_id)
        # A write locks the metadata before inspecting pending state or cloning.
        locked = session.execute(update(ResearchRecord).where(ResearchRecord.id == id)
            .values(updated_at=ResearchRecord.updated_at).returning(ResearchRecord.id)).scalar_one_or_none()
        if locked is None: raise LookupError('研究记录不存在。')
        # Another request may have committed this key while we waited for the
        # writer lock. Replay before evaluating its now-pending operation.
        existing = session.scalar(select(ResearchOperation).where(ResearchOperation.idempotency_key == idempotency_key))
        if existing:
            if existing.request_hash != digest or existing.request_json != request:
                raise IdempotencyConflict('提交标识已用于另一项操作。')
            return read_research(engine, existing.research_id)
        record = session.get(ResearchRecord, id)
        if record.pool_revision_id != body['expected_pool_revision_id']:
            raise ResearchConflict('论文范围已更新，请刷新后再提交。')
        active = session.get(ResearchOperation, record.operation_id) if record.operation_id else None
        if active and active.status == 'pending': raise ResearchConflict('当前研究仍在处理，请等待完成。')
        if kind == 'prepare':
            versions = body['versioned_ids']
            if not versions or len(versions) > 10 or len(set(versions)) != len(versions):
                raise ValueError('请选择 1–10 篇不同的固定版本。')
            if not record.search_id: raise ResearchConflict('这份旧记录没有可核对的检索来源，请重新检索。')
            _resolve_choices(session, data_root=data_root, choices=tuple(SearchChoice(record.search_id, v) for v in versions))
        else:
            question = body['question']
            if not question.strip() or len(question) > 10000: raise ValueError('请输入 1–10000 个字符的问题。')
            ids = body['document_ids']
            members = session.scalars(select(CollectionRevisionDocument.document_id).where(
                CollectionRevisionDocument.revision_id == record.pool_revision_id)).all()
            if not ids or len(set(ids)) != len(ids) or any(id not in members for id in ids):
                raise ValueError('请选择当前研究中准备成功的论文。')
            pool = read_revision_state(engine, record.collection_id, record.pool_revision_id) if record.pool_revision_id else None
            if not pool or not pool['index_artifact_id']: raise ResearchConflict('论文尚未准备完成。')
        # Every subsequent generation/changed preparation keeps the previous record.
        target = new_record(session, search_id=record.search_id, parent=record) if record.question or record.answer_run_id else record
        now = utcnow()
        operation = ResearchOperation(id=uuid4().hex, research_id=target.id,
            idempotency_key=idempotency_key, request_hash=digest, request_json=request,
            kind=kind, phase='start', status='pending', created_at=now, updated_at=now)
        session.add(operation)
        target.operation_id = operation.id
        target.updated_at = now
        if kind == 'prepare': target.prepare_operation_id = operation.id
        else:
            target.question = body['question']
            if not target.title_is_custom: target.title = target.question
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            existing = session.scalar(select(ResearchOperation).where(ResearchOperation.idempotency_key == idempotency_key))
            if existing is None: raise
            if existing.request_hash != digest: raise IdempotencyConflict('提交标识已用于另一项操作。')
            return read_research(engine, existing.research_id)
        target_id = target.id
    return read_research(engine, target_id)


def prepare_research(engine, id, *, versioned_ids, expected_pool_revision_id, idempotency_key, data_root):
    return _submit(engine, id, kind='prepare', body={'versioned_ids': list(versioned_ids),
        'expected_pool_revision_id': expected_pool_revision_id}, idempotency_key=idempotency_key, data_root=data_root)


def answer_research(engine, id, *, question, document_ids, expected_pool_revision_id, idempotency_key, data_root):
    return _submit(engine, id, kind='answer', body={'question': question, 'document_ids': list(document_ids),
        'expected_pool_revision_id': expected_pool_revision_id}, idempotency_key=idempotency_key, data_root=data_root)


def retry_research(engine, id, *, operation_id, idempotency_key, data_root):
    # New explicit intent; successes reuse their content-addressed PDFs and cache.
    with Session(engine) as session:
        previous = session.get(ResearchOperation, operation_id)
        record = session.get(ResearchRecord, id)
        replay = session.scalar(select(ResearchOperation).where(ResearchOperation.idempotency_key == idempotency_key))
        if replay:
            payload = json.loads(replay.request_json)
            if payload.get('retry_operation_id') != operation_id or payload.get('research_id') != id:
                raise IdempotencyConflict('提交标识已用于另一项操作。')
            return read_research(engine, replay.research_id)
        if not previous or not record or previous.research_id != id or record.operation_id != operation_id or previous.status != 'failed':
            raise ResearchConflict('这次任务不可重试，请刷新状态。')
        body = json.loads(previous.request_json)
        body.pop('research_id'); body.pop('kind')
        body['expected_pool_revision_id'] = record.pool_revision_id
        body['retry_operation_id'] = operation_id
        kind = previous.kind
    return _submit(engine, id, kind=kind, body=body, idempotency_key=idempotency_key, data_root=data_root)


def _progress(engine, operation, *, record_values=None, **values):
    with Session(engine) as session:
        active = select(ResearchRecord.id).where(ResearchRecord.id == operation.research_id,
            ResearchRecord.operation_id == operation.id).exists()
        changed = session.execute(update(ResearchOperation).where(ResearchOperation.id == operation.id,
            ResearchOperation.phase == operation.phase, ResearchOperation.status == 'pending', active)
            .values(updated_at=utcnow(), **values).returning(ResearchOperation.id)).scalar_one_or_none()
        if changed and record_values:
            session.execute(update(ResearchRecord).where(ResearchRecord.id == operation.research_id,
                ResearchRecord.operation_id == operation.id).values(updated_at=utcnow(), **record_values))
        session.commit()


def freeze_answer_subset(engine, pool_revision_id, document_ids):
    """Copy only immutable pool members, independent of the current mutable draft."""
    with Session(engine) as session:
        parent = session.get(CollectionRevision, pool_revision_id)
        if parent is None: raise LookupError('保存的论文池不存在。')
        # Serialize creation of a collection's revision numbers on SQLite.
        from backend.app.models import Collection
        session.execute(update(Collection).where(Collection.id == parent.collection_id).values(updated_at=Collection.updated_at))
        members = [m for m in _revision_record(session, parent).members if m.document_id in document_ids]
        if not members or len(members) != len(set(document_ids)) or len(document_ids) != len(set(document_ids)):
            raise ValueError('回答范围包含当前研究以外的论文。')
        snapshot = {'schema': 1, 'documents': sorted([asdict(m) for m in members], key=lambda m: (m['versioned_id'], m['sha256']))}
        digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        existing = session.scalar(select(CollectionRevision).where(CollectionRevision.collection_id == parent.collection_id,
            CollectionRevision.import_run_id == parent.import_run_id, CollectionRevision.snapshot_hash == digest))
        if existing: return existing.id
        number = (session.scalar(select(func.max(CollectionRevision.revision_no)).where(CollectionRevision.collection_id == parent.collection_id)) or 0) + 1
        revision = CollectionRevision(id=uuid4().hex, collection_id=parent.collection_id,
            import_run_id=parent.import_run_id, selection_revision=parent.selection_revision,
            snapshot_hash=digest, revision_no=number, confirmed_at=utcnow())
        session.add(revision)
        session.flush()
        for position, member in enumerate(members):
            session.add(CollectionRevisionDocument(id=uuid4().hex, revision_id=revision.id, position=position, **asdict(member)))
        session.commit()
        return revision.id


def _scope_index(engine, revision_id, key, settings):
    from backend.app.collections.indexing import get_index_artifact
    artifact = get_index_artifact(engine, revision_id, settings=settings)
    with Session(engine) as session:
        revision = session.get(CollectionRevision, revision_id)
        if revision is None: raise LookupError('回答范围不存在。')
        latest = session.scalar(select(Job).where(Job.kind == 'index', Job.resource_id == revision_id)
            .order_by(Job.created_at.desc(), Job.id.desc()).limit(1))
        if latest and (latest.status in ('queued', 'running') or latest.status == 'succeeded' and artifact and artifact.status == 'ready'):
            return latest.id
        existing = session.scalar(select(Job).where(Job.kind == 'index', Job.idempotency_key == key))
        if existing: return existing.id
        request = _encoded({'collection_id': revision.collection_id, 'selection_revision': revision.selection_revision,
            'import_run_id': revision.import_run_id, 'document_ids': [m.document_id for m in _revision_record(session, revision).members],
            'revision_id': revision.id, 'snapshot_hash': revision.snapshot_hash})
        job = Job(id=uuid4().hex, kind='index', resource_id=revision.id, status='queued', idempotency_key=key,
            request_hash=hashlib.sha256(request.encode()).hexdigest(), created_at=utcnow(), updated_at=utcnow())
        session.add(job); session.flush()
        session.add(JobRequest(job_id=job.id, format_version=1, request_json=request))
        session.commit()
        return job.id


def advance_research_operations(engine, *, data_root, index_settings=None):
    """Called by the singleton dispatcher. No timers or writes in the read APIs."""
    with Session(engine) as session:
        operations = session.scalars(select(ResearchOperation).where(ResearchOperation.status == 'pending')
            .order_by(ResearchOperation.created_at, ResearchOperation.id)).all()
        session.expunge_all()
    for operation in operations:
        try:
            if hashlib.sha256(operation.request_json.encode()).hexdigest() != operation.request_hash:
                raise ValueError('准备请求无法核对。')
            request = json.loads(operation.request_json)
            key = f'research:{operation.id}'
            if operation.phase == 'start':
                if operation.kind == 'prepare':
                    with Session(engine) as session:
                        record = session.get(ResearchRecord, operation.research_id)
                        if record.pool_revision_id != request['expected_pool_revision_id']:
                            raise ResearchConflict('论文范围已改变。')
                        search_id = record.search_id
                    submitted = create_collection_with_import(engine, data_root=data_root,
                        choices=tuple(SearchChoice(search_id, v) for v in request['versioned_ids']), idempotency_key=key + ':import')
                    collection = get_collection_state(engine, submitted.collection_id)
                    _progress(engine, operation, phase='import', collection_id=submitted.collection_id,
                        selection_revision=collection.selection_revision, job_id=submitted.job_id,
                        record_values={'collection_id': submitted.collection_id, 'pool_revision_id': None, 'scope_revision_id': None})
                else:
                    revision_id = freeze_answer_subset(engine, request['expected_pool_revision_id'], request['document_ids'])
                    if index_settings is None:
                        from backend.app.jobs.index_worker import _runtime_settings
                        index_settings = _runtime_settings()
                    job_id = _scope_index(engine, revision_id, key + ':index', index_settings)
                    _progress(engine, operation, phase='index', revision_id=revision_id, job_id=job_id,
                        record_values={'scope_revision_id': revision_id})
                continue
            job = get_job(engine, operation.job_id)
            if job is None: raise LookupError('准备任务记录不存在。')
            if job.status in ('queued', 'running'): continue
            if job.status != 'succeeded': raise ResearchConflict('任务已失败或中断。')
            if operation.phase == 'import':
                collection = get_collection_state(engine, operation.collection_id)
                report = collection.import_report if collection else None
                if not collection or collection.selection_revision != operation.selection_revision or collection.import_job_id != operation.job_id or not report:
                    raise StaleSelection('导入报告与本次选择不一致。')
                successes = tuple(item['document_id'] for item in report['items'] if item['status'] == 'succeeded' and item['document_id'])
                if not successes:
                    _progress(engine, operation, status='failed', message='所有论文导入失败，请重试准备。', report_json=_encoded(report))
                    continue
                submitted = submit_revision_index(engine, operation.collection_id,
                    expected_selection_revision=operation.selection_revision, import_run_id=report['import_run_id'],
                    document_ids=successes, idempotency_key=key + ':index', data_root=data_root)
                _progress(engine, operation, phase='index', revision_id=submitted.revision_id,
                    job_id=submitted.job_id, report_json=_encoded(report),
                    record_values={'pool_revision_id': submitted.revision_id})
            elif operation.phase == 'index':
                with Session(engine) as session:
                    revision = session.get(CollectionRevision, operation.revision_id)
                    collection_id = revision.collection_id if revision else None
                state = read_revision_state(engine, collection_id, operation.revision_id)
                if not state or not state['index_artifact_id']: raise ValueError('索引结果无法核对。')
                if operation.kind == 'prepare':
                    _progress(engine, operation, phase='ready', status='succeeded')
                else:
                    submitted = submit_answer(engine, revision_id=operation.revision_id,
                        index_artifact_id=state['index_artifact_id'], question=request['question'], idempotency_key=key + ':answer')
                    _progress(engine, operation, phase='answer', job_id=submitted.job_id,
                        record_values={'answer_run_id': submitted.answer_run_id})
            elif operation.phase == 'answer':
                _progress(engine, operation, phase='ready', status='succeeded')
            else: raise ValueError('无效准备阶段。')
        except (ValueError, LookupError) as error:
            message = {'import': '导入未完成，请重试准备。', 'index': '索引准备失败，请重试。',
                'answer': '本次回答未完成，请重新生成。'}.get(operation.phase, '这次准备无法继续，请刷新后重试。')
            if isinstance(error, StaleSelection): message = '选择或导入报告已变化，本次准备已停止。'
            _progress(engine, operation, status='failed', error_kind=type(error).__name__, message=message)


def recover_orphan_research_attempts(engine, *, worker_id, singleton_acquired):
    """Only the OS-lock-owning replacement worker can acknowledge a dead owner.

    A live lease is left alone until expiry. Interrupted jobs require an explicit
    retry intent; no external execution or fresh job is submitted here.
    """
    if not singleton_acquired: raise ValueError('Worker singleton takeover must be confirmed.')
    from backend.app.jobs.worker import interrupt_expired_attempt
    with Session(engine) as session:
        attempts = session.scalars(select(JobAttempt.id).join(Job, Job.active_attempt_id == JobAttempt.id)
            .join(ResearchOperation, ResearchOperation.job_id == Job.id)
            .where(ResearchOperation.status == 'pending', Job.status == 'running',
                JobAttempt.status == 'running', JobAttempt.worker_id != worker_id).distinct()).all()
    return [id for id in attempts if interrupt_expired_attempt(engine, id, worker_stopped_confirmed=True)]
