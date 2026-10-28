import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiFailure, type AnswerState, type JobState, type JobStatus, type PdfPageLocation, type PdfPageLookup, type ResearchState, type SearchState, describeSavedContextCoverage, mutateApi, pendingSubmissions, readApi, submitApi } from './api'
import { recoverPending } from './recovery'
import { CitationAnswer, CitationDetails } from './CitationAnswer'
import { EvidencePanel } from './EvidencePanel'
import { citationReturnId, splitCitationText } from './citations'
import { RuntimeNotice, useRuntime } from './RuntimeNotice'
import { QueryAssistant, type QuerySuggestion } from './QueryAssistant'
import { QueryExpansion } from './QueryExpansion'
import { SearchFusion } from './SearchFusion'
import { SearchFailureNotice } from './SearchFailureNotice'
import { normalizedKeywords, type ExpansionSuggestion } from './query-plan'
import { historyHref } from './history-navigation'
import { AbstractScreening } from './AbstractScreening'
import { changeVersionSelection } from './screening'
import { WorkspaceNav } from './WorkspaceNav'
import { QuestionComposer } from './QuestionComposer'
import { initialAnswerSelection, changeAnswerSelection } from './research-state'

type Route = { research: string; search: string; collection: string; revision: string; answer: string; evidence: string; view: 'search' | 'qa' }
const labels: Record<JobStatus, string> = { queued: '排队中', running: '执行中', succeeded: '已完成', failed: '失败', interrupted: '已中断' }
const outcomes: Record<string, string> = { reported_sufficient: '上游报告可结束（尚未经人工充分性核验）', insufficient: '上游认为证据不足', truncated: '问答被截断，答案可能不完整', no_answer: '没有生成答案', scope_violation: '超出冻结文献范围', unknown: '未知' }
function routeFromUrl(): Route {
  const p = new URLSearchParams(location.search)
  return { research: p.get('research') || '', search: p.get('search') || '', collection: p.get('collection') || '', revision: p.get('revision') || '', answer: p.get('answer') || '', evidence: p.get('evidence') || '', view: p.get('view') === 'qa' || !p.has('view') && (p.has('answer') || p.has('revision')) ? 'qa' : 'search' }
}

function showCitation(id: string, block: ScrollLogicalPosition = 'start') {
  const target = document.getElementById(id)
  for (let parent = target?.parentElement; parent; parent = parent.parentElement) {
    if (parent instanceof HTMLDetailsElement && parent.classList.contains('answer-reading-fold')) parent.open = true
  }
  target?.scrollIntoView({ block }); target?.focus({ preventScroll: true })
}

export function ResearchApp() {
  const { runtime, disconnected, ready: workerReady } = useRuntime()
  const [route, setRoute] = useState<Route>(routeFromUrl)
  const [historyView, setHistoryView] = useState(() => new URLSearchParams(location.search).get('history') === '1')
  const [navExpanded, setNavExpanded] = useState(historyView)
  const [research, setResearch] = useState<ResearchState | null>(null)
  const [search, setSearch] = useState<SearchState | null>(null)
  const [answer, setAnswer] = useState<AnswerState | null>(null)
  const [jobs, setJobs] = useState<Record<string, JobState>>({})
  const [question, setQuestion] = useState('')
  const [queryText, setQueryText] = useState('')
  const [querySuggestion, setQuerySuggestion] = useState<QuerySuggestion | null>(null)
  const [queryExpansion, setQueryExpansion] = useState<ExpansionSuggestion | null>(null)
  const [answerQuestion, setAnswerQuestion] = useState('')
  const [selectedVersions, setSelectedVersions] = useState<string[]>([])
  const [answerIds, setAnswerIds] = useState<string[]>([])
  const [composerOpen, setComposerOpen] = useState(false)
  const [evidenceOpen, setEvidenceOpen] = useState(location.hash === '#evidence')
  const [returnCitationId, setReturnCitationId] = useState('')
  const [renderedCitations, setRenderedCitations] = useState<{ answerId: string; ids: ReadonlySet<string> }>({ answerId: '', ids: new Set() })
  const [pageLookup, setPageLookup] = useState<{ key: string; value: PdfPageLookup } | null>(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [now, setNow] = useState(Date.now())
  const latestRefresh = useRef(0)
  const observedStudy = useRef('')
  const observedAnswer = useRef('')
  const answerReading = useRef<HTMLElement>(null)

  const navigate = (href: string) => {
    ++latestRefresh.current
    const previous = routeFromUrl()
    history.pushState(null, '', href)
    const next = routeFromUrl()
    if (next.research !== previous.research || next.answer !== previous.answer || next.search !== previous.search) {
      setResearch(null); setSearch(null); setAnswer(null); setJobs({}); setError(''); setNotice('')
      observedStudy.current = ''; observedAnswer.current = ''
    }
    setRoute(next); setReturnCitationId(''); setEvidenceOpen(location.hash === '#evidence')
    const open = new URLSearchParams(location.search).get('history') === '1'
    setHistoryView(open); if (open) setNavExpanded(true)
  }
  const goResearch = (id: string, view: Route['view']) => {
    navigate(historyHref(`/?research=${encodeURIComponent(id)}&view=${view}`, historyView))
  }
  const refresh = useCallback(async () => {
    const serial = ++latestRefresh.current
    try {
      let id = route.research
      if (!id && (route.search || route.collection || route.revision || route.answer)) {
        const p = new URLSearchParams()
        for (const k of ['search', 'collection', 'revision', 'answer'] as const) if (route[k]) p.set(k, route[k])
        id = (await readApi<{ research_id: string }>(`/api/researches/resolve?${p}`)).research_id
      }
      const record = id ? await readApi<ResearchState>(`/api/researches/${id}`) : null
      const readings = await Promise.allSettled([
        record?.search_id ? readApi<SearchState>(`/api/searches/${record.search_id}`) : Promise.resolve(null),
        record?.answer_run_id ? readApi<AnswerState>(`/api/answers/${record.answer_run_id}`) : Promise.resolve(null),
      ])
      if (serial !== latestRefresh.current) return
      const [s, a] = readings.map(r => r.status === 'fulfilled' ? r.value : null) as [SearchState | null, AnswerState | null]
      setResearch(record); setSearch(s); setAnswer(a)
      const marker = record ? `${record.research_id}:${record.pool_revision_id || ''}` : 'new'
      if (observedStudy.current !== marker) {
        observedStudy.current = marker
        setQuestion(s?.question || ''); setQueryText(s?.query_text || '')
        setQuerySuggestion(null); setQueryExpansion(null)
        setSelectedVersions(record?.selected_versions || [])
        setAnswerIds(initialAnswerSelection(record?.papers || [], record?.question || record?.answer_run_id ? record.answer_document_ids : null))
        setAnswerQuestion(record?.question || s?.question || '')
        setComposerOpen(!a?.answer)
      }
      if (a?.answer && observedAnswer.current !== a.answer_run_id) { observedAnswer.current = a.answer_run_id; setComposerOpen(false) }
      const failure = readings.find(r => r.status === 'rejected')
      if (failure?.status === 'rejected') throw failure.reason
      const jobIds = [...new Set([s?.job_id, a?.job_id, record?.operation?.job_id, record?.preparation?.index_job_id].filter((id): id is string => !!id))]
      const results = await Promise.allSettled(jobIds.map(id => readApi<JobState>(`/api/jobs/${id}`)))
      if (serial === latestRefresh.current) setJobs(Object.fromEntries(results.flatMap(r => r.status === 'fulfilled' ? [[r.value.job_id, r.value]] : [])))
    } catch (cause) {
      if (serial === latestRefresh.current) setError(cause instanceof ApiFailure && cause.status === 404 ? '找不到这份保存记录。可以从历史记录重开，或开始新研究。' : cause instanceof Error ? cause.message : '读取失败')
    }
  }, [route.research, route.search, route.collection, route.revision, route.answer])
  useEffect(() => { void refresh() }, [refresh])
  const active = research?.operation?.status === 'pending' || [search?.status, answer?.status].some(s => s === 'queued' || s === 'running')
  useEffect(() => {
    if (!active) return
    const timer = window.setInterval(() => { setNow(Date.now()); void refresh() }, 1500)
    return () => window.clearInterval(timer)
  }, [active, refresh])
  useEffect(() => {
    const restore = () => { ++latestRefresh.current; setRoute(routeFromUrl()); setReturnCitationId(''); setEvidenceOpen(location.hash === '#evidence'); const open = new URLSearchParams(location.search).get('history') === '1'; setHistoryView(open); if (open) setNavExpanded(true) }
    window.addEventListener('popstate', restore)
    return () => window.removeEventListener('popstate', restore)
  }, [])
  const perform = async (name: string, action: () => Promise<void>) => {
    if (busy) return
    setBusy(name); setError(''); setNotice('')
    try { await action() }
    catch (cause) { setError(cause instanceof Error ? cause.message : '操作失败'); if (pendingSubmissions().length) setNotice('提交结果未确认。可点击“恢复未确认提交”，使用原请求和原幂等键继续。') }
    finally { setBusy('') }
  }
  const submitSearch = () => perform('search', async () => {
    if (queryExpansion && (queryExpansion.question !== question || queryExpansion.query_text !== normalizedKeywords(queryText))) throw new Error('问题或检索词已修改，请重新采用扩展或选择单查询。')
    const result = await submitApi<{ research_id: string }>('search', '/api/searches', { question, query_text: queryText, ...(queryExpansion ? { query_suggestion_id: queryExpansion.suggestion_id, query_plan: queryExpansion.query_plan } : querySuggestion?.question === question ? { query_suggestion_id: querySuggestion.suggestion_id } : {}) })
    goResearch(result.research_id, 'search')
  })
  const prepare = (versions = selectedVersions) => perform('prepare', async () => {
    if (!research || active || !versions.length) return
    const result = await submitApi<ResearchState>('prepare', `/api/researches/${research.research_id}/prepare`, { versioned_ids: versions, expected_pool_revision_id: research.pool_revision_id })
    goResearch(result.research_id, 'qa'); setResearch(result)
  })
  const submitAnswer = () => perform('research-answer', async () => {
    if (!research?.pool_revision_id || active || !answerIds.length || !answerQuestion.trim()) return
    const result = await submitApi<ResearchState>('research-answer', `/api/researches/${research.research_id}/answers`, { question: answerQuestion, document_ids: answerIds, expected_pool_revision_id: research.pool_revision_id })
    goResearch(result.research_id, 'qa'); setResearch(result); setComposerOpen(false)
  })
  const retryResearch = () => perform('research-retry', async () => {
    if (!research?.operation || research.operation.status !== 'failed') return
    const result = await submitApi<ResearchState>('research-retry', `/api/researches/${research.research_id}/retry`, { operation_id: research.operation.operation_id })
    goResearch(result.research_id, 'qa'); setResearch(result)
  })
  const retrySearch = (jobId: string) => perform('retry-search', async () => {
    const job = jobs[jobId]
    if (!job?.active_attempt_id) return
    await mutateApi(`/api/jobs/${jobId}/retry`, { expected_attempt_id: job.active_attempt_id }, 'POST'); await refresh()
  })
  const status = (id: string | null | undefined, fallback?: JobStatus | null) => {
    const job = id ? jobs[id] : null, state = job?.status || fallback
    if (!state) return null
    const from = state === 'running' ? job?.attempt?.started_at : job?.updated_at
    const seconds = from ? Math.max(0, Math.floor((now - Date.parse(from)) / 1000)) : null
    return <span className={`status ${state}`}>{labels[state]}{(state === 'running' || state === 'queued') && seconds !== null ? ` · ${seconds} 秒` : ''}</span>
  }
  const parts = splitCitationText(answer?.answer?.raw_answer || '', answer?.answer?.citations || [], new Set(answer?.answer?.evidence.map(e => e.evidence_id)))
  const numbers = new Map(parts.flatMap(p => p.kind === 'link' ? [[p.evidenceId, p.number] as const] : []))
  const selectedEvidenceId = route.evidence && numbers.has(route.evidence) ? route.evidence : ''
  const returnId = citationReturnId(parts, selectedEvidenceId, location.hash, returnCitationId || new URLSearchParams(location.search).get('citation') || '',
    renderedCitations.answerId === answer?.answer_run_id ? renderedCitations.ids : new Set())
  useEffect(() => {
    // A verified raw occurrence inside code or math remains literal text. Only
    // rendered prose anchors can be targets for the evidence return link.
    setRenderedCitations({ answerId: answer?.answer_run_id || '', ids: new Set(
      Array.from(answerReading.current?.querySelectorAll<HTMLElement>('[data-citation-anchor], a.citation-link[id]') || [], link => link.id)) })
  }, [answer?.answer_run_id, !!answer?.answer, route.view])
  const hasPdf = !!answer?.answer?.evidence.find(e => e.evidence_id === selectedEvidenceId)?.document_id
  const locationKey = `${answer?.answer_run_id || ''}:${selectedEvidenceId}`
  useEffect(() => {
    if (!evidenceOpen || !answer?.answer_run_id || !selectedEvidenceId || !hasPdf) return
    let current = true
    setPageLookup({ key: locationKey, value: { status: 'loading' } })
    void readApi<PdfPageLocation>(`/api/answers/${answer.answer_run_id}/evidence/${encodeURIComponent(selectedEvidenceId)}/location`).then(location => { if (current) setPageLookup({ key: locationKey, value: { status: 'ready', location } }) }).catch(() => { if (current) setPageLookup({ key: locationKey, value: { status: 'error' } }) })
    return () => { current = false }
  }, [evidenceOpen, locationKey, hasPdf])
  useEffect(() => {
    if (!route.evidence || !answer?.answer || route.view !== 'qa') return
    const frame = requestAnimationFrame(() => showCitation(location.hash === '#evidence' ? 'evidence' : returnId))
    return () => cancelAnimationFrame(frame)
  }, [route.evidence, answer?.answer_run_id, !!answer?.answer, route.view, returnId])
  const toggleHistory = (open: boolean) => { setHistoryView(open); if (open) setNavExpanded(true); history.replaceState(null, '', historyHref(location.href, open)) }
  const pending = pendingSubmissions()
  const poolReady = !!research?.preparation?.index_artifact_id
  const operation = research?.operation
  const failures = research?.import_report?.items.filter(item => item.status !== 'succeeded') || []
  const coverage = describeSavedContextCoverage(answer?.answer?.saved_context_coverage)
  return <div className={`app-shell workbench-shell${navExpanded ? ' nav-expanded' : ''}${historyView ? ' history-open' : ''}`}>
    <WorkspaceNav expanded={navExpanded} onExpanded={value => { setNavExpanded(value); if (!value) toggleHistory(false) }} historyView={historyView} onHistory={toggleHistory} onHistoryNavigate={navigate} page={route.view} currentId={research?.research_id} historyRevision={`${research?.updated_at}:${research?.answer_run_id}:${operation?.status}`} onRenamed={(id, title) => setResearch(old => old?.research_id === id ? { ...old, title, title_is_custom: true } : old)} onPage={view => {
      const p = new URLSearchParams(location.search); p.set('view', view); p.delete('evidence'); p.delete('citation'); navigate(`${location.pathname}?${p}`); window.scrollTo({ top: 0 })
    }} />
    <div className="workspace-content"><main className="workbench-main" onClick={event => {
      if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return
      const link = (event.target as HTMLElement).closest<HTMLAnchorElement>('a[data-evidence-id], a.return-citation')
      if (!link) return
      if (link.classList.contains('return-citation')) { event.preventDefault(); history.pushState(null, '', link.href); setEvidenceOpen(false); requestAnimationFrame(() => showCitation(link.hash.slice(1), 'center')); return }
      const id = link.dataset.evidenceId
      if (!id || !numbers.has(id)) return
      const origin = link.classList.contains('citation-link') ? link.dataset.citationId || link.id :
        answerReading.current?.querySelector<HTMLElement>(`[data-citation-anchor][data-evidence-id="${CSS.escape(id)}"], a.citation-link[id][data-evidence-id="${CSS.escape(id)}"]`)?.id || ''
      event.preventDefault(); setReturnCitationId(origin)
      setEvidenceOpen(true); history.pushState(null, '', link.href); setRoute(current => ({ ...current, evidence: id }))
      requestAnimationFrame(() => { const target = document.getElementById('evidence'); target?.scrollIntoView({ block: 'start' }); target?.focus({ preventScroll: true }) })
    }}>
      {error && <div className="alert error" role="alert">{error}<button className="text-button" onClick={() => { setError(''); void refresh() }}>重新读取</button></div>}
      {notice && <div className="alert notice" role="status">{notice}</div>}
      {pending.length > 0 && <div className="alert notice">{pending.length} 次提交尚未确认。<button className="secondary" disabled={!!busy} onClick={() => void perform('recover', () => recoverPending((_name, result) => { if (result.research_id) goResearch(result.research_id, _name === 'search' ? 'search' : 'qa'); else if (result.answer_run_id) navigate(`/?answer=${result.answer_run_id}`); else if (result.search_id) navigate(`/?search=${result.search_id}`); else if (result.collection_id) navigate(`/?collection=${result.collection_id}`) }, () => void refresh(), cause => setError(cause instanceof Error ? cause.message : '恢复失败')))}>恢复未确认提交</button></div>}
      {route.view === 'search' ? <>
        <div className={`intro research-intro${search ? ' has-search' : ''}`}><div><h1 tabIndex={-1}>{search?.question || '从问题出发，读懂论文。'}</h1>{!search && <p>检索文献，让回答有据可循。</p>}</div><RuntimeNotice runtime={runtime} disconnected={disconnected} compact /></div>
        <SearchFailureNotice search={search} />
        {search?.provenance?.fallback_used && <p className="section-note" role="status">arXiv 搜索暂不可用，已使用 OpenAlex 备用检索。以下论文的摘要和固定版本已从 arXiv 原文页核实；本次使用单路关键词搜索。</p>}
        <details className="search-composer" open={!search || search.status === 'failed' || search.status === 'interrupted'} key={`composer-${search?.search_id}`}>
          <summary><strong>{search ? '检索设置' : '开始你的研究'}</strong><span>{search ? search.query_text : '输入问题与英文检索词'}</span>{search && <span className="composer-edit">查看 / 修改</span>}</summary>
          <div className="search-form"><div className="form-grid"><label>研究问题（中文）<textarea value={question} onChange={e => setQuestion(e.target.value)} rows={2} placeholder="例如：智能体长期记忆如何评估？" /></label><label>arXiv 检索词（英文）<input value={queryText} onChange={e => setQueryText(e.target.value)} placeholder="LLM agent long-term memory evaluation" /></label></div>
            <QueryAssistant question={question} disabled={!!busy} onAdopt={s => { setQueryText(s.query_text); setQuerySuggestion(s); setQueryExpansion(null) }} />
            <QueryExpansion question={question} queryText={queryText} disabled={!!busy} adopted={queryExpansion} onAdopt={setQueryExpansion} />
            <div className="action-row"><button disabled={!!busy || !workerReady || !question.trim() || !queryText.trim() || !!queryExpansion && (queryExpansion.question !== question || queryExpansion.query_text !== normalizedKeywords(queryText))} onClick={() => void submitSearch()}>{queryExpansion ? '开始多查询搜索' : '开始搜索'}</button>{status(search?.job_id, search?.status)}{search && ['failed', 'interrupted'].includes(search.status) && <button className="text-button" disabled={!!busy || !workerReady} onClick={() => void retrySearch(search.job_id)}>重试检索</button>}</div>
          </div>
        </details>
        {search?.fusion && <details className="search-provenance"><summary>多查询检索记录<span>{search.fusion.successful_queries} 组成功{search.fusion.partial ? '，有查询失败' : ''}</span></summary><SearchFusion search={search} /></details>}
        {search && ['queued', 'running'].includes(search.status) && <div className="search-waiting" role="status">{status(search.job_id, search.status)}<p>结果会自动出现。</p></div>}
        {search?.status === 'succeeded' && search.candidates?.length === 0 && <p className="empty">没有找到候选，可以调整检索词再搜索。</p>}
        {search?.candidates && search.candidates.length > 0 && <AbstractScreening key={`${search.search_id}:${search.snapshot_sha256}`} search={search} draftQuestion={question} selectedVersions={selectedVersions} savedVersions={research?.selected_versions} disabled={!!busy || active || !workerReady} onSelect={(id, checked) => setSelectedVersions(previous => changeVersionSelection(previous, id, checked))} onContinue={() => void prepare()} />}
      </> : <>
        {!answer?.answer && <div className="intro research-intro"><div><h1 tabIndex={-1}>{research?.title || '把问题交给论文。'}</h1></div><RuntimeNotice runtime={runtime} disconnected={disconnected} compact /></div>}
        {!research && !error && <p className="empty" role="status">{route.research || route.answer || route.collection || route.search || route.revision ? '正在读取研究记录…' : '从检索论文中选择文献，再准备问答。'}</p>}
        {research && !poolReady && !active && operation?.status !== 'failed' && <p className="empty">尚未准备文献。<button className="text-button" onClick={() => goResearch(research.research_id, 'search')}>选择论文</button></p>}
        {operation?.status === 'pending' && <section className="preparation-progress" role="status"><span className="progress-spinner" aria-hidden="true" /><div><strong>{operation.kind === 'prepare' ? operation.phase === 'index' ? '正在建立论文索引' : '正在导入已选论文' : operation.phase === 'answer' ? '正在生成有证据的回答' : '正在准备本次回答范围'}</strong><p>{status(operation.job_id, 'queued')} · 完成后自动继续</p></div></section>}
        {operation?.status === 'failed' && <div className="alert error" role="alert">{operation.message || '这次任务未完成。'}<button className="secondary" disabled={!!busy || !workerReady} onClick={() => void retryResearch()}>重试{operation.kind === 'prepare' ? '准备' : '回答'}</button></div>}
        {!!research?.import_report && failures.length > 0 && <details className="preparation-report"><summary>{research.import_report.items.length - failures.length} 篇导入成功，{failures.length} 篇未完成</summary><ul>{failures.map(item => <li key={item.versioned_id}><strong>{item.title || item.versioned_id}</strong><span>{item.failure_reason || '导入失败'}</span></li>)}</ul>{operation?.status === 'succeeded' && <button className="secondary" disabled={!!busy || !workerReady} onClick={() => void prepare(research.selected_versions)}>重试未完成论文</button>}</details>}
        {answer?.answer && <header className="question-header"><div className="question-title"><h1 tabIndex={-1}>{answer.answer.question || '这份记录未保存原问题'}</h1><span className="answer-status">{answer.status === 'succeeded' ? '已保存' : labels[answer.status]}</span></div></header>}
        {poolReady && <details className={`qa-composer-disclosure${answer?.answer ? ' has-answer' : ''}`} open={composerOpen} onToggle={event => setComposerOpen(event.currentTarget.open)}>
          <summary><svg className="composer-toggle" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14" /><path className="composer-toggle-vertical" d="M12 5v14" /></svg><strong>{answer?.answer ? '调整问题或文献' : '问题与文献'}</strong><span>{research!.papers.length} 篇已准备</span></summary>
          <QuestionComposer papers={research!.papers} selected={answerIds} question={answerQuestion} disabled={!!busy || active} ready={workerReady && !active} onQuestion={setAnswerQuestion} onSelect={(id, checked) => setAnswerIds(previous => changeAnswerSelection(research!.papers, previous, id, checked))} onSubmit={() => void submitAnswer()} />
        </details>}
        {answer && !answer.answer && !active && <p className="empty">{['failed', 'interrupted'].includes(answer.status) ? '这次回答在结果保存前结束，旧链接保留此状态。可调整问题后重新生成。' : '正在读取保存结果。'}</p>}
        {answer?.answer && <><div className="answer-panel">
          {answer.answer.answer_outcome !== 'reported_sufficient' && <p className="answer-warning" role="status">{outcomes[answer.answer.answer_outcome] || answer.answer.answer_outcome}</p>}
          <section ref={answerReading} className="answer-reading" aria-labelledby="answer-text-heading"><CitationAnswer rawAnswer={answer.answer.raw_answer} displayAnswer={answer.answer.display_answer} citations={answer.answer.citations} evidenceIds={new Set(answer.answer.evidence.map(e => e.evidence_id))} evidence={answer.answer.evidence} documentSources={answer.answer.document_sources} pathname={location.pathname} search={location.search} selectedEvidenceId={selectedEvidenceId} hideDetails /></section>
          <EvidencePanel answerRunId={answer.answer_run_id} open={evidenceOpen} onOpenChange={setEvidenceOpen} evidence={answer.answer.evidence} selectedEvidenceId={selectedEvidenceId} citationNumbers={numbers} pathname={location.pathname} search={location.search} documentTitles={answer.answer.document_titles} documentSources={answer.answer.document_sources} returnCitationId={returnId} pageLookup={pageLookup?.key === locationKey ? pageLookup.value : undefined} />
        </div>
          <details className="technical-details"><summary>技术详情<span>运行状态、来源与原始引用记录</span></summary><p className="section-note">执行：{answer.answer.job_status} · 上游状态：{answer.answer.upstream_status}</p><p className="section-note">结论状态：{outcomes[answer.answer.answer_outcome] || answer.answer.answer_outcome}</p><p className="section-note">{coverage.summary}。进入这份上下文：{coverage.included.join('、') || '未记录'}。完整供应商请求不可见。</p><CitationDetails rawAnswer={answer.answer.raw_answer} displayAnswer={answer.answer.display_answer} citations={answer.answer.citations} evidenceIds={new Set(answer.answer.evidence.map(e => e.evidence_id))} pathname={location.pathname} search={location.search} /><p className="usage">Embedding 用量：{answer.answer.usage.embedding_tokens ?? '未记录'} · 费用：{answer.answer.usage.cost_usd ?? '未记录'}</p></details>
        </>}
      </>}
    </main><footer>PaperTrail · 本机研究 · 固定版本，保留依据</footer></div>
  </div>
}
