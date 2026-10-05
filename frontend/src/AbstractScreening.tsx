import { useEffect, useRef, useState } from 'react'
import { ApiFailure, type Candidate, type SearchState } from './api'
import { PaperDetail } from './PaperDetail'
import { arxivPdfUrl } from './research-flow'
import { ExternalIcon } from './Modal'
import {
  candidatePage, relevanceLabels, screenedCandidates, screeningMatches, screeningPlanMatches,
  type AbstractScreeningResult, type ScreeningView, type ScreeningPlan,
} from './screening'

async function responseJson(response: Response) {
  let value
  try {
    value = await response.json()
  } catch {
    throw new ApiFailure('SCREENING_INVALID_RESPONSE', '筛选服务返回异常，原搜索结果仍可查看。', response.status)
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new ApiFailure('SCREENING_INVALID_RESPONSE', '筛选服务返回异常，原搜索结果仍可查看。', response.status)
  }
  if (!response.ok) throw new ApiFailure(typeof value.code === 'string' ? value.code : 'SCREENING_FAILED', typeof value.message === 'string' ? value.message : '摘要筛选未完成，原结果保留。', response.status)
  return value
}

function failureMessage(error: unknown, fallback: string) {
  if (error instanceof TypeError) return '暂时无法连接筛选服务，原搜索结果仍可查看。'
  return error instanceof Error ? error.message : fallback
}

export function AbstractScreening({ search, draftQuestion, selectedVersions, savedVersions, disabled, onSelect, onContinue }: {
  search: SearchState; draftQuestion: string; selectedVersions: string[]; disabled: boolean
  onSelect: (versionId: string, checked: boolean) => void
  savedVersions?: string[]; onContinue: () => void
}) {
  const [saved, setSaved] = useState<AbstractScreeningResult | null>(null)
  const [plan, setPlan] = useState<ScreeningPlan | null>(null)
  const [loading, setLoading] = useState(true)
  const [pending, setPending] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [message, setMessage] = useState('')
  const [view, setView] = useState<ScreeningView>('all')
  const [page, setPage] = useState(1)
  const [detailId, setDetailId] = useState<string | null>(null)
  const listHeading = useRef<HTMLHeadingElement>(null)
  const identity = `${search.search_id}:${search.snapshot_sha256}:${search.question}`
  const current = useRef(identity)
  current.current = identity
  const controller = useRef<AbortController | null>(null)
  const path = `/api/searches/${encodeURIComponent(search.search_id)}/screening`

  const readSnapshot = async (signal: AbortSignal) => {
    const value = await responseJson(await fetch(path, { headers: { Accept: 'application/json' }, signal })) as {
      screening: AbstractScreeningResult | null; screening_plan: ScreeningPlan
    }
    if (value.screening && !screeningMatches(search, value.screening)) throw new Error('保存的筛选与当前候选不一致，请使用原搜索结果。')
    if (!screeningPlanMatches(search.candidates?.length || 0, value.screening_plan) ||
      value.screening_plan.verified_count !== (value.screening?.papers.length || 0)) throw new Error('筛选计划无法核对，请刷新后重试。')
    return value
  }

  useEffect(() => {
    let disposed = false
    const request = new AbortController()
    setLoading(true); setPending(false); setSaved(null); setPlan(null); setMessage(''); setView('all'); setPage(1); setDetailId(null)
    void readSnapshot(request.signal).then(value => {
        if (disposed || current.current !== identity) return
        setSaved(value.screening); setPlan(value.screening_plan)
      }).catch(error => {
        if (!disposed) setMessage(failureMessage(error, '暂时无法读取摘要筛选，原结果保留。'))
      }).finally(() => { if (!disposed) setLoading(false) })
    return () => { disposed = true; request.abort(); controller.current?.abort(); controller.current = null }
  }, [identity, path])

  useEffect(() => {
    if (!pending) return
    const timer = window.setInterval(() => setElapsed(s => s + 1), 1000)
    return () => window.clearInterval(timer)
  }, [pending])

  useEffect(() => {
    if (!pending) return
    const progress = new AbortController()
    let inFlight = false
    const timer = window.setInterval(() => {
      if (inFlight) return
      inFlight = true
      void readSnapshot(progress.signal).then(value => {
        if (!progress.signal.aborted && current.current === identity && controller.current && !controller.current.signal.aborted) {
          // Progress is read-only. Hold the displayed rows until the POST ends.
          setPlan(value.screening_plan)
        }
      }).catch(() => { /* The POST reports failures; a missed progress GET changes no rows. */ })
        .finally(() => { inFlight = false })
    }, 3000)
    return () => { window.clearInterval(timer); progress.abort() }
  }, [pending, identity, path])

  const generate = async () => {
    if (pending || controller.current || loading || !plan || plan.remaining_count === 0) return
    const source = identity
    const request = new AbortController()
    controller.current = request
    setPending(true); setElapsed(0); setMessage('')
    const timer = window.setTimeout(() => request.abort(), plan.wait_timeout_ms)
    try {
      const value = await responseJson(await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}', signal: request.signal })) as AbstractScreeningResult
      if (current.current !== source || request.signal.aborted || controller.current !== request) return
      if (!screeningMatches(search, value)) throw new Error('筛选与当前候选不一致，请使用原搜索结果。')
      setSaved(value); setView('all'); setPage(1)
    } catch (error) {
      if (current.current === source && controller.current === request) setMessage(request.signal.aborted
        ? '本次等待结束，已完成结果保留。可继续筛选未核实论文。'
        : failureMessage(error, '摘要筛选未完成，原结果保留。'))
    } finally {
      window.clearTimeout(timer)
      if (current.current === source && controller.current === request) {
        const latest = new AbortController()
        controller.current = latest
        const limit = window.setTimeout(() => latest.abort(), 10000)
        try {
          const value = await readSnapshot(latest.signal)
          if (current.current === source && controller.current === latest) {
            setSaved(value.screening); setPlan(value.screening_plan); setView('all'); setPage(1)
          }
        } catch {
          if (current.current === source && controller.current === latest) {
            setPlan(null); setMessage('已保留收到的结果，筛选进度暂时无法读取。请刷新后继续。')
          }
        } finally {
          window.clearTimeout(limit)
          if (controller.current === latest) { setPending(false); controller.current = null }
        }
      }
    }
  }

  const result = saved && screeningMatches(search, saved) ? saved : null
  const candidates = search.candidates || []
  const displayed = screenedCandidates(candidates, result, view, selectedVersions)
  const assessments = new Map(result?.papers.map(p => [p.paper_id, p]) || [])
  const unverified = new Map(result?.unverified_papers?.map(p => [p.paper_id, p]) || [])
  const high = result?.papers.filter(p => p.label === 'high').length || 0
  const partial = result?.papers.filter(p => p.label === 'partial').length || 0
  const remaining = candidates.length - (result?.papers.length || 0)
  const unscreened = remaining - (result?.unverified_papers?.length || 0)
  const pagination = candidatePage(displayed, page)
  const detail = candidates.find(candidate => candidate.versioned_id === detailId)
  const originsFor = (candidate: Candidate) => search.fusion?.ranking.find(row => row.arxiv_id === candidate.arxiv_id)?.matches
    .map(origin => `查询 ${origin.query_index} 第 ${origin.rank} 名`).join(' · ') || `arXiv 第 ${candidate.display_rank} 名`
  const openDetail = (candidate: Candidate, article?: HTMLElement) => {
    article?.querySelector<HTMLButtonElement>('.candidate-title')?.focus({ preventScroll: true })
    setDetailId(candidate.versioned_id)
  }
  const selectionSaved = !!savedVersions && savedVersions.length === selectedVersions.length
    && savedVersions.every(id => selectedVersions.includes(id))
  const chooseView = (next: ScreeningView) => { setView(next); setPage(1) }
  const turnPage = (next: number) => {
    setPage(next)
    requestAnimationFrame(() => {
      listHeading.current?.scrollIntoView({ block: 'start' })
      listHeading.current?.focus({ preventScroll: true })
    })
  }

  return <div className="screened-search">
    <section className="abstract-screening" aria-label="候选文献与摘要筛选">
      <div className="screening-heading"><div><h2 ref={listHeading} tabIndex={-1} id="candidate-list-heading">{view === 'selected' ? '已选文献' : '候选文献'} <span>{view === 'selected' ? displayed.length : candidates.length}</span></h2>
        {view !== 'selected' && result && <p role="status">已核对 {result.papers.length}/{candidates.length} 篇 · {high} 篇优先阅读{remaining > 0 ? ` · ${remaining} 篇待核对` : ''}</p>}</div>
        {(pending || remaining > 0) && view !== 'selected' && <div className="screening-action"><button className="secondary" disabled={disabled || loading || pending || !plan} onClick={() => void generate()}>{pending ? '正在按摘要核对…' : loading ? '读取筛选中…' : result ? unscreened > 0 ? `继续筛选剩余 ${remaining} 篇` : `重试未核实 ${remaining} 篇` : '按摘要筛选'}</button><span>调用模型 · {result ? '只处理待核对论文' : `全部 ${candidates.length} 篇`}</span></div>}
        {result && view !== 'selected' && <details className="screening-record"><summary>筛选记录</summary>
          <p><strong>研究问题</strong>：{search.question}</p><p>{result.interpretation}</p>
          <ul>{result.criteria.map((criterion, i) => <li key={i}>{criterion}</li>)}</ul>
          <p>{high} 篇优先阅读，{partial} 篇可能相关；未筛选的候选仍可手动选择。</p>
          {!!result.unverified_papers?.length && <ul className="screening-unverified">{result.unverified_papers.map(item => <li key={item.paper_id}><strong>{candidates.find(p => p.versioned_id === item.paper_id)?.title || item.paper_id}</strong>：{item.message}</li>)}</ul>}
          <p className="section-note">{new Date(result.created_at).toLocaleString()} 保存 · {result.model} · {(result.elapsed_ms / 1000).toFixed(1)} 秒{typeof result.usage?.total_tokens === 'number' ? ` · ${result.usage.total_tokens} token` : ''} · {result.prompt_version}</p>
          <p className="section-note">列表按阅读建议排列，同档保留检索顺序；这不是准确相关性的评分。</p>
        </details>}
      </div>
      {draftQuestion !== search.question && <p className="section-note">上方问题已修改。这批候选与筛选仍对应原问题；要研究新问题，请重新搜索。</p>}
      {pending && <p className="screening-message" role="status">已核对 {plan?.verified_count || 0}/{candidates.length} 篇 · {plan?.remaining_count ?? remaining} 篇待处理 · 已等待 {elapsed} 秒。每批最多 10 篇，等待上限 60 秒。</p>}
      {message && <p className="screening-message" role="status">{message}</p>}
      <div className="candidate-toolbar">
        <div className="screening-views" role="group" aria-label="候选展示方式">
          <button className="secondary" aria-pressed={view === 'all'} onClick={() => chooseView('all')}>全部候选 <span>{candidates.length}</span></button>
          <button className="secondary" aria-pressed={view === 'selected'} onClick={() => chooseView('selected')}>已选清单 <span>{selectedVersions.length}</span></button>
        </div>
        <span className="candidate-range" role="status">{pagination.start}–{pagination.end} / {displayed.length} 篇</span>
      </div>
    </section>
    <p className="sr-only" id="selection-rule">最多选入 10 个固定版本。点击准备并进入问答后，统一导入并建立索引。</p>
    {view === 'selected' && displayed.length === 0 && <p className="empty">还没有选入论文。切换到全部候选，勾选要阅读的固定版本。</p>}
    <div className={`candidates${view === 'selected' ? ' selected-papers' : ''}`} aria-labelledby="candidate-list-heading">
      {pagination.items.map((candidate, index) => {
        const assessment = assessments.get(candidate.versioned_id)
        const chosen = candidate.versions.some(version => selectedVersions.includes(version.versioned_id))
        const selected = candidate.versions.filter(version => selectedVersions.includes(version.versioned_id))
        const pdf = arxivPdfUrl(selected[0]?.versioned_id || candidate.versioned_id)
        return <article className={`candidate${chosen ? ' is-selected' : ''}`} key={candidate.versioned_id} aria-labelledby={`paper-${candidate.versioned_id}`} onClick={event => {
          if ((event.target as HTMLElement).closest('button, input, label, a, fieldset')) return
          openDetail(candidate, event.currentTarget)
        }}>
          <span className="candidate-number" aria-label={`第 ${pagination.start + index} 篇`}>{pagination.start + index}</span>
          <div className="candidate-content"><h3 id={`paper-${candidate.versioned_id}`}><button className="candidate-title" aria-label={`查看论文详情：${candidate.title}`} onClick={() => openDetail(candidate)}>{candidate.title}</button></h3>
            <div className="candidate-meta-line">
              {assessment && <span className={`relevance-label relevance-${assessment.label}`} title="已保存的摘要筛选建议">{relevanceLabels[assessment.label]}</span>}
              {unverified.has(candidate.versioned_id) && <span className="relevance-label relevance-insufficient" title={unverified.get(candidate.versioned_id)?.message}>依据未核实</span>}
              {!assessment && !unverified.has(candidate.versioned_id) && <span className="relevance-label relevance-unscreened">尚未筛选</span>}
              <p className="byline" title={candidate.authors.join(', ')}>{candidate.authors.slice(0, 2).join(', ') || '作者未知'}{candidate.authors.length > 2 && ` 等 ${candidate.authors.length} 位作者`}</p>
            </div>
            {assessment?.reason && <p className="reason-preview">{assessment.reason}</p>}
            <p className="abstract-preview">{candidate.abstract || '来源未提供摘要。'}</p>
          </div>
          <fieldset className="version-list"><legend className="sr-only">选择《{candidate.title}》的固定版本</legend>{(view === 'selected' ? selected : candidate.versions).map(version => {
            const checked = selectedVersions.includes(version.versioned_id)
            return <label key={version.versioned_id} title={version.versioned_id} className={`version-option${checked ? ' is-selected' : ''}`}><input type="checkbox" aria-label={`${candidate.title}，固定版本 ${version.versioned_id}`} aria-describedby="selection-rule" disabled={disabled || !checked && selectedVersions.length >= 10} checked={checked} onChange={event => onSelect(version.versioned_id, event.target.checked)} /><span>{candidate.versions.length > 1 ? `v${version.version}` : checked ? '已选' : '选入'}</span></label>
          })}</fieldset>
          {pdf && <a className="selected-pdf" href={pdf} target="_blank" rel="noopener noreferrer" aria-label={`阅读 arXiv PDF，固定版本 ${selected[0]?.versioned_id || candidate.versioned_id}`}>阅读 PDF<ExternalIcon /></a>}
        </article>
      })}
    </div>
    {pagination.pages > 1 && <nav className="candidate-pagination" aria-label="候选分页">
      <button className="secondary" disabled={pagination.page === 1} onClick={() => turnPage(pagination.page - 1)}>上一页</button>
      <span role="status">第 {pagination.page} / {pagination.pages} 页</span>
      <button className="secondary" disabled={pagination.page === pagination.pages} onClick={() => turnPage(pagination.page + 1)}>下一页</button>
    </nav>}
    <div className={`selection-dock${selectedVersions.length === 0 ? ' is-empty' : ''}`}>
      <div className="selection-summary" role="status"><strong>{selectedVersions.length}<small> / 10</small></strong><span>已选版本<small>{selectedVersions.length >= 10 ? '已达选择上限' : selectionSaved ? '与已保存清单一致' : '下一步统一导入'}</small></span></div>
      <button className="selection-next" disabled={disabled || selectedVersions.length < 1 || selectedVersions.length > 10} onClick={onContinue}><span>准备并进入问答</span><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10h12m-5-5 5 5-5 5" /></svg></button>
    </div>
    {detail && <PaperDetail candidate={detail} assessment={assessments.get(detail.versioned_id)} unverified={unverified.get(detail.versioned_id)?.message} origins={originsFor(detail)} selectedVersions={selectedVersions} disabled={disabled} onSelect={onSelect} onClose={() => setDetailId(null)} />}
  </div>
}
