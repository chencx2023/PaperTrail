import { useEffect, useRef, useState } from 'react'
import { ApiFailure } from './api'
import { normalizedKeywords, validatedExpansionPlan, type ExpansionSuggestion, type KeywordQuery } from './query-plan'

export function QueryExpansion({ question, queryText, disabled, adopted, onAdopt }: {
  question: string; queryText: string; disabled: boolean; adopted: ExpansionSuggestion | null
  onAdopt: (value: ExpansionSuggestion | null) => void
}) {
  const [record, setRecord] = useState<ExpansionSuggestion | null>(null)
  const [draft, setDraft] = useState<KeywordQuery[]>([])
  const [pending, setPending] = useState(false)
  const [message, setMessage] = useState('')
  const baseline = normalizedKeywords(queryText)
  const identity = JSON.stringify([question, baseline])
  const current = useRef(identity)
  current.current = identity
  const controller = useRef<AbortController | null>(null)
  useEffect(() => () => controller.current?.abort(), [])
  const fresh = record?.question === question && record?.query_text === baseline
  const stale = adopted && (adopted.question !== question || adopted.query_text !== baseline)

  const generate = async () => {
    if (pending || !question.trim() || !baseline) return
    const source = identity
    const request = new AbortController()
    controller.current = request
    onAdopt(null)
    setPending(true); setMessage(''); setRecord(null); setDraft([])
    const timer = window.setTimeout(() => request.abort(), 25000)
    try {
      const response = await fetch('/api/query-expansions', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, query_text: baseline }), signal: request.signal })
      const result = await response.json()
      if (!response.ok) throw new ApiFailure(result.code, result.message || '生成未完成，可继续使用单查询。', response.status)
      if (current.current !== source) { setMessage('问题或原检索词已修改，请为当前输入重新生成扩展。'); return }
      if (result.question !== question || result.query_text !== baseline || result.query_plan?.baseline_query_text !== baseline ||
        !/^[0-9a-f]{32}$/.test(result.suggestion_id)) throw new Error('扩展与当前输入不一致，请重新生成。')
      const plan = validatedExpansionPlan(baseline, result.query_plan.expansions)
      setRecord({ ...result, query_plan: plan }); setDraft(plan.expansions)
    } catch (error) {
      setMessage(error instanceof ApiFailure || error instanceof Error && error.name !== 'AbortError'
        ? error.message : '生成未完成，未自动重试。原检索词保留，可继续单查询。')
    } finally { window.clearTimeout(timer); setPending(false); controller.current = null }
  }

  const edit = (index: number, groupIndex: number, value: string, exclusion = false) => {
    const phrases = value.split('|').map(p => p.trim())
    setDraft(items => items.map((item, i) => i !== index ? item : exclusion
      ? { ...item, exclude: phrases } : { ...item, groups: item.groups.map((g, j) => j === groupIndex ? phrases : g) }))
    onAdopt(null); setMessage('修改后请重新采用扩展，再开始搜索。')
  }
  const adopt = () => {
    if (!record || !fresh) return
    try {
      onAdopt({ ...record, query_plan: validatedExpansionPlan(baseline, draft) })
      setMessage('已采用扩展。点击下方开始搜索时才会检索论文。')
    } catch (error) { setMessage(error instanceof Error ? error.message : '请检查扩展表达。') }
  }

  return <details className="query-expansion">
    <summary>扩展查询（可选）<span>{adopted ? `已采用 ${adopted.query_plan.expansions.length + 1} 组` : '原查询 + 同义表达或研究角度'}</span></summary>
    <p className="section-note">保留你确认的原检索词，让模型补充最多两组。每组最多搜索 20 篇，再融合排名；顺序执行各组，等待时间可能增加。</p>
    <div className="assistant-action"><button className="secondary" disabled={disabled || pending || !question.trim() || !baseline} onClick={() => void generate()}>{pending ? '正在生成扩展…' : '生成扩展查询'}</button><span>调用一次模型，只生成建议；20 秒超时停止，不自动重试。</span></div>
    {stale && <p className="assistant-message" role="status">已采用的扩展对应旧输入。请为当前问题重新生成并采用，或选择“使用单查询”。</p>}
    {message && <p className="assistant-message" role="status">{message}</p>}
    {fresh && record && <div className="expansion-editor">
      <p className="expansion-baseline"><strong>查询 1 · 原查询</strong><span>{record.query_text}</span></p>
      <p className="section-note">同一行的表达用 | 分隔，任选一种匹配；不同行的概念需要同时匹配。多词表达按完整短语搜索。</p>
      {draft.map((query, index) => <fieldset key={index} className="expansion-query"><legend>查询 {index + 2} · {query.label}</legend>
        {query.groups.map((group, groupIndex) => <label key={groupIndex}>{groupIndex === 0 ? '研究对象（任选表达）' : `检索角度 ${groupIndex}（任选表达）`}<input disabled={disabled} value={group.join(' | ')} onChange={e => edit(index, groupIndex, e.target.value)} /></label>)}
        {query.exclude.length > 0 && <label>排除表达<input disabled={disabled} value={query.exclude.join(' | ')} onChange={e => edit(index, 0, e.target.value, true)} /></label>}
      </fieldset>)}
      <div className="action-row"><button disabled={disabled || pending} onClick={adopt}>采用多查询（{draft.length + 1} 组）</button></div>
      <p className="section-note">{record.model} · {typeof record.usage?.total_tokens === 'number' ? `${record.usage.total_tokens} token` : '用量未知'} · {record.cost_usd == null ? '费用未知' : `$${record.cost_usd}`}</p>
    </div>}
    {adopted && <button className="text-button" disabled={disabled} onClick={() => { onAdopt(null); setMessage('已恢复单查询，原检索词保留。') }}>使用单查询</button>}
  </details>
}
