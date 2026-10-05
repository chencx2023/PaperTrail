import { useEffect, useRef, useState } from 'react'
import { mutateApi, readApi } from './api'
import { historyHref } from './history-navigation'

type HistoryItem = { research_id: string; title: string; status: string; question: string | null; created_at: string; href: string }
type HistoryPage = { items: HistoryItem[]; next_cursor: string | null }
const statuses: Record<string, string> = { queued: '排队中', running: '执行中', pending: '处理中', succeeded: '已完成', failed: '失败', interrupted: '已中断', saved: '已保存' }

export function HistoryPanel({ onNavigate, currentId, revision, onRenamed }: {
  onNavigate: (href: string) => void; currentId?: string; revision?: string; onRenamed?: (id: string, title: string) => void
}) {
  const [cursor, setCursor] = useState(() => new URLSearchParams(location.search).get('cursor') || '')
  const [page, setPage] = useState<HistoryPage | null>(null)
  const [error, setError] = useState('')
  const [version, setVersion] = useState(0)
  const [editing, setEditing] = useState('')
  const [name, setName] = useState('')
  const [saving, setSaving] = useState(false)
  const list = useRef<HTMLDivElement>(null)
  const nameInput = useRef<HTMLInputElement>(null)
  const navigate = (nextCursor = '') => {
    setCursor(nextCursor); onNavigate(historyHref(location.href, true, { kind: 'all', cursor: nextCursor }))
    list.current?.scrollTo({ top: 0 })
  }
  useEffect(() => {
    const restore = () => setCursor(new URLSearchParams(location.search).get('cursor') || '')
    window.addEventListener('popstate', restore)
    return () => window.removeEventListener('popstate', restore)
  }, [])
  useEffect(() => {
    let current = true
    setError('')
    const query = new URLSearchParams({ limit: '12', ...(cursor ? { cursor } : {}) })
    void readApi<HistoryPage>(`/api/researches?${query}`).then(result => { if (current) setPage(result) }).catch(cause => { if (current) setError(cause instanceof Error ? cause.message : '读取历史失败') })
    return () => { current = false }
  }, [cursor, version, currentId, revision])
  useEffect(() => { if (editing) { nameInput.current?.focus(); nameInput.current?.select() } }, [editing])
  const saveName = async () => {
    if (saving || !name.trim()) return
    const id = editing
    setSaving(true); setError('')
    try {
      await mutateApi(`/api/researches/${id}`, { title: name.trim() }, 'PATCH')
      setPage(old => old ? { ...old, items: old.items.map(item => item.research_id === id ? { ...item, title: name.trim() } : item) } : old)
      onRenamed?.(id, name.trim()); setEditing('')
    } catch (cause) { setError(cause instanceof Error ? cause.message : '改名失败') }
    finally { setSaving(false) }
  }
  return <section className="history-panel" aria-labelledby="history-heading">
    <header className="history-heading"><h2 id="history-heading">研究记录</h2></header>
    <div className="history-list-scroll" ref={list} tabIndex={-1} aria-label="保存的研究记录">
      {error && <div className="alert error" role="alert">{error}<button className="text-button" onClick={() => setVersion(old => old + 1)}>重新读取</button></div>}
      {!page ? <p className="loading-state" role="status">正在读取研究记录…</p> : page.items.length === 0 ? <p className="empty">从新研究开始，记录会保存在这里。</p> : <div className="history-list">{page.items.map(item => <article className={`history-item${item.research_id === currentId ? ' is-current' : ''}`} key={item.research_id}>
        <div className="history-meta"><span>{item.question ? '论文问答' : '论文检索'}</span><span className={`history-status ${item.status}`}>{statuses[item.status] || '状态未知'}</span></div>
        {editing === item.research_id ? <form className="history-rename" onSubmit={event => { event.preventDefault(); void saveName() }}>
          <input ref={nameInput} aria-label="研究记录名称" maxLength={200} value={name} disabled={saving} onChange={event => setName(event.target.value)} onKeyDown={event => { if (event.key === 'Escape' && !saving) setEditing('') }} />
          <div><button type="submit" disabled={saving || !name.trim()}>保存</button><button type="button" className="secondary" disabled={saving} onClick={() => setEditing('')}>取消</button></div>
        </form> : <div className="history-title"><h3><a href={historyHref(item.href, true, { kind: 'all', cursor })} aria-current={item.research_id === currentId ? 'page' : undefined} onClick={event => { if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return; event.preventDefault(); onNavigate(event.currentTarget.href) }}>{item.title || '未命名研究'}</a></h3>
          <button className="history-rename-trigger" aria-label={`重命名：${item.title}`} title="重命名" onClick={() => { setEditing(item.research_id); setName(item.title) }}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m15 5 4 4M4 20l4-1L20 7a3 3 0 0 0-4-4L4 15z" /></svg></button>
        </div>}
        <time dateTime={item.created_at}>{new Date(item.created_at).toLocaleString('zh-CN', { dateStyle: 'medium', timeStyle: 'short' })}</time>
      </article>)}</div>}
      {page && <nav className="history-pagination" aria-label="历史分页">{cursor && <button className="secondary" onClick={() => navigate()}>返回最新</button>}{page.next_cursor && <button className="secondary" onClick={() => navigate(page.next_cursor!)}>更早的记录</button>}</nav>}
    </div><p className="history-storage">记录保存在本机</p>
  </section>
}
