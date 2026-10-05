import { useState } from 'react'
import { Modal, CloseIcon } from './Modal'
import { HistoryPanel } from './HistoryPanel'

function NavIcon({ name }: { name: 'new' | 'history' | 'answer' | 'trail' | 'toggle' }) {
  const paths = { new: 'M12 5v14M5 12h14', history: 'M5 5H2v-3M2 5a10 10 0 1 1-1 9m11-9v7l4 2', answer: 'M5 3h14v18H5zM8 7h8M8 11h8M8 15h5', trail: 'M5 5h14M5 12h14M5 19h14M5 5v14', toggle: 'M4 4h16v16H4zM9 4v16m6-7-3 3 3 3' }
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d={paths[name]} /></svg>
}

export function WorkspaceNav({ expanded, onExpanded, historyView, onHistory, onHistoryNavigate, page, onPage, currentId, historyRevision, onRenamed }: {
  expanded: boolean; onExpanded: (value: boolean) => void; historyView: boolean; page: 'search' | 'qa'
  onHistory: (value: boolean) => void
  onHistoryNavigate: (href: string) => void
  onPage: (value: 'search' | 'qa') => void
  currentId?: string; historyRevision?: string; onRenamed?: (id: string, title: string) => void
}) {
  const [mobileOpen, setMobileOpen] = useState(false)
  const links = <>
    <a href="/" className="nav-item" aria-label="新研究" title="新研究"><NavIcon name="new" /><span>新研究</span></a>
    <button className="nav-item" aria-label="历史记录" title="历史记录" aria-expanded={historyView} aria-controls="sidebar-history" aria-pressed={historyView} onClick={() => onHistory(!historyView)}><NavIcon name="history" /><span>历史记录</span></button>
    <div className="nav-research"><span className="nav-caption">研究</span>
      <button className="nav-item" aria-label="检索论文" title="检索论文" aria-current={page === 'search' ? 'page' : undefined} onClick={() => { onPage('search'); setMobileOpen(false) }}><NavIcon name="trail" /><span>检索论文</span></button>
      <button className="nav-item" aria-label="论文问答" title="论文问答" aria-current={page === 'qa' ? 'page' : undefined} onClick={() => { onPage('qa'); setMobileOpen(false) }}><NavIcon name="answer" /><span>论文问答</span></button>
    </div>
  </>
  return <>
    <aside className={`workspace-sidebar${expanded ? ' is-expanded' : ''}${historyView ? ' has-history' : ''}`} aria-label="PaperTrail 导航">
      <a href="/" className="sidebar-brand" aria-label="PaperTrail 首页" title="PaperTrail"><span className="brand-mark">P</span><span className="brand-name">PaperTrail<small>论文研究助手</small></span></a>
      <button className="nav-item sidebar-toggle" aria-label={expanded ? '收起侧栏' : '展开侧栏'} aria-expanded={expanded} title={expanded ? '收起侧栏' : '展开侧栏'} onClick={() => onExpanded(!expanded)}><NavIcon name="toggle" /><span>收起侧栏</span></button>
      <nav aria-label="研究导航">{links}</nav>
      {historyView && <div className="sidebar-history" id="sidebar-history"><HistoryPanel onNavigate={onHistoryNavigate} currentId={currentId} revision={historyRevision} onRenamed={onRenamed} /></div>}
      <div className="sidebar-foot"><span className="sidebar-dot" />{expanded && <span>本机研究</span>}</div>
    </aside>
    <button className="mobile-menu secondary" aria-label="打开导航" aria-haspopup="dialog" onClick={() => setMobileOpen(true)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6h16M4 12h16M4 18h16" /></svg><span>PaperTrail</span></button>
    {mobileOpen && <Modal className="mobile-navigation" labelledBy="mobile-nav-title" onClose={() => setMobileOpen(false)}>
      <div className="mobile-nav-head"><div className="brand"><span className="brand-mark">P</span><h2 id="mobile-nav-title">PaperTrail</h2></div><button className="icon-button secondary" aria-label="关闭导航" onClick={() => setMobileOpen(false)}><CloseIcon /></button></div>
      <p className="nav-description">论文研究助手</p><nav aria-label="移动研究导航">{links}</nav>
    </Modal>}
  </>
}
