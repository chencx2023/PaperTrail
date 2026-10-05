import { Fragment } from 'react'
import type { DocumentSource, Evidence, PdfPageLookup } from './api'
import { evidenceHref } from './citations'
import { paperGroups, paperCitationLabel, savedPaperSource } from './paper-sources'

type Props = {
  answerRunId: string
  evidence: Evidence[]
  selectedEvidenceId: string
  citationNumbers: ReadonlyMap<string, number>
  pathname?: string
  search?: string
  pageLookup?: PdfPageLookup
  documentTitles?: Record<string, string>
  documentSources?: Record<string, DocumentSource>
  returnCitationId?: string
  open?: boolean
  onOpenChange?: (open: boolean) => void
}

function pageRange(start: number, end: number) {
  return start === end ? `${start}` : `${start}–${end}`
}

function OriginalPassage({ text, expanded = false }: { text: string | null; expanded?: boolean }) {
  if (!text) return <p className="section-note">没有保存实际原文。</p>
  const characters = Array.from(text)
  if (characters.length <= 600) return <p className="original-preview">{text}</p>
  const prefix = characters.slice(0, 600).join('')
  const boundary = prefix.search(/\s+\S*$/)
  const preview = boundary > prefix.length - 40 ? prefix.slice(0, boundary) : prefix
  return <div className="original-reading">
    <details className="original-full" open={expanded || undefined}>
      <summary><span className="expand-label">展开完整片段</span>
        <span className="collapse-label">收起完整片段</span></summary>
      <div className="original-passage">{text.split(/\n\s*\n/).map((paragraph, index) =>
        <p key={index}>{paragraph}</p>)}</div>
    </details>
    <p className="original-preview">{preview}…</p>
  </div>
}

function EvidenceCard({ answerRunId, item, number, selected = false, pageLookup, title, returnCitationId, inline = false }: {
  answerRunId: string; item: Evidence; number?: number; selected?: boolean; pageLookup?: PdfPageLookup
  title?: string; returnCitationId?: string; inline?: boolean
}) {
  const pdfUrl = `/api/answers/${encodeURIComponent(answerRunId)}/evidence/${encodeURIComponent(item.evidence_id)}/pdf`
  const location = selected && pageLookup?.status === 'ready' &&
    pageLookup.location.evidence_id === item.evidence_id ? pageLookup.location : null
  const uniqueMatch = location?.status === 'unique' && location.matches.length === 1
    ? location.matches[0] : null
  const multipleMatches = location?.status === 'multiple' ? location.matches : []
  const pageText = uniqueMatch
    ? `PDF 物理第 ${pageRange(uniqueMatch.start_page, uniqueMatch.end_page)} 页（提取文字完整匹配）`
    : multipleMatches.length > 1 ? `PDF 多处匹配（${multipleMatches.length} 处）`
      : selected && pageLookup?.status === 'loading' ? '正在核对 PDF 物理页码'
        : selected && pageLookup?.status === 'error' ? 'PDF 物理页码暂不可核对'
          : item.pdf_page == null ? 'PDF 物理页码未知' : `PDF 物理第 ${item.pdf_page} 页`
  return <article className={`evidence${selected ? ' selected' : ''}`}
    id={selected ? 'evidence' : undefined} tabIndex={selected ? -1 : undefined}>
    <div className="evidence-title"><h4>{inline ? '引用原文' : number ? `引用 [${number}]` : '保存的证据片段'}</h4>
      {returnCitationId && <a className="return-citation" href={`#${returnCitationId}`}>返回引用处</a>}</div>
    {!inline && <p className="paper-title">{title || item.upstream_dockey}</p>}
    {inline && title && <p className="original-source-title">{title}</p>}
    <p className="section-note">固定版本 · {item.upstream_dockey}</p>
    {inline && <div className="source-section"><OriginalPassage text={item.original_text} expanded={selected} /></div>}
    <p className="evidence-location" aria-live={selected ? 'polite' : undefined}>{pageText}</p>
    <p className="pdf-link-row">{item.document_id
      ? <><a href={uniqueMatch ? `${pdfUrl}#page=${uniqueMatch.start_page}` : pdfUrl}
          target="_blank" rel="noopener noreferrer">
          打开固定版本 PDF（新标签页）</a>
          <a href={`${pdfUrl}?download=1`}>下载 PDF（本机查看）</a></>
      : <span>固定版本 PDF 无法定位</span>}</p>
    {multipleMatches.length > 1 && <p className="pdf-link-row">
      原文在 PDF 中有多处完整匹配，请选择候选：{' '}
      {multipleMatches.map((match, index) => <span key={`${index}-${match.start_page}-${match.end_page}`}>
        {index > 0 && ' · '}<a href={`${pdfUrl}#page=${match.start_page}`}
          target="_blank" rel="noopener noreferrer">
          第 {index + 1} 处：物理第 {pageRange(match.start_page, match.end_page)} 页</a>
      </span>)}
    </p>}
    {inline ? <details className="saved-summary"><summary>保存的证据摘要</summary><p className="evidence-summary">{item.summary || '没有保存摘要'}</p></details> : <><div className="summary-section"><h5>模型摘要</h5>
    <p className="section-note">已保存的自动整理内容，请结合原文核查。</p>
    <p className="evidence-summary">{item.summary || '没有保存摘要'}</p></div>
    <div className="source-section"><h5>论文原文</h5>
    <p className="section-note">完整展开的是本次保存的文本块；表格、图注和前后文请结合 PDF 阅读。</p>
    <OriginalPassage text={item.original_text} />
    </div></>}
    <details className="evidence-debug"><summary>查看定位信息与原始提取格式</summary>
      <p className="section-note">内部引用 ID：{item.upstream_context_id}</p>
      <p className="section-note">上游位置标签：{item.upstream_locator || '未知'}（未核准为 PDF 物理页码）</p>
      <pre className="saved-extract">{item.original_text || '原文未知'}</pre>
    </details>
  </article>
}

export function EvidencePanel({ answerRunId, evidence, selectedEvidenceId, citationNumbers,
  pathname = '/', search = `?answer=${encodeURIComponent(answerRunId)}`, pageLookup,
  documentTitles = {}, documentSources = {}, returnCitationId, open, onOpenChange }: Props) {
  const references = evidence.filter((item) => citationNumbers.has(item.evidence_id))
    .sort((a, b) => citationNumbers.get(a.evidence_id)! - citationNumbers.get(b.evidence_id)!)
  const otherEvidence = evidence.filter((item) => !citationNumbers.has(item.evidence_id) &&
    item.evidence_id !== selectedEvidenceId)
  const selected = evidence.find((item) => item.evidence_id === selectedEvidenceId)
  const papers = paperGroups(references)
  const identitiesKnown = references.every(item => !!item.document_id)
  return <details className="evidence-panel paper-sources" aria-label="引用论文与保存原文" open={open}
    onToggle={event => onOpenChange?.(event.currentTarget.open)}>
    <summary className="evidence-disclosure"><strong>{identitiesKnown ? `查看 ${papers.length} 篇引用论文` : `查看 ${references.length} 条引用记录`}</strong><span>{references.length} 个原文片段</span><svg viewBox="0 0 20 20" aria-hidden="true"><path d="m5 8 5 5 5-5" /></svg></summary>
    {papers.length > 0 && <table className="source-table"><colgroup><col className="source-paper-column" /><col /></colgroup><thead><tr><th scope="col">论文</th><th scope="col">答案引用的原文</th></tr></thead>
      <tbody>{papers.map(({key, fragments}, paperIndex) => {
        const item = fragments[0], source = savedPaperSource(item, documentSources)
        const current = fragments.find(fragment => fragment.evidence_id === selectedEvidenceId)
        const title = source?.title || documentTitles[item.document_id || ''] || item.upstream_dockey
        const pdf = `/api/answers/${encodeURIComponent(answerRunId)}/evidence/${encodeURIComponent(item.evidence_id)}/pdf`
        return <Fragment key={key}><tr className={`paper-source-row${current ? ' is-current' : ''}`}><td>
          <h3>{title}</h3><p className="source-author">{source?.authors.length ? paperCitationLabel(source, paperIndex + 1, documentSources) : '作者年份未保存'}</p>
          {item.document_id && <a className="source-pdf" href={pdf} target="_blank" rel="noopener noreferrer" title={`固定版本 ${item.upstream_dockey}`}>阅读 PDF <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3H3v10h10v-3M8 3h5v5M7 9l6-6" /></svg></a>}
          <details className="source-abstract"><summary>论文摘要</summary>{source?.authors.length ? <p className="source-full-authors">{source.authors.join(', ')}</p> : null}
            <p>{source?.abstract_status === 'saved' && source.abstract ? source.abstract : source?.abstract_status === 'unverified' ? '保存摘要的来源快照无法核对。' : '这份记录未保存可核对的论文摘要。'}</p>
          </details>
        </td><td>{fragments.map((fragment, index) => <a className={`source-preview${fragment.evidence_id === selectedEvidenceId ? ' is-current' : ''}`} key={fragment.evidence_id}
          data-evidence-id={fragment.evidence_id} href={evidenceHref(pathname, search, fragment.evidence_id)}
          aria-current={fragment.evidence_id === selectedEvidenceId ? 'true' : undefined} aria-label={`查看 ${title} 的原文片段 ${index + 1}`}>
          <span className="source-preview-meta">原文 {index + 1}<span>展开原文</span></span>
          <span className="source-preview-text">{fragment.original_text ? Array.from(fragment.original_text).slice(0, 220).join('') + (Array.from(fragment.original_text).length > 220 ? '…' : '') : '原文未保存'}</span>
        </a>)}</td></tr>
        {current && <tr className="source-expanded-row"><td colSpan={2}><EvidenceCard key={current.evidence_id} answerRunId={answerRunId} item={current} title={title} inline selected pageLookup={pageLookup} returnCitationId={returnCitationId} /></td></tr>}
        </Fragment>
      })}</tbody></table>}
    {!references.length && <p className="empty">{evidence.length ? '这份答案没有可定位的正文引用。' : '没有保存的证据片段。'}</p>}
    {selected && !citationNumbers.has(selected.evidence_id) && <EvidenceCard key={selected.evidence_id} answerRunId={answerRunId} item={selected} inline selected pageLookup={pageLookup} returnCitationId={returnCitationId} />}
    {otherEvidence.length > 0 && <details className="other-evidence">
      <summary>其他保存的片段（{otherEvidence.length}）</summary>
      <p className="section-note">这些片段未在正文形成可定位引用。</p>
      {otherEvidence.map((item, index) => <details key={item.evidence_id} className="other-fragment">
        <summary><span>片段 {index + 1} · {documentTitles[item.document_id || ''] || item.upstream_dockey}</span>
          <span className="fragment-preview">{item.summary || '没有保存摘要'}</span></summary>
        <EvidenceCard answerRunId={answerRunId} item={item} title={documentTitles[item.document_id || '']} />
      </details>)}
    </details>}
  </details>
}
