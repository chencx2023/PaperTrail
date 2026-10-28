import { evidenceHref, splitCitationText, type SavedCitation } from './citations'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { citationMarkdown } from './markdown-citations'
import type { DocumentSource, Evidence } from './api'
import { paperKey, paperCitationLabel, savedPaperSource } from './paper-sources'
import { hasUnreadableSymbols } from './answer-reading'

type Props = {
  rawAnswer: string
  displayAnswer: string
  citations: SavedCitation[]
  evidenceIds: ReadonlySet<string>
  pathname: string
  search: string
  selectedEvidenceId?: string
  hideDetails?: boolean
  evidence?: Evidence[]
  documentSources?: Record<string, DocumentSource>
}

const mappingLabels: Record<string, string> = {
  mapped: '已映射',
  evidence_unavailable: '保存证据未载入',
  missing_context: '缺少对应证据',
  outside_scope: '不在本次文献范围',
}

export function CitationAnswer({ rawAnswer, displayAnswer, citations, evidenceIds, pathname, search, selectedEvidenceId, hideDetails, evidence, documentSources = {} }: Props) {
  const parts = splitCitationText(rawAnswer, citations, evidenceIds)
  const unverifiedOffsets = citations.length - parts.filter((part) => part.kind !== 'text').length
  const { markdown, groups, remarkCitations } = citationMarkdown(rawAnswer ? parts :
    [{ kind: 'text', text: displayAnswer || '没有生成可展示答案。' }], !!evidence)
  const byId = new Map(evidence?.map(item => [item.evidence_id, item]) || [])
  const paperNumbers = new Map<string, number>()
  parts.forEach(part => { if (part.kind === 'link') {
    const item = byId.get(part.evidenceId), key = item ? paperKey(item) : part.evidenceId
    if (!paperNumbers.has(key)) paperNumbers.set(key, paperNumbers.size + 1)
  } })
  const renderGroup = (key: string) => {
    const group = groups.get(key)!
    if (group.role === 'origin') return <>{group.items.map(reference => reference.kind === 'link' && <span
      key={reference.partIndex} id={`citation-${reference.partIndex}`} className="citation-origin" tabIndex={-1}
      data-citation-anchor="true" data-evidence-id={reference.evidenceId}><span className="sr-only">引用原始位置</span></span>)}</>
    if (group.role === 'paragraph') {
      const papers = new Map<string, typeof group.items>()
      for (const reference of group.items) if (reference.kind === 'link') {
        const item = byId.get(reference.evidenceId), key = item ? paperKey(item) : reference.evidenceId
        papers.set(key, [...(papers.get(key) || []), reference])
      }
      return <span className="paragraph-citations">{Array.from(papers, ([key, references], index) => {
        const first = references[0]
        if (first.kind !== 'link') return null
        const item = byId.get(first.evidenceId), source = item && savedPaperSource(item, documentSources)
        const label = paperCitationLabel(source, paperNumbers.get(key)!, documentSources)
        const ids = [...new Set(references.flatMap(ref => ref.kind === 'link' ? [ref.evidenceId] : []))]
        return <span key={key}>{index > 0 && <span className="citation-separator">; </span>}<a className="citation-link paper-citation"
          data-citation-id={`citation-${first.partIndex}`} data-evidence-id={first.evidenceId} data-evidence-ids={ids.join(' ')}
          aria-current={selectedEvidenceId && ids.includes(selectedEvidenceId) ? 'true' : undefined}
          title={source?.authors.length && source.year ? `${source.title} · ${ids.length} 个对应原文片段` : '作者年份未保存，使用可核对的论文编号'}
          aria-label={`查看 ${label} 的引用原文，共 ${ids.length} 个片段`}
          href={evidenceHref(pathname, search, first.evidenceId, `citation-${first.partIndex}`)}>{label}</a></span>
      })}</span>
    }
    return <sup className="citation-group">{group.items.map(reference => reference.kind === 'link' ?
      <a className="citation-link" key={reference.partIndex} id={`citation-${reference.partIndex}`}
        data-evidence-id={reference.evidenceId}
        aria-current={reference.evidenceId === selectedEvidenceId ? 'true' : undefined}
        href={evidenceHref(pathname, search, reference.evidenceId, `citation-${reference.partIndex}`)}
        aria-label={`查看引用 ${reference.number} 的证据`}>[{reference.number}]</a> :
      <span className="citation-unavailable" key={reference.partIndex}
        title={mappingLabels[reference.status] || reference.status}>[{reference.number}]（{mappingLabels[reference.status] || reference.status}）</span>
    )}</sup>
  }
  return <>
    <h2 id="answer-text-heading" className="sr-only">答案</h2>
    <div className="answer-text markdown-answer"><Markdown
      remarkPlugins={[remarkGfm, remarkMath, remarkCitations]} rehypePlugins={[[rehypeKatex, { strict: false, throwOnError: false, trust: false }]]}
      components={{ a: ({ href, children, node: _node, ...props }) => href && groups.has(href)
        ? renderGroup(href) : <a {...props} href={href}>{children}</a> }}
    >{markdown}</Markdown></div>
    {hasUnreadableSymbols(rawAnswer || displayAnswer) && <p className="section-note">提取文本含无法识别的符号（�）；公式与符号请核对引用 PDF，原始记录保留在技术详情中。</p>}
    {unverifiedOffsets > 0 && <p className="section-note">{unverifiedOffsets} 条引用位置无法核对；已保留原始文字，未生成猜测链接。</p>}
    {!hideDetails && <CitationDetails {...{ rawAnswer, displayAnswer, citations, evidenceIds, pathname, search }} />}
  </>
}

export function CitationDetails({ rawAnswer, displayAnswer, citations, evidenceIds }: Props) {
  const parts = splitCitationText(rawAnswer, citations, evidenceIds)
  const numbers = new Map(parts.flatMap((part) => part.kind === 'link' ? [[part.evidenceId, part.number] as const] : []))
  const groups = new Map<string, SavedCitation[]>()
  for (const citation of citations) groups.set(citation.upstream_context_id,
    [...(groups.get(citation.upstream_context_id) || []), citation])
  return <details className="citation-details"><summary>原始答案与引用映射</summary>
    <p className="raw-answer">{displayAnswer || rawAnswer || '无格式化答案'}</p>
    <div className="citation-list">{Array.from(groups, ([key, occurrences]) => <details key={key}>
      <summary>{key} · {occurrences.length} 次出现</summary>
      {occurrences.map((citation, index) => <p key={index} className="section-note">
        {citation.evidence_id && numbers.has(citation.evidence_id) ? `正文 [${numbers.get(citation.evidence_id)}] · ` : ''}
        位置 {citation.start}–{citation.end} · {mappingLabels[citation.mapping_status] || citation.mapping_status}
        {citation.mapping_status === 'mapped' && (!citation.evidence_id || !evidenceIds.has(citation.evidence_id)) && ' · 保存证据未载入'}
      </p>)}
    </details>)}</div>
  </details>
}
