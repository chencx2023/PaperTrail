import type { Candidate } from './api'
import { Modal, CloseIcon, ExternalIcon } from './Modal'
import { arxivPdfUrl } from './research-flow'
import { relevanceLabels, type PaperAssessment } from './screening'

export function PaperDetail({ candidate, assessment, unverified, origins, selectedVersions, disabled, onSelect, onClose }: {
  candidate: Candidate; assessment?: PaperAssessment; unverified?: string; origins?: string
  selectedVersions: string[]; disabled: boolean
  onSelect: (id: string, checked: boolean) => void; onClose: () => void
}) {
  const pdf = arxivPdfUrl(candidate.versioned_id)
  return <Modal className="paper-detail" labelledBy="paper-detail-title" onClose={onClose}>
    <div className="detail-top"><span>论文详情</span><div className="detail-top-actions">{pdf && <a href={pdf} target="_blank" rel="noopener noreferrer" aria-label={`阅读 arXiv PDF，固定版本 ${candidate.versioned_id}`} title={`arXiv ${candidate.versioned_id}`}>阅读 PDF<ExternalIcon /></a>}<button className="icon-button secondary" aria-label="关闭论文详情" onClick={onClose}><CloseIcon /></button></div></div>
    <div className="detail-reading">
      <div className="detail-meta"><span>arXiv · {candidate.versioned_id}</span><span className={`relevance-label ${assessment ? `relevance-${assessment.label}` : ''}`}>{assessment ? relevanceLabels[assessment.label] : unverified ? '依据未核实' : '未筛选'}</span></div>
      <h2 id="paper-detail-title">{candidate.title}</h2>
      <p className="full-byline">{candidate.authors.join(', ') || '作者未知'}</p>
      <section className="detail-abstract" aria-labelledby="detail-abstract-heading"><h3 id="detail-abstract-heading">摘要</h3><p>{candidate.abstract || '来源未提供摘要。'}</p></section>
      {assessment && <section className="detail-assessment" aria-labelledby="detail-assessment-heading"><h3 id="detail-assessment-heading">与研究问题的关联</h3><p>{assessment.reason}</p>{assessment.evidence_quote && <blockquote><span>已核对的摘要原句</span>{assessment.evidence_quote}</blockquote>}</section>}
      {unverified && <p className="section-note">{unverified} · 可在候选列表重试未核实论文。</p>}
      {origins && <p className="candidate-origins">来源排名 · {origins}</p>}
    </div>
    <div className="detail-versions"><h3>阅读与选择版本</h3><p className="section-note">摘要来自 {candidate.versioned_id}；PDF 按下方版本打开。</p>
      {candidate.versions.map(version => {
        const checked = selectedVersions.includes(version.versioned_id)
        const pdf = arxivPdfUrl(version.versioned_id)
        return <div className="detail-version" key={version.versioned_id}>
          <label><input type="checkbox" checked={checked} disabled={disabled || !checked && selectedVersions.length >= 10} aria-label={`选入固定版本 ${version.versioned_id}`} onChange={event => onSelect(version.versioned_id, event.target.checked)} /><span>{version.versioned_id}<small>{checked ? '已选入研究' : '选入研究'}</small></span></label>
          {pdf && <a href={pdf} target="_blank" rel="noopener noreferrer" className="button-link secondary" aria-label={`阅读 arXiv PDF，固定版本 ${version.versioned_id}`}>阅读 PDF<ExternalIcon /></a>}
        </div>
      })}
      {selectedVersions.length >= 10 && <p className="section-note">已选满 10 个固定版本，可先取消一项再换入其他版本。</p>}
    </div>
  </Modal>
}
