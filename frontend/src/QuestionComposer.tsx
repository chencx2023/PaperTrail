import type { ResearchState } from './api'

export function QuestionComposer({ papers, selected, question, disabled, ready, onSelect, onQuestion, onSubmit }: {
  papers: ResearchState['papers']; selected: string[]; question: string; disabled: boolean; ready: boolean
  onSelect: (id: string, checked: boolean) => void; onQuestion: (value: string) => void; onSubmit: () => void
}) {
  return <form className="question-composer" onSubmit={event => { event.preventDefault(); onSubmit() }}>
    <div className="question-editor"><label htmlFor="research-question">向论文提问</label>
      <textarea id="research-question" rows={5} value={question} onChange={event => onQuestion(event.target.value)} placeholder="想从这些论文中了解什么？" disabled={disabled} />
      <div className="question-submit"><span>{selected.length} / {papers.length} 篇参与回答</span><button type="submit" disabled={disabled || !ready || !question.trim() || !selected.length}>生成有证据的回答</button></div>
    </div>
    <fieldset className="answer-paper-pool"><legend>参与回答的文献 <span>{selected.length}</span></legend>
      <div className="answer-paper-list">{papers.map((paper, index) => <label className={`answer-paper${selected.includes(paper.document_id) ? ' is-selected' : ''}`} key={paper.document_id} title={paper.versioned_id}>
        <input type="checkbox" checked={selected.includes(paper.document_id)} disabled={disabled} onChange={event => onSelect(paper.document_id, event.target.checked)} aria-label={`参与回答：${paper.title}`} />
        <span className="answer-paper-number">{index + 1}</span><span><strong>{paper.title}</strong><small>{paper.authors.slice(0, 2).join(', ') || '作者未记录'}{paper.authors.length > 2 ? ` 等 ${paper.authors.length} 位作者` : ''}</small></span>
      </label>)}</div>
    </fieldset>
  </form>
}
