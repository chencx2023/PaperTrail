import type { CollectionState } from './api'
import { successfulImportIds } from './research-flow'

export function ImportReport({ collection, confirmedIds, dirty, disabled, canConfirm = true, alreadyConfirmed = false, onSelect, onConfirm, onContinue, onRefresh }: {
  collection: CollectionState; confirmedIds: string[]; dirty: boolean; disabled: boolean; canConfirm?: boolean; alreadyConfirmed?: boolean
  onSelect: (ids: string[]) => void; onConfirm: () => void; onContinue: () => void; onRefresh: () => void
}) {
  const report = collection.import_report
  const current = !!report && report.selection_revision === collection.selection_revision && !dirty
  const successful = successfulImportIds(collection)
  const selected = current ? [...new Set(confirmedIds.filter(id => successful.includes(id)))] : []
  return <div className="report">
    <div className="report-header"><h3>导入结果</h3>{report && <span className="muted">{report.items.filter(item => item.status === 'succeeded' && item.document_id).length} 篇成功 · {report.items.filter(item => item.status !== 'succeeded').length} 篇失败</span>}</div>
    {dirty && <p className="report-message">文献选择已更改，请先更新并导入。</p>}
    {!report && <p className="empty">导入完成后，成功论文会默认选中，确认一次即可建立索引。</p>}
    {report && <>
      {!current && !dirty && <p className="report-message">这份报告对应旧的文献清单，请导入当前选择后再确认。</p>}
      {current && successful.length === 0 && <p className="report-message">本次没有成功导入的论文。可以重试或调整选择。</p>}
      {current && successful.length > 0 && <div className="report-selection"><span>成功论文默认选中，可取消单篇。</span><button className="secondary" disabled={disabled || selected.length === successful.length} onClick={() => onSelect(successful)}>全选成功论文</button></div>}
      <div className="import-list">{report.items.map(item => <label className={`import-item${item.status === 'succeeded' ? '' : ' import-failed'}`} key={`${report.import_run_id}-${item.position}`}>
        <input type="checkbox" aria-label={`确认 ${item.title?.trim() || item.versioned_id}，固定版本 ${item.versioned_id}`} disabled={disabled || !current || !item.document_id || !successful.includes(item.document_id)} checked={!!item.document_id && selected.includes(item.document_id)} onChange={event => {
          if (!item.document_id || !successful.includes(item.document_id)) return
          onSelect(event.target.checked ? [...new Set([...selected, item.document_id])] : selected.filter(id => id !== item.document_id))
        }} />
        <span><strong>{item.title?.trim() || item.versioned_id}</strong><small>固定版本 {item.versioned_id} · {item.status === 'succeeded' ? '导入成功' : item.failure_reason || '导入失败'}</small></span>
      </label>)}</div>
      <div className={`action-row report-confirmation${current && selected.length > 0 ? ' is-actionable' : ''}`}>
        {alreadyConfirmed && current ? <><button disabled={disabled} onClick={onContinue}>继续提问</button><span className="muted">这批文献已确认</span></>
          : <><button disabled={disabled || !canConfirm || !current || selected.length === 0} onClick={onConfirm}>确认 {selected.length} 篇并建索引</button><span className="muted">确认后用于本次问答</span></>}
        <button className="secondary" disabled={disabled} onClick={onRefresh}>刷新报告</button>
      </div>
    </>}
  </div>
}
