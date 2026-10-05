import type { CollectionState } from './api'

export function arxivPdfUrl(versionId: string): string | null {
  if (!/^(?:\d{4}\.\d{4,5}|[a-z][a-z.-]*\/\d{7})v[1-9]\d*$/i.test(versionId)) return null
  return `https://arxiv.org/pdf/${versionId.split('/').map(encodeURIComponent).join('/')}`
}

type Selection = { search_id: string | null; versioned_id: string }
export function sameSelections(a: Selection[], b: Selection[]) {
  return a.length === b.length && a.every(item => b.some(other =>
    other.search_id === item.search_id && other.versioned_id === item.versioned_id))
}

export function successfulImportIds(collection: CollectionState): string[] {
  const report = collection.import_report
  if (!report || report.selection_revision !== collection.selection_revision) return []
  const versions = new Set(collection.selections.map(item => item.versioned_id))
  return [...new Set(report.items.flatMap(item => item.status === 'succeeded' && item.document_id && versions.has(item.versioned_id) ? [item.document_id] : []))]
}

export function confirmationForReport(collection: CollectionState, previousMarker: string, previousIds: string[]) {
  const report = collection.import_report
  const marker = `${collection.collection_id}:${collection.selection_revision}:${report?.import_run_id || ''}:${report?.selection_revision || ''}`
  const successful = successfulImportIds(collection)
  return { marker, ids: marker === previousMarker ? previousIds.filter(id => successful.includes(id)) : successful }
}

type SavedDraft = { collection_id: string; selection_revision: number }
export async function updateAndImport(collection: CollectionState, selections: Selection[], io: {
  save: (body: { expected_selection_revision: number; selections: Selection[] }) => Promise<SavedDraft>
  onSaved: (draft: SavedDraft) => void | Promise<void>
  import: (body: { expected_selection_revision: number }) => Promise<unknown>
}) {
  let version = collection.selection_revision
  if (!sameSelections(collection.selections, selections)) {
    const saved = await io.save({ expected_selection_revision: version, selections })
    if (saved.collection_id !== collection.collection_id || !Number.isInteger(saved.selection_revision) || saved.selection_revision <= version) {
      throw new Error('未能核对保存后的文献版本，请重新读取清单后再导入。')
    }
    version = saved.selection_revision
    await io.onSaved(saved)
  }
  return io.import({ expected_selection_revision: version })
}
