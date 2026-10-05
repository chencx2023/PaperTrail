import assert from 'node:assert/strict'
import test from 'node:test'
import * as flow from './research-flow.ts'

const selections = [{ search_id: 'search-a', versioned_id: '2503.07919v3' }, { search_id: 'search-a', versioned_id: '2406.12373v3' }]
const collection = { collection_id: 'collection-a', selection_revision: 2, selections,
  import_report: { import_run_id: 'run-a', selection_revision: 2, items: [
    { versioned_id: selections[0].versioned_id, status: 'succeeded', document_id: 'doc-a' },
    { versioned_id: selections[1].versioned_id, status: 'failed', document_id: null },
    { versioned_id: 'outside-v1', status: 'succeeded', document_id: 'outside' },
  ] } }

test('paper PDF links keep the chosen fixed version for modern and legacy arXiv identities', () => {
  assert.equal(typeof flow.arxivPdfUrl, 'function')
  assert.equal(flow.arxivPdfUrl('2003.02320v6'), 'https://arxiv.org/pdf/2003.02320v6')
  assert.equal(flow.arxivPdfUrl('hep-th/9901001v2'), 'https://arxiv.org/pdf/hep-th/9901001v2')
  for (const invalid of ['2003.02320', '2003.02320v0', '../secret', 'https://example.org/', '2003.02320v6?redirect=1']) assert.equal(flow.arxivPdfUrl(invalid), null)
})

test('confirmation defaults contain only successful documents from the current draft and report', () => {
  assert.equal(typeof flow.successfulImportIds, 'function')
  assert.deepEqual(flow.successfulImportIds(collection), ['doc-a'])
  assert.deepEqual(flow.successfulImportIds({ ...collection, selection_revision: 3 }), [])
  assert.deepEqual(flow.successfulImportIds({ ...collection, import_report: null }), [])
  assert.deepEqual(flow.successfulImportIds({ ...collection, import_report: { ...collection.import_report, items: collection.import_report.items.filter(item => item.status === 'failed') } }), [])
})

test('report refresh preserves manual deselection; a new report or changed draft resets confirmation', () => {
  assert.equal(typeof flow.confirmationForReport, 'function')
  const initial = flow.confirmationForReport(collection, '', [])
  assert.deepEqual(initial.ids, ['doc-a'])
  assert.deepEqual(flow.confirmationForReport(collection, initial.marker, []).ids, [])
  const next = { ...collection, import_report: { ...collection.import_report, import_run_id: 'run-b' } }
  assert.deepEqual(flow.confirmationForReport(next, initial.marker, []).ids, ['doc-a'])
  assert.deepEqual(flow.confirmationForReport({ ...collection, selection_revision: 3 }, initial.marker, ['doc-a']).ids, [])
})

test('updating and importing uses the acknowledged new draft version; unchanged selections skip PATCH', async () => {
  assert.equal(typeof flow.updateAndImport, 'function')
  const calls = []
  const io = {
    save: async body => { calls.push(['PATCH', body]); return { collection_id: 'collection-a', selection_revision: 3 } },
    onSaved: result => calls.push(['saved', result.selection_revision]),
    import: async body => { calls.push(['POST', body]); return { job_id: 'job-a' } },
  }
  const changed = selections.slice(0, 1)
  await flow.updateAndImport(collection, changed, io)
  assert.deepEqual(calls, [['PATCH', { expected_selection_revision: 2, selections: changed }], ['saved', 3], ['POST', { expected_selection_revision: 3 }]])
  calls.length = 0
  await flow.updateAndImport(collection, [...selections].reverse(), io)
  assert.deepEqual(calls, [['POST', { expected_selection_revision: 2 }]])
})

test('a rejected or ambiguous draft save never starts an import against a stale version', async () => {
  assert.equal(typeof flow.updateAndImport, 'function')
  let imports = 0
  for (const save of [async () => { throw new Error('VERSION_CONFLICT') }, async () => ({ collection_id: 'collection-a', selection_revision: 2 }), async () => ({ collection_id: 'other', selection_revision: 3 })]) {
    await assert.rejects(flow.updateAndImport(collection, selections.slice(0, 1), { save, onSaved: () => {}, import: async () => imports++ }))
  }
  assert.equal(imports, 0)
})

test('the acknowledged draft is reflected in the UI before the import is submitted', async () => {
  const events = []
  await flow.updateAndImport(collection, selections.slice(0, 1), {
    save: async () => ({ collection_id: 'collection-a', selection_revision: 3 }),
    onSaved: async () => { await Promise.resolve(); events.push('draft-refreshed') },
    import: async () => { events.push('import-submitted') },
  })
  assert.deepEqual(events, ['draft-refreshed', 'import-submitted'])
})
