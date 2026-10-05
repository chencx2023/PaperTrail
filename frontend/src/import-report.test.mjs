import assert from 'node:assert/strict'
import test from 'node:test'
import { build } from 'esbuild'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

async function render(collection, extra = {}) {
  const output = await build({ entryPoints: ['src/ImportReport.tsx'], bundle: true, platform: 'node', format: 'esm', write: false, logLevel: 'silent' })
  const { ImportReport } = await import(`data:text/javascript;base64,${Buffer.from(output.outputFiles[0].text).toString('base64')}`)
  return renderToStaticMarkup(createElement(ImportReport, { collection, confirmedIds: ['success', 'failed'], dirty: false, disabled: false,
    onSelect: () => {}, onConfirm: () => {}, onContinue: () => {}, onRefresh: () => {}, ...extra }))
}
const collection = { collection_id: 'a', selection_revision: 2, import_job_status: 'succeeded',
  selections: [{ versioned_id: '2501.00001v1' }, { versioned_id: '2501.00002v1' }],
  import_report: { import_run_id: 'r1', selection_revision: 2, status: 'partial_failed', items: [
    { position: 1, title: 'Successful paper', versioned_id: '2501.00001v1', status: 'succeeded', document_id: 'success' },
    { position: 2, title: 'Failed paper', versioned_id: '2501.00002v1', status: 'failed', document_id: 'failed', failure_reason: 'PDF unavailable' },
  ] } }

test('partial reports allow one explicit batch confirmation and never select failed items', async () => {
  const html = await render(collection)
  assert.match(html, /确认 1 篇并建索引/)
  assert.match(html, /全选成功论文/)
  assert.match(html, /aria-label="确认 Failed paper[^>]*disabled=""/)
  assert.equal((html.match(/checked=""/g) || []).length, 1)
})
test('dirty selections and obsolete reports prevent confirmation against the old draft', async () => {
  for (const html of [await render(collection, { dirty: true }), await render({ ...collection, selection_revision: 3 })]) {
    assert.doesNotMatch(html, /checked=""/)
    assert.match(html, /disabled=""[^>]*>确认 0 篇并建索引/)
    assert.match(html, /1 篇成功 · 1 篇失败/)
  }
})
test('all-failed and in-progress reports cannot be confirmed', async () => {
  const failed = { ...collection, import_report: { ...collection.import_report, items: collection.import_report.items.slice(1) } }
  assert.match(await render(failed), /没有成功导入的论文/)
  assert.match(await render(collection, { disabled: true }), /disabled=""[^>]*>确认 1 篇并建索引/)
})
