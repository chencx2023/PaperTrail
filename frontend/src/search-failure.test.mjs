import assert from 'node:assert/strict'
import test from 'node:test'
import { build } from 'esbuild'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

async function render(search) {
  const output = await build({ entryPoints: ['src/SearchFailureNotice.tsx'], bundle: true,
    platform: 'node', format: 'esm', write: false, logLevel: 'silent' })
  const { SearchFailureNotice } = await import(`data:text/javascript;base64,${Buffer.from(output.outputFiles[0].text).toString('base64')}`)
  return renderToStaticMarkup(createElement(SearchFailureNotice, { search }))
}

test('failed search explains the observed cause and refresh keeps the request count', async () => {
  const search = { status: 'failed', failure: { kind: 'rate_limited',
    message: 'arXiv 暂时限制请求（HTTP 429），请稍后重试。', http_attempts: 9 } }
  const first = await render(search)
  assert.match(first, /HTTP 429/)
  assert.match(first, /9 次/)
  assert.match(first, /已保存/)
  assert.doesNotMatch(first, /修改检索词/)
  assert.equal(await render(search), first)
  assert.match(await render({ status: 'failed', failure: null }), /本次搜索未完成/)
  assert.match(await render({ status: 'failed', failure: { kind: 'cooldown', message: '本地冷却中', http_attempts: 0 } }), /0 次/)
  assert.equal(await render({ status: 'succeeded', failure: null }), '')
  assert.equal(await render({ status: 'queued', failure: null }), '')
})
