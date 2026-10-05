import assert from 'node:assert/strict'
import test from 'node:test'
import { build } from 'esbuild'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import fs from 'node:fs'

async function render(fusion) {
  assert.ok(fs.existsSync('src/SearchFusion.tsx'), '融合结果说明尚未实现')
  const output = await build({ entryPoints: ['src/SearchFusion.tsx'], bundle: true, platform: 'node', format: 'esm', write: false, logLevel: 'silent' })
  const { SearchFusion } = await import(`data:text/javascript;base64,${Buffer.from(output.outputFiles[0].text).toString('base64')}`)
  return renderToStaticMarkup(createElement(SearchFusion, { search: { query_text: 'coding agent', fusion, candidates: [], query_plan: null } }))
}

test('fusion explains partial results, preserved source ranks and unknown model cost', async () => {
  const html = await render({ method: 'rrf', successful_queries: 2, partial: true, candidate_count: 35,
    http_attempts: 3, cache_hits: 0, elapsed_ms: 1000, generation: { model: 'test-model', usage: { total_tokens: 42 }, cost_usd: null },
    queries: [{ index: 1, label: '原查询', status: 'success', candidate_count: 20, cache_hit: false, original_order: [], fetched_at: null },
      { index: 2, label: '同义表达', status: 'failed', candidate_count: null, cache_hit: false, original_order: [], fetched_at: null },
      { index: 3, label: '能力评估', status: 'success', candidate_count: 20, cache_hit: false, original_order: [], fetched_at: null }] })
  assert.match(html, /2\/3 组查询完成/)
  assert.match(html, /35 篇/)
  assert.match(html, /部分查询未完成/)
  assert.match(html, /按摘要筛选后优先展示阅读建议/)
  assert.doesNotMatch(html, /前 30 篇|RRF 融合顺序/)
  assert.match(html, /费用未知/)
  assert.doesNotMatch(html, /置信度|更相关的35篇/)
  assert.equal(await render(null), '')
})
