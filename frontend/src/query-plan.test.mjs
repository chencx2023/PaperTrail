import assert from 'node:assert/strict'
import test from 'node:test'
import fs from 'node:fs'

const feature = fs.existsSync(new URL('./query-plan.ts', import.meta.url)) ? await import('./query-plan.ts') : null
const expansions = [
  { label: '同义表达', groups: [['coding agent', 'software engineering agent']], exclude: [] },
  { label: '能力评测', groups: [['coding agent'], ['evaluation', 'benchmark']], exclude: [] },
]

test('validated editable plan retains baseline and supports phrases/OR without adding queries', () => {
  assert.ok(feature, '多查询前端模块尚未实现')
  assert.deepEqual(feature.validatedExpansionPlan('  coding   agent ', expansions), { version: 1, baseline_query_text: 'coding agent', expansions })
})

test('bad syntax, duplicate searches or too many expansions cannot be adopted', () => {
  assert.ok(feature, '多查询前端模块尚未实现')
  for (const values of [[], [...expansions, expansions[0]], [expansions[0], expansions[0]],
    [{ label: 'bad', groups: [['ti:agent']], exclude: [] }],
    [{ label: 'bad', groups: [['agent OR memory']], exclude: [] }],
    [{ label: 'duplicate', groups: [['coding'], ['agent']], exclude: [] }],
  ]) assert.throws(() => feature.validatedExpansionPlan('coding agent', values))
  assert.throws(() => feature.validatedExpansionPlan('coding agent agent', [{ label: 'duplicate', groups: [['coding'], ['agent']], exclude: [] }]))
})
