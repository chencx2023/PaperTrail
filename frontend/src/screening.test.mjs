import assert from 'node:assert/strict'
import test from 'node:test'
import fs from 'node:fs'

const exists = fs.existsSync(new URL('./screening.ts', import.meta.url))
const feature = exists ? await import('./screening.ts') : null
const candidates = ['a', 'b', 'c', 'd'].map((id, i) => ({
  display_rank: i + 1, versioned_id: id + 'v2', title: id, abstract: 'saved abstract',
  authors: [], versions: [{ versioned_id: id + 'v2', version: 2 }, { versioned_id: id + 'v1', version: 1 }],
}))
const search = { search_id: 'search', snapshot_sha256: 'hash', question: 'original', candidates, status: 'succeeded' }
const result = { search_id: 'search', snapshot_sha256: 'hash', question: 'original', candidate_count: 4,
  interpretation: '原问题', criteria: ['对象一致', '有依据'], papers: [
    { paper_id: 'av2', label: 'low', reason: '不同对象', evidence_quote: '' },
    { paper_id: 'bv2', label: 'partial', reason: '部分相关', evidence_quote: 'saved' },
    { paper_id: 'cv2', label: 'high', reason: '相关', evidence_quote: 'saved' },
    { paper_id: 'dv2', label: 'insufficient', reason: '不足', evidence_quote: '' },
  ] }

test('partial screening authenticates only verified rows and accounts for every unresolved identity', () => {
  const partial = { ...result, status: 'partial', papers: result.papers.slice(1),
    unverified_papers: [{ paper_id: 'av2', kind: 'unverified', message: '摘要依据未核实' }] }
  assert.equal(feature.screeningMatches(search, partial), true)
  assert.deepEqual(feature.screenedCandidates(candidates, partial, 'all', []).map(p => p.versioned_id), ['cv2', 'bv2', 'dv2', 'av2'])
  for (const invalid of [
    { ...partial, unverified_papers: [] },
    { ...partial, unverified_papers: [{ ...partial.unverified_papers[0], paper_id: 'bv2' }] },
    { ...partial, unverified_papers: [{ ...partial.unverified_papers[0], paper_id: 'outside' }] },
    { ...partial, status: 'succeeded' },
  ]) assert.equal(feature.screeningMatches(search, invalid), false)
})

test('all candidates prioritize verified reading advice and use saved ranking within each tier', () => {
  assert.ok(feature, '摘要筛选展示模块尚未实现')
  assert.deepEqual(feature.screenedCandidates([...candidates].reverse(), result, 'all', []).map(p => p.versioned_id), ['cv2', 'bv2', 'av2', 'dv2'])
  assert.deepEqual(candidates.map(p => p.versioned_id), ['av2', 'bv2', 'cv2', 'dv2'])
})

test('selected papers use the same reading tiers and retain chosen alternate versions', () => {
  assert.ok(feature, '摘要筛选展示模块尚未实现')
  assert.deepEqual(feature.screenedCandidates(candidates, result, 'selected', ['cv2', 'av1']).map(p => p.versioned_id), ['cv2', 'av2'])
})

test('missing or unmatched screening always preserves original candidates', () => {
  assert.ok(feature, '摘要筛选展示模块尚未实现')
  assert.deepEqual(feature.screenedCandidates(candidates, null, 'all', []), candidates)
  assert.equal(feature.screeningMatches(search, result), true)
  for (const changed of [
    { ...result, search_id: 'other' }, { ...result, question: 'edited' },
    { ...result, snapshot_sha256: 'changed' }, { ...result, papers: result.papers.slice(1) },
    { ...result, papers: [result.papers[0], result.papers[0], ...result.papers.slice(2)] },
  ]) assert.equal(feature.screeningMatches(search, changed), false)
})

test('weak relevance labels do not hide any candidate or invent a recommendation quota', () => {
  assert.ok(feature, '摘要筛选展示模块尚未实现')
  const low = { ...result, papers: result.papers.map(p => ({ ...p, label: 'low' })) }
  assert.equal(feature.screenedCandidates(candidates, low, 'all', []).length, 4)
  assert.equal(feature.screenedCandidates(candidates, low, 'selected', ['dv2']).length, 1)
})

test('explicit RRF screening scope accepts only top30 and retains unscreened papers and selected versions', () => {
  const pool = Array.from({ length: 35 }, (_, i) => ({ ...candidates[0], display_rank: i + 1,
    versioned_id: `paper-${i}`, versions: [{ versioned_id: `paper-${i}`, version: 1 }] }))
  const state = { ...search, candidates: pool, fusion: { method: 'rrf', candidate_count: 35 } }
  const scoped = { ...result, candidate_count: 30,
    screening_scope: { method: 'rrf_top', limit: 30, source_candidate_count: 35, paper_ids: pool.slice(0, 30).map(p => p.versioned_id) },
    papers: pool.slice(0, 30).map(p => ({ paper_id: p.versioned_id, label: 'high', reason: '相关', evidence_quote: 'saved' })) }
  assert.equal(feature.screeningMatches(state, scoped), true)
  assert.equal(feature.screeningMatches(state, { ...scoped, screening_scope: { ...scoped.screening_scope, source_candidate_count: 30 } }), false)
  assert.equal(feature.screeningMatches({ ...state, fusion: null }, scoped), false)
  assert.equal(feature.screenedCandidates(pool, scoped, 'all', []).length, 35)
  assert.deepEqual(feature.screenedCandidates(pool, scoped, 'selected', ['paper-34', 'paper-0']).map(p => p.versioned_id), ['paper-0', 'paper-34'])
})

test('full pool scope binds all sixty fixed identities and the new validation version', () => {
  const pool = Array.from({ length: 60 }, (_, i) => ({ ...candidates[0], display_rank: i + 1,
    versioned_id: `paper-${i}`, versions: [{ versioned_id: `paper-${i}`, version: 1 }] }))
  const state = { ...search, candidates: pool, fusion: { method: 'rrf', candidate_count: 60 } }
  const full = { ...result, candidate_count: 60, validation_version: 'abstract-relevance-fullpool-validation-v4',
    screening_scope: { method: 'all_candidates', limit: 60, source_candidate_count: 60, paper_ids: pool.map(p => p.versioned_id) },
    papers: pool.map(p => ({ paper_id: p.versioned_id, label: 'high', reason: '相关', evidence_quote: 'saved' })) }
  assert.equal(feature.screeningMatches(state, full), true)
  assert.equal(feature.screeningMatches({ ...state, fusion: null }, full), true)
  for (const changed of [
    { ...full, validation_version: 'unknown' },
    { ...full, candidate_count: 30 },
    { ...full, screening_scope: { ...full.screening_scope, paper_ids: full.screening_scope.paper_ids.slice(1) } },
    { ...full, screening_scope: { ...full.screening_scope, limit: 30 } },
  ]) assert.equal(feature.screeningMatches(state, changed), false)
})

test('same-tier ranking, ten-row pages and exact version selections survive reprioritization', () => {
  const ordered = { ...result, papers: result.papers.map(p => ({ ...p, label: 'high' })) }
  assert.deepEqual(feature.screenedCandidates([...candidates].reverse(), ordered, 'all', []), candidates)
  const many = Array.from({ length: 60 }, (_, i) => ({ ...candidates[0], display_rank: i + 1,
    versioned_id: `p${i}v2`, versions: [{ versioned_id: `p${i}v2` }, { versioned_id: `p${i}v1` }] }))
  const assessments = { papers: many.map((p, i) => ({ paper_id: p.versioned_id, label: i % 2 ? 'high' : 'low' })) }
  const selected = ['p0v1', 'p59v2']
  const sorted = feature.screenedCandidates(many, assessments, 'all', selected)
  assert.equal(feature.candidatePage(sorted, 2).start, 11)
  assert.equal(feature.candidatePage(sorted, 2).items.length, 10)
  assert.equal(feature.candidatePage(sorted, 6).end, 60)
  assert.deepEqual(feature.screenedCandidates(many, assessments, 'selected', selected).map(p => p.versioned_id), ['p59v2', 'p0v2'])
  assert.deepEqual(selected, ['p0v1', 'p59v2'])
})

test('server plan controls wait time and rejects inconsistent progress', () => {
  const plan = { total_count: 60, verified_count: 30, remaining_count: 30, batch_count: 3, wait_timeout_ms: 195000 }
  assert.equal(feature.screeningPlanMatches(60, plan), true)
  assert.equal(feature.screeningPlanMatches(60, { ...plan, batch_count: 6, wait_timeout_ms: 375000 }), true)
  for (const changed of [
    { ...plan, total_count: 30 }, { ...plan, remaining_count: 31 },
    { ...plan, wait_timeout_ms: 75000 }, { ...plan, batch_count: 0 },
    { ...plan, verified_count: -1 }, { ...plan, batch_count: 31, wait_timeout_ms: 1875000 },
  ]) assert.equal(feature.screeningPlanMatches(60, changed), false)
})
