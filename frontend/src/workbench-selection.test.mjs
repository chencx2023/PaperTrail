import assert from 'node:assert/strict'
import test from 'node:test'
import * as screening from './screening.ts'

const pool = Array.from({ length: 60 }, (_, index) => ({
  display_rank: index + 1, versioned_id: `paper-${index}v2`, title: `Paper ${index}`,
  abstract: 'Saved abstract', authors: [],
  versions: [{ versioned_id: `paper-${index}v2`, version: 2 }, { versioned_id: `paper-${index}v1`, version: 1 }],
}))

test('20, 30 and 60 candidates remain reachable exactly once through ten-paper pages', () => {
  assert.equal(typeof screening.candidatePage, 'function', '候选分页尚未实现')
  for (const count of [20, 30, 60]) {
    const papers = pool.slice(0, count)
    const original = structuredClone(papers)
    const visited = []
    const first = screening.candidatePage(papers, 1)
    assert.equal(first.pages, Math.ceil(count / 10))
    for (let page = 1; page <= first.pages; page++) {
      const slice = screening.candidatePage(papers, page)
      assert.equal(slice.items.length, 10)
      assert.equal(slice.start, (page - 1) * 10 + 1)
      assert.equal(slice.end, page * 10)
      visited.push(...slice.items.map(item => item.versioned_id))
    }
    assert.deepEqual(visited, papers.map(item => item.versioned_id))
    assert.deepEqual(papers, original)
  }
})

test('a shrinking selected view clamps the page and an empty view has no phantom rows', () => {
  assert.equal(typeof screening.candidatePage, 'function', '候选分页尚未实现')
  const last = screening.candidatePage(pool.slice(0, 11), 10)
  assert.equal(last.page, 2)
  assert.equal(last.start, 11)
  assert.equal(last.end, 11)
  assert.equal(last.items[0].display_rank, 11)
  const empty = screening.candidatePage([], 4)
  assert.deepEqual([empty.page, empty.pages, empty.start, empty.end, empty.items.length], [1, 1, 0, 0, 0])
})

test('alternate fixed versions stay selected when reading advice reorders pages and views', () => {
  assert.equal(typeof screening.changeVersionSelection, 'function', '固定版本选择尚未实现')
  let selected = screening.changeVersionSelection([], 'paper-0v1', true)
  selected = screening.changeVersionSelection(selected, 'paper-20v2', true)
  const result = { papers: pool.map((paper, index) => ({ paper_id: paper.versioned_id,
    label: index === 20 ? 'high' : 'low', reason: '', evidence_quote: '' })) }
  const all = screening.screenedCandidates(pool, result, 'all', selected)
  assert.equal(all[0].display_rank, 21)
  assert.equal(screening.candidatePage(all, 3).items[0].versioned_id, 'paper-19v2')
  assert.deepEqual(screening.screenedCandidates(pool, result, 'selected', selected).map(p => p.versioned_id), ['paper-20v2', 'paper-0v2'])
  assert.deepEqual(screening.screenedCandidates(pool, null, 'selected', selected).map(p => p.versioned_id), ['paper-0v2', 'paper-20v2'])
  assert.deepEqual(selected, ['paper-0v1', 'paper-20v2'])
})

test('selection is capped at ten fixed versions, is idempotent, and can always release a slot', () => {
  assert.equal(typeof screening.changeVersionSelection, 'function', '固定版本选择尚未实现')
  const ten = pool.slice(0, 10).map(p => p.versioned_id)
  assert.deepEqual(screening.changeVersionSelection(ten, pool[10].versioned_id, true), ten)
  assert.deepEqual(screening.changeVersionSelection(ten, ten[0], true), ten)
  const nine = screening.changeVersionSelection(ten, ten[0], false)
  const replaced = screening.changeVersionSelection(nine, pool[10].versioned_id, true)
  assert.equal(replaced.length, 10)
  assert.equal(replaced.includes(ten[0]), false)
  assert.equal(replaced.includes(pool[10].versioned_id), true)
  assert.equal(ten.length, 10)
})
