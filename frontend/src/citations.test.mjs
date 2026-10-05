import assert from 'node:assert/strict'
import test from 'node:test'
import { citationReturnId, evidenceHref, splitCitationText } from './citations.ts'

test('inline citations navigate without a References section', () => {
  const parts = splitCitationText(
    'A pqac-a B pqac-b.',
    [
      { upstream_context_id: 'pqac-a', start: 2, end: 8, evidence_id: 'run:pqac-a', mapping_status: 'mapped' },
      { upstream_context_id: 'pqac-b', start: 11, end: 17, evidence_id: 'run:pqac-b', mapping_status: 'mapped' },
    ],
    new Set(['run:pqac-a', 'run:pqac-b']),
  )
  assert.deepEqual(parts, [
    { kind: 'text', text: 'A ' },
    { kind: 'link', text: 'pqac-a', evidenceId: 'run:pqac-a', number: 1 },
    { kind: 'text', text: ' B ' },
    { kind: 'link', text: 'pqac-b', evidenceId: 'run:pqac-b', number: 2 },
    { kind: 'text', text: '.' },
  ])
})

test('saved Python offsets still locate citations after emoji', () => {
  assert.deepEqual(splitCitationText(
    '🧭结论pqac-a。',
    [{ upstream_context_id: 'pqac-a', start: 3, end: 9, evidence_id: 'run:pqac-a', mapping_status: 'mapped' }],
    new Set(['run:pqac-a']),
  ), [
    { kind: 'text', text: '🧭结论' },
    { kind: 'link', text: 'pqac-a', evidenceId: 'run:pqac-a', number: 1 },
    { kind: 'text', text: '。' },
  ])
})

test('unmapped or inconsistent citations remain visible but cannot open evidence', () => {
  assert.deepEqual(splitCitationText(
    'A pqac-a B pqac-b C pqac-c.',
    [
      { upstream_context_id: 'pqac-a', start: 2, end: 8, evidence_id: null, mapping_status: 'missing_context' },
      { upstream_context_id: 'pqac-b', start: 11, end: 17, evidence_id: 'other:pqac-b', mapping_status: 'mapped' },
      { upstream_context_id: 'pqac-c', start: 19, end: 25, evidence_id: 'run:pqac-c', mapping_status: 'mapped' },
    ],
    new Set(['run:pqac-c']),
  ), [
    { kind: 'text', text: 'A ' },
    { kind: 'unavailable', text: 'pqac-a', status: 'missing_context', number: 1 },
    { kind: 'text', text: ' B ' },
    { kind: 'unavailable', text: 'pqac-b', status: 'evidence_unavailable', number: 2 },
    { kind: 'text', text: ' C pqac-c.' },
  ])
})

test('evidence links preserve the saved answer address across refresh', () => {
  assert.equal(
    evidenceHref('/', '?search=s1&answer=a1', 'a1:pqac-123'),
    '/?search=s1&answer=a1&evidence=a1%3Apqac-123#evidence',
  )
  assert.equal(
    evidenceHref('/', '?answer=a1&evidence=old', 'a1:pqac-456'),
    '/?answer=a1&evidence=a1%3Apqac-456#evidence',
  )
})

test('repeated passages keep their number in answer order even if citations arrive out of order', () => {
  const citations = [
    { upstream_context_id: 'pqac-a', start: 14, end: 20, evidence_id: 'run:a', mapping_status: 'mapped' },
    { upstream_context_id: 'pqac-b', start: 0, end: 6, evidence_id: 'run:b', mapping_status: 'mapped' },
    { upstream_context_id: 'pqac-a', start: 7, end: 13, evidence_id: 'run:a', mapping_status: 'mapped' },
    { upstream_context_id: 'pqac-b', start: 21, end: 27, evidence_id: 'run:b', mapping_status: 'mapped' },
  ]
  const before = structuredClone(citations)
  const links = splitCitationText('pqac-b pqac-a pqac-a pqac-b', citations, new Set(['run:a', 'run:b']))
    .filter((part) => part.kind === 'link')
  assert.deepEqual(links.map((part) => [part.number, part.evidenceId]), [
    [1, 'run:b'], [2, 'run:a'], [2, 'run:a'], [1, 'run:b'],
  ])
  assert.deepEqual(citations, before)
})

test('citation deep links save their exact occurrence while evidence selection clears the old occurrence', () => {
  assert.equal(evidenceHref('/', '?research=r1&view=qa', 'run:a', 'citation-3'),
    '/?research=r1&view=qa&evidence=run%3Aa&citation=citation-3#evidence')
  assert.equal(evidenceHref('/', '?research=r1&evidence=run%3Aa&citation=citation-3', 'run:b'),
    '/?research=r1&evidence=run%3Ab#evidence')
})

test('saved citation navigation restores only a verified occurrence of the selected evidence', () => {
  const parts = [
    { kind: 'text', text: 'A ' },
    { kind: 'link', text: 'pqac-a', evidenceId: 'run:a', number: 1 },
    { kind: 'text', text: ' B ' },
    { kind: 'link', text: 'pqac-a', evidenceId: 'run:a', number: 1 },
    { kind: 'link', text: 'pqac-b', evidenceId: 'run:b', number: 2 },
    { kind: 'unavailable', text: 'missing', status: 'missing_context', number: 3 },
  ]
  assert.equal(citationReturnId(parts, 'run:a', '#citation-3', 'citation-1'), 'citation-3')
  assert.equal(citationReturnId(parts, 'run:a', '#evidence', 'citation-3'), 'citation-3')
  assert.equal(citationReturnId(parts, 'run:a', '#evidence'), 'citation-1')
  assert.equal(citationReturnId(parts, 'run:a', '#citation-4', 'citation-4'), 'citation-1')
  assert.equal(citationReturnId(parts, 'run:a', '#citation-5'), 'citation-1')
  assert.equal(citationReturnId(parts, 'missing', '#citation-5'), '')
})
