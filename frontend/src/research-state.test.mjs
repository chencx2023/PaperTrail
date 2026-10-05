import assert from 'node:assert/strict'
import test from 'node:test'
import { initialAnswerSelection, changeAnswerSelection } from './research-state.ts'

test('new study selects all prepared papers; saved scope restores only its fixed document IDs', () => {
  const papers = [{document_id: 'v1'}, {document_id: 'v2'}, {document_id: 'other'}]
  assert.deepEqual(initialAnswerSelection(papers, null), ['v1', 'v2', 'other'])
  assert.deepEqual(initialAnswerSelection(papers, ['v2', 'foreign']), ['v2'])
  assert.deepEqual(initialAnswerSelection(papers, []), [])
})

test('local scope changes are distinct, stay in pool order and cannot add a foreign document', () => {
  const papers = [{document_id: 'v1'}, {document_id: 'v2'}, {document_id: 'other'}]
  assert.deepEqual(changeAnswerSelection(papers, ['v2'], 'v1', true), ['v1', 'v2'])
  assert.deepEqual(changeAnswerSelection(papers, ['v1', 'v2'], 'v2', true), ['v1', 'v2'])
  assert.deepEqual(changeAnswerSelection(papers, ['v1'], 'foreign', true), ['v1'])
  assert.deepEqual(changeAnswerSelection(papers, ['v1'], 'v1', false), [])
})
