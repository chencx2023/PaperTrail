import assert from 'node:assert/strict'
import test from 'node:test'
import * as api from './api.ts'
import { pendingSubmissions, replayPending, submitApi } from './api.ts'

class MemoryStorage {
  data = new Map()
  getItem(key) { return this.data.get(key) ?? null }
  setItem(key, value) { this.data.set(key, value) }
  removeItem(key) { this.data.delete(key) }
}

test('a lost HTTP response replays the same key and original body', async () => {
  globalThis.sessionStorage = new MemoryStorage()
  globalThis.window = { setTimeout, clearTimeout }
  const requests = []
  let committed = null
  globalThis.fetch = async (_path, options) => {
    requests.push({ key: options.headers['Idempotency-Key'], body: options.body })
    if (committed === null) {
      committed = { search_id: 'saved-search', job_id: 'saved-job' }
      throw new TypeError('response lost after commit')
    }
    return { ok: true, json: async () => committed }
  }
  const body = { question: '如何评估记忆？', query_text: 'LLM agent memory' }
  await assert.rejects(submitApi('search', '/api/searches', body), TypeError)
  const [pending] = pendingSubmissions()
  assert.equal(pending.body, JSON.stringify(body))
  const replay = await replayPending(pending)
  assert.deepEqual(replay, { search_id: 'saved-search', job_id: 'saved-job' })
  assert.deepEqual(requests, [requests[0], requests[0]])
  assert.deepEqual(pendingSubmissions(), [])
})

test('a changed request gets a different key', async () => {
  globalThis.sessionStorage = new MemoryStorage()
  globalThis.window = { setTimeout, clearTimeout }
  const keys = []
  globalThis.fetch = async (_path, options) => {
    keys.push(options.headers['Idempotency-Key'])
    throw new TypeError('offline')
  }
  await assert.rejects(submitApi('answer', '/api/answers', { question: 'A' }), TypeError)
  await assert.rejects(submitApi('answer', '/api/answers', { question: 'A' }), TypeError)
  await assert.rejects(submitApi('answer', '/api/answers', { question: 'B' }), TypeError)
  assert.equal(keys[0], keys[1])
  assert.notEqual(keys[1], keys[2])
})

test('saved context coverage distinguishes included and missing frozen papers', () => {
  assert.equal(typeof api.describeSavedContextCoverage, 'function')
  const view = api.describeSavedContextCoverage({
    status: 'known', basis: 'saved_paperqa_session_context', total_documents: 2,
    included_documents: [{ document_id: 'doc-a', versioned_id: '2503.07919v3' }],
    missing_documents: [{ document_id: 'doc-b', versioned_id: '2406.12373v3' }],
  })
  assert.equal(view.summary, '已保存的生成上下文涉及 1/2 篇固定论文')
  assert.deepEqual(view.included, ['2503.07919v3'])
  assert.deepEqual(view.missing, ['2406.12373v3'])
})

test('unknown saved context never becomes a zero-paper claim', () => {
  assert.equal(typeof api.describeSavedContextCoverage, 'function')
  const view = api.describeSavedContextCoverage({
    status: 'unknown', basis: 'saved_paperqa_session_context', total_documents: 2,
    included_documents: null, missing_documents: null,
  })
  assert.equal(view.summary, '已保存的生成上下文覆盖未知')
  assert.deepEqual(view.included, [])
  assert.deepEqual(view.missing, [])
})
