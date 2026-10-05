import assert from 'node:assert/strict'
import test from 'node:test'
import { pendingSubmissions, submitApi } from './api.ts'
import { recoverPending } from './recovery.ts'

class MemoryStorage {
  data = new Map()
  getItem(key) { return this.data.get(key) ?? null }
  setItem(key, value) { this.data.set(key, value) }
  removeItem(key) { this.data.delete(key) }
}

test('a recovered import makes the page re-read the collection after replay', async () => {
  globalThis.sessionStorage = new MemoryStorage()
  globalThis.window = { setTimeout, clearTimeout }
  const events = []
  let committed = false
  globalThis.fetch = async (_path, options) => {
    events.push(`POST ${options.headers['Idempotency-Key']}`)
    if (!committed) {
      committed = true
      throw new TypeError('request outcome unknown')
    }
    return { ok: true, json: async () => ({ collection_id: 'collection-1', job_id: 'import-2' }) }
  }
  await assert.rejects(submitApi('import', '/api/collections/collection-1/imports', {
    expected_selection_revision: 1,
  }), TypeError)
  const onError = (error) => { throw error }
  await recoverPending(
    (operation) => events.push(`recovered ${operation}`),
    () => events.push('refresh collection'),
    onError,
  )
  assert.equal(events[0], events[1])
  assert.deepEqual(events.slice(2), ['recovered import', 'refresh collection'])
  assert.deepEqual(pendingSubmissions(), [])
})
