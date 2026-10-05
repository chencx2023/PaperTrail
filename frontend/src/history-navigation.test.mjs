import assert from 'node:assert/strict'
import test from 'node:test'
import * as navigation from './history-navigation.ts'

test('opening and closing sidebar history preserves the current research and citation', () => {
  assert.equal(typeof navigation.historyHref, 'function', '侧栏历史导航尚未实现')
  const current = '/?search=s&collection=c&revision=r&answer=a&evidence=e#citation-5'
  const opened = navigation.historyHref(current, true)
  const query = new URL(opened, 'http://localhost').searchParams
  assert.equal(query.get('history'), '1')
  for (const [key, value] of new URL(current, 'http://localhost').searchParams) assert.equal(query.get(key), value)
  assert.equal(new URL(opened, 'http://localhost').hash, '#citation-5')
  assert.equal(navigation.historyHref(`${opened.replace('#', '&kind=answer&cursor=older#')}`, false), current)
})

test('saved searches, answers and frozen revisions reopen with history in the sidebar', () => {
  assert.equal(typeof navigation.historyHref, 'function', '侧栏历史导航尚未实现')
  for (const href of ['/?search=s', '/?answer=a&evidence=e#evidence', '/?collection=c&revision=r']) {
    const reopened = new URL(navigation.historyHref(href, true, { kind: 'answer', cursor: 'older page' }), 'http://localhost')
    assert.equal(reopened.searchParams.get('history'), '1')
    assert.equal(reopened.searchParams.get('kind'), 'answer')
    assert.equal(reopened.searchParams.get('cursor'), 'older page')
    for (const [key, value] of new URL(href, 'http://localhost').searchParams) assert.equal(reopened.searchParams.get(key), value)
    assert.equal(reopened.hash, new URL(href, 'http://localhost').hash)
  }
})
