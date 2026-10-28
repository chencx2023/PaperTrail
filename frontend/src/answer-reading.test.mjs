import assert from 'node:assert/strict'
import test from 'node:test'
import { build } from 'esbuild'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { readingText } from './answer-reading.ts'

const components = new Map()
async function component(file, name) {
  if (!components.has(file)) components.set(file, build({ entryPoints: [`src/${file}`], bundle: true,
    platform: 'node', format: 'esm', write: false, logLevel: 'silent' }).then(output =>
    import(`data:text/javascript;base64,${Buffer.from(output.outputFiles[0].text).toString('base64')}`)))
  return (await components.get(file))[name]
}

function answerProps(raw) {
  const chars = Array.from(raw), citations = []
  for (let start = 0; start < chars.length; start++) for (const id of ['a', 'b']) {
    if (chars.slice(start, start + 6).join('') === `pqac-${id}`) citations.push({
      upstream_context_id: `pqac-${id}`, start, end: start + 6,
      evidence_id: `run:${id}`, mapping_status: 'mapped',
    })
  }
  return { rawAnswer: raw, displayAnswer: '', citations, evidenceIds: new Set(['run:a', 'run:b']),
    evidence: ['a', 'b'].map(id => ({ evidence_id: `run:${id}`, upstream_context_id: `pqac-${id}`,
      document_id: 'doc', upstream_dockey: 'fixed-v1', original_text: 'Saved text' })),
    hideDetails: true, pathname: '/', search: '?answer=run' }
}

async function renderAnswer(props) {
  return renderToStaticMarkup(createElement(await component('CitationAnswer.tsx', 'CitationAnswer'), props))
}

test('identical sourced paragraphs are folded with every original citation occurrence retained', async () => {
  const props = answerProps('仅在任务成功后计算成本。（pqac-a）\n\n仅在任务成功后计算成本。（pqac-a）')
  const original = structuredClone(props)
  const html = await renderAnswer(props)
  assert.equal((html.match(/class="answer-reading-fold"/g) || []).length, 1)
  assert.doesNotMatch(html, /<details[^>]* open/)
  assert.equal((html.match(/仅在任务成功后计算成本。/g) || []).length, 2)
  assert.equal((html.match(/data-citation-anchor="true"/g) || []).length, 2)
  assert.equal((html.match(/class="citation-link paper-citation"/g) || []).length, 2)
  assert.deepEqual(props, original)
})

test('different conditions, different sources, and unavailable sources are never treated as duplicates', async () => {
  for (const raw of [
    '仅在成功后评分。（pqac-a）\n\n失败时也评分。（pqac-a）',
    '仅在成功后评分。（pqac-a）\n\n仅在成功后评分。（pqac-b）',
    '仅在成功后评分。（pqac-a）\n\n### 另一方法\n\n仅在成功后评分。（pqac-a）',
  ]) assert.doesNotMatch(await renderAnswer(answerProps(raw)), /class="answer-reading-fold"/)
  const props = answerProps('同一说法（pqac-a）\n\n同一说法（pqac-a）')
  props.citations.forEach(c => { c.evidence_id = null; c.mapping_status = 'missing_context' })
  assert.doesNotMatch(await renderAnswer(props), /class="answer-reading-fold"/)
})

test('short construction labels preserve the distinction from model answer requirements and retain the raw answer', async () => {
  const prefix = '参考答案或数据的构建要求（不等同于模型作答规则）：'
  const props = { ...answerProps(prefix + '答案位于无需登录的公开网站。（pqac-a）'), hideDetails: false }
  const html = await renderAnswer(props)
  assert.match(html, /class="answer-scope-note"/)
  assert.match(html, /资料构建（非作答限制）/)
  assert.match(html, /title="参考答案或数据的构建要求（不等同于模型作答规则）"/)
  assert.match(html, /<p class="raw-answer">参考答案或数据的构建要求/)
  assert.match(html, /答案位于无需登录的公开网站。/)
})

test('a long original criterion remains complete in a collapsed disclosure, including its final exception', async () => {
  const body = 'Each node must pass its evaluator. '.repeat(24) + 'Otherwise the task is not successful.'
  const props = answerProps('原文判定依据（保留限定条件）：\n\n> ' + body + ' (pqac-a)')
  const html = await renderAnswer(props)
  assert.match(html, /<details class="answer-reading-fold"><summary>判定条件原文/)
  assert.match(html, /Otherwise the task is not successful\./)
  assert.match(html, /id="citation-1"/)
  assert.equal(props.rawAnswer.endsWith('Otherwise the task is not successful. (pqac-a)'), true)
})

test('code and mathematical conditions remain literal and are not folded or normalized as prose', async () => {
  const raw = '只有 $x \\le 0$ 时成立。（pqac-a）\n\n`参考答案或数据的构建要求（不等同于模型作答规则）：`\n\n$$\nU=\\frac{1}{1+c+t}\n$$'
  const html = await renderAnswer(answerProps(raw))
  assert.match(html, /<annotation encoding="application\/x-tex">x \\le 0<\/annotation>/)
  assert.match(html, /<annotation encoding="application\/x-tex">U=\\frac\{1\}\{1\+c\+t\}<\/annotation>/)
  assert.match(html, /<code>参考答案或数据的构建要求（不等同于模型作答规则）：<\/code>/)
  assert.doesNotMatch(html, /class="answer-reading-fold"/)
})

test('selected long evidence is initially collapsed; unknown extracted glyphs are marked, not guessed', async () => {
  const EvidencePanel = await component('EvidencePanel.tsx', 'EvidencePanel')
  const originalText = 'P(A|B) = exp\u0012'.concat('condition and value '.repeat(60), '\u0013. Last exception remains.')
  const props = { answerRunId: 'run', selectedEvidenceId: 'run:a', citationNumbers: new Map([['run:a', 1]]),
    evidence: [{ evidence_id: 'run:a', upstream_context_id: 'pqac-a', document_id: 'doc', upstream_dockey: 'fixed-v1',
      original_text: originalText, summary: 'Saved summary', upstream_locator: null, locator_status: 'unknown', pdf_page: null }] }
  const before = structuredClone(props)
  const html = renderToStaticMarkup(createElement(EvidencePanel, props))
  assert.match(html, /<details class="original-full"><summary>/)
  assert.match(html, /exp�/)
  assert.match(html, /Last exception remains\./)
  assert.match(html, /公式与符号请核对 PDF/)
  assert.match(html, /<pre class="saved-extract">P\(A\|B\) = exp\u0012/)
  assert.match(html, /href="\/api\/answers\/run\/evidence\/run%3Aa\/pdf"/)
  assert.deepEqual(props, before)
})

test('format-only page breaks are readable without changing other characters or making up unknown symbols', () => {
  assert.equal(readingText('α ≤ β\fγ\vδ\u0001'), 'α ≤ β\n\nγ\nδ�')
})

test('consecutive criterion quotes share a label while every condition, formula and citation stays available', async () => {
  const label = '原文判定依据（保留限定条件）：'
  const raw = `${label}\n\n> Each node must pass its evaluator. (pqac-a)\n\n${label}\n\n> All required nodes must pass. ES = L\n> P. (pqac-b)`
  const props = answerProps(raw)
  const html = await renderAnswer(props)
  assert.equal((html.match(/原文判定依据（保留限定条件）：/g) || []).length, 1)
  assert.equal((html.match(/<blockquote>/g) || []).length, 2)
  assert.match(html, /Each node must pass its evaluator\./)
  assert.match(html, /All required nodes must pass\. ES = L\nP\./)
  assert.equal((html.match(/data-citation-anchor="true"/g) || []).length, 2)
  assert.match(html, /公式与表格请结合引用 PDF 核对/)
  assert.equal(props.rawAnswer, raw)
})
