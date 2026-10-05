import assert from 'node:assert/strict'
import test from 'node:test'
import { build } from 'esbuild'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { citationReturnId, splitCitationText } from './citations.ts'

async function component(file, exportName) {
  const output = await build({
    entryPoints: [`src/${file}`], bundle: true, platform: 'node', format: 'esm',
    write: false, logLevel: 'silent',
  })
  const url = `data:text/javascript;base64,${Buffer.from(output.outputFiles[0].text).toString('base64')}`
  return (await import(url))[exportName]
}

test('a refreshed evidence link returns to a rendered prose citation after an earlier code occurrence', async () => {
  const CitationAnswer = await component('CitationAnswer.tsx', 'CitationAnswer')
  const raw = '`pqac-a` is literal code. Prose pqac-a.'
  const citations = [...raw.matchAll(/pqac-a/g)].map(match => ({
    upstream_context_id: 'pqac-a', start: match.index, end: match.index + 6,
    evidence_id: 'run:a', mapping_status: 'mapped',
  }))
  const evidenceIds = new Set(['run:a'])
  const html = renderToStaticMarkup(createElement(CitationAnswer, {
    rawAnswer: raw, displayAnswer: '', citations, evidenceIds,
    pathname: '/', search: '?answer=run&evidence=run%3Aa', hideDetails: true,
  }))
  const renderedIds = new Set([...html.matchAll(/id="(citation-\d+)"/g)].map(match => match[1]))
  assert.deepEqual([...renderedIds], ['citation-3'])
  assert.equal(citationReturnId(splitCitationText(raw, citations, evidenceIds), 'run:a', '#evidence', '', renderedIds), 'citation-3')
  assert.equal(citationReturnId(splitCitationText(raw, citations, evidenceIds), 'run:a', '#citation-1', '', renderedIds), 'citation-3')
})

test('bold headings and superscript citation groups preserve evidence links, prose and raw data', async () => {
  const CitationAnswer = await component('CitationAnswer.tsx', 'CitationAnswer')
  const raw = '🧭**结论。 ** 内容（pqac-a, pqac-b）。数学式（x+y）保留。<img src=x onerror=alert(1)>'
  const citations = ['pqac-a', 'pqac-b'].map(id => {
    const start = Array.from(raw.slice(0, raw.indexOf(id))).length
    return { upstream_context_id: id, start, end: start + id.length, evidence_id: `run:${id}`, mapping_status: 'mapped' }
  })
  const before = structuredClone(citations)
  const html = renderToStaticMarkup(createElement(CitationAnswer, {
    rawAnswer: raw, displayAnswer: '', citations, hideDetails: true,
    evidenceIds: new Set(citations.map(c => c.evidence_id)), pathname: '/', search: '?answer=run',
  }))
  assert.match(html, /<strong>结论。<\/strong> /)
  assert.match(html, /<sup class="citation-group">[\s\S]*href="\/\?answer=run&amp;evidence=run%3Apqac-a&amp;citation=citation-1#evidence"[\s\S]*\[1\][\s\S]*\[2\][\s\S]*<\/sup>/)
  const plain = html.replace(/<[^>]*>/g, '')
  assert.match(plain, /内容。\[1\]\[2\]数学式（x\+y）保留/)
  assert.doesNotMatch(plain, /\*\*|（\[1\]|\[2\]）/)
  assert.doesNotMatch(html, /<img|onerror="/)
  assert.match(html, /&lt;img/)
  assert.deepEqual(citations, before)
})

test('bold can span a verified citation while explanatory parentheses and unmatched markers remain', async () => {
  const CitationAnswer = await component('CitationAnswer.tsx', 'CitationAnswer')
  const raw = '**证据 (pqac-a)** 是依据。说明（解释 pqac-a）保留。未闭合**文字'
  const first = raw.indexOf('pqac-a'), second = raw.indexOf('pqac-a', first + 1)
  const citations = [first, second].map(start => ({
    upstream_context_id: 'pqac-a', start, end: start + 6, evidence_id: 'run:pqac-a', mapping_status: 'mapped',
  }))
  const html = renderToStaticMarkup(createElement(CitationAnswer, {
    rawAnswer: raw, displayAnswer: '', citations, hideDetails: true,
    evidenceIds: new Set(['run:pqac-a']), pathname: '/', search: '?answer=run',
  }))
  assert.match(html, /<strong>证据<sup class="citation-group">[\s\S]*\[1\]<\/a><\/sup><\/strong>/)
  assert.match(html.replace(/<[^>]*>/g, ''), /说明（解释 \[1\]）保留。未闭合\*\*文字/)
  assert.equal((html.match(/data-evidence-id="run:pqac-a"/g) || []).length, 2)
})

test('the answer exposes inline evidence links even without References', async () => {
  const CitationAnswer = await component('CitationAnswer.tsx', 'CitationAnswer')
  const html = renderToStaticMarkup(createElement(CitationAnswer, {
    rawAnswer: 'A pqac-a B pqac-b.', displayAnswer: 'Question: Q\n\nA pqac-a B pqac-b.',
    citations: [
      { upstream_context_id: 'pqac-a', start: 2, end: 8, evidence_id: 'run:pqac-a', mapping_status: 'mapped' },
      { upstream_context_id: 'pqac-b', start: 11, end: 17, evidence_id: null, mapping_status: 'missing_context' },
    ],
    evidenceIds: new Set(['run:pqac-a']), pathname: '/', search: '?answer=run',
  }))
  assert.match(html, /href="\/\?answer=run&amp;evidence=run%3Apqac-a&amp;citation=citation-1#evidence"[^>]*>\[1\]<\/a>/)
  assert.match(html, /pqac-b/)
  assert.doesNotMatch(html, /href="[^"]+pqac-b/)
  assert.match(html, /Question: Q/)
})

test('Markdown lists, GFM tables and inline/display maths coexist with verified Unicode citations', async () => {
  const CitationAnswer = await component('CitationAnswer.tsx', 'CitationAnswer')
  const raw = '# 结果\n\n- 🧭**第一项** $G=(V,E)$ pqac-a\n- 第二项 pqac-a\n\n|方法|结果|\n|---|---|\n|A|~~旧值~~ 新值|\n\n$$\n\\frac{x}{y}=2\n$$\n\n```text\n- 原样代码\n```'
  const chars = Array.from(raw)
  const citations = []
  for (let i = 0; i < chars.length; i++) if (chars.slice(i, i + 6).join('') === 'pqac-a') {
    citations.push({ upstream_context_id: 'pqac-a', start: i, end: i + 6, evidence_id: 'run:a', mapping_status: 'mapped' })
  }
  const before = structuredClone(citations)
  const html = renderToStaticMarkup(createElement(CitationAnswer, {
    rawAnswer: raw, displayAnswer: '', citations, evidenceIds: new Set(['run:a']),
    pathname: '/', search: '?answer=run', hideDetails: true,
  }))
  assert.match(html, /<h1>结果<\/h1>/)
  assert.match(html, /<ul>[\s\S]*<li>/)
  assert.match(html, /<table>/)
  assert.match(html, /<del>旧值<\/del>/)
  assert.match(html, /class="katex"/)
  assert.match(html, /class="katex-display"/)
  assert.match(html, /<pre><code class="language-text">- 原样代码/)
  assert.equal((html.match(/data-evidence-id="run:a"/g) || []).length, 2)
  assert.match(html, /id="citation-1"/)
  assert.match(html, /id="citation-3"/)
  assert.equal((html.match(/>\[1\]<\/a>/g) || []).length, 2)
  assert.doesNotMatch(html, /点击引用编号，核查/)
  assert.deepEqual(citations, before)
})

test('evidence is one collapsed disclosure by default and opens explicitly for a citation deep link', async () => {
  const EvidencePanel = await component('EvidencePanel.tsx', 'EvidencePanel')
  const props = { answerRunId: 'run', evidence: [], selectedEvidenceId: '', citationNumbers: new Map() }
  const collapsed = renderToStaticMarkup(createElement(EvidencePanel, props))
  assert.match(collapsed, /<details class="evidence-panel paper-sources"/)
  assert.doesNotMatch(collapsed, /<details class="evidence-panel paper-sources"[^>]* open/)
  assert.match(collapsed, /<summary[\s\S]*查看 0 篇引用论文/)
  assert.match(renderToStaticMarkup(createElement(EvidencePanel, {...props, open: true})), /<details class="evidence-panel paper-sources"[^>]* open/)
})

test('the selected evidence exposes saved summary and original text', async () => {
  const EvidencePanel = await component('EvidencePanel.tsx', 'EvidencePanel')
  const html = renderToStaticMarkup(createElement(EvidencePanel, {
    answerRunId: 'run',
    selectedEvidenceId: 'run:pqac-a',
    citationNumbers: new Map([['run:pqac-a', 1]]),
    evidence: [{
      evidence_id: 'run:pqac-a', upstream_context_id: 'pqac-a',
      document_id: 'doc-1', upstream_dockey: '2503.07919v3',
      summary: 'Saved model summary', original_text: 'Actual source passage',
      upstream_locator: 'pages 3-4', locator_status: 'unverified_upstream_label', pdf_page: null,
    }],
  }))
  assert.match(html, /id="evidence"/)
  assert.match(html, /pqac-a/)
  assert.match(html, /Saved model summary/)
  assert.match(html, /Actual source passage/)
  assert.match(html, /页码未知/)
  assert.match(html, /href="\/api\/answers\/run\/evidence\/run%3Apqac-a\/pdf"/)
  assert.match(html, /href="\/api\/answers\/run\/evidence\/run%3Apqac-a\/pdf\?download=1"/)
  assert.match(html, /下载 PDF（本机查看）/)
  assert.doesNotMatch(html, /#page=/)
  assert.ok(html.indexOf('Actual source passage') < html.indexOf('固定版本 PDF（新标签页）'))
})

test('a unique PDF text match exposes its physical page without changing saved evidence', async () => {
  const EvidencePanel = await component('EvidencePanel.tsx', 'EvidencePanel')
  const html = renderToStaticMarkup(createElement(EvidencePanel, {
    answerRunId: 'run', selectedEvidenceId: 'run:pqac-a', citationNumbers: new Map([['run:pqac-a', 1]]),
    evidence: [{ evidence_id: 'run:pqac-a', upstream_context_id: 'pqac-a',
      document_id: 'doc-1', upstream_dockey: '2503.07919v3',
      summary: 'Saved summary', original_text: 'Saved original',
      upstream_locator: 'pages 3-4', locator_status: 'unverified_upstream_label', pdf_page: null }],
    pageLookup: { status: 'ready', location: {
      evidence_id: 'run:pqac-a', pdf_sha256: 'fixed-hash', status: 'unique',
      matches: [{ start_page: 14, end_page: 15 }],
    } },
  }))
  assert.match(html, /PDF 物理第 14–15 页/)
  assert.match(html, /提取文字完整匹配/)
  assert.match(html, /href="\/api\/answers\/run\/evidence\/run%3Apqac-a\/pdf#page=14"/)
  assert.match(html, /上游位置标签：pages 3-4（未核准为 PDF 物理页码）/)
})

test('multiple PDF matches offer each candidate and never pick one automatically', async () => {
  const EvidencePanel = await component('EvidencePanel.tsx', 'EvidencePanel')
  const html = renderToStaticMarkup(createElement(EvidencePanel, {
    answerRunId: 'run', selectedEvidenceId: 'run:pqac-a', citationNumbers: new Map([['run:pqac-a', 1]]),
    evidence: [{ evidence_id: 'run:pqac-a', upstream_context_id: 'pqac-a',
      document_id: 'doc-1', upstream_dockey: 'paper-v1',
      summary: 'Saved summary', original_text: 'Saved original',
      upstream_locator: null, locator_status: 'unknown', pdf_page: null }],
    pageLookup: { status: 'ready', location: {
      evidence_id: 'run:pqac-a', pdf_sha256: 'fixed-hash', status: 'multiple',
      matches: [{ start_page: 2, end_page: 2 }, { start_page: 7, end_page: 8 }],
    } },
  }))
  assert.match(html, /PDF 多处匹配（2 处）/)
  assert.match(html, /href="\/api\/answers\/run\/evidence\/run%3Apqac-a\/pdf#page=2"/)
  assert.match(html, /href="\/api\/answers\/run\/evidence\/run%3Apqac-a\/pdf#page=7"/)
  assert.match(html, /href="\/api\/answers\/run\/evidence\/run%3Apqac-a\/pdf"/)
})

test('an unmatched PDF text keeps the physical page unknown', async () => {
  const EvidencePanel = await component('EvidencePanel.tsx', 'EvidencePanel')
  const html = renderToStaticMarkup(createElement(EvidencePanel, {
    answerRunId: 'run', selectedEvidenceId: 'run:pqac-a', citationNumbers: new Map([['run:pqac-a', 1]]),
    evidence: [{ evidence_id: 'run:pqac-a', upstream_context_id: 'pqac-a',
      document_id: 'doc-1', upstream_dockey: 'paper-v1',
      summary: 'Saved summary', original_text: 'Saved original',
      upstream_locator: null, locator_status: 'unknown', pdf_page: null }],
    pageLookup: { status: 'ready', location: {
      evidence_id: 'run:pqac-a', pdf_sha256: 'fixed-hash', status: 'unknown', matches: [],
    } },
  }))
  assert.match(html, /PDF 物理页码未知/)
  assert.doesNotMatch(html, /#page=/)
})

test('evidence without a bound document does not offer a PDF link', async () => {
  const EvidencePanel = await component('EvidencePanel.tsx', 'EvidencePanel')
  const html = renderToStaticMarkup(createElement(EvidencePanel, {
    answerRunId: 'run', selectedEvidenceId: 'run:pqac-a', citationNumbers: new Map([['run:pqac-a', 1]]),
    evidence: [{ evidence_id: 'run:pqac-a', upstream_context_id: 'pqac-a',
      document_id: null, upstream_dockey: 'unknown', summary: 'Saved summary',
      original_text: 'Original', upstream_locator: null,
      locator_status: 'unknown', pdf_page: null }],
  }))
  assert.match(html, /固定版本 PDF 无法定位/)
  assert.doesNotMatch(html, /href="\/api\/answers\/[^\"]+\/pdf/)
})

test('a mapped citation whose evidence was not loaded is labelled unavailable', async () => {
  const CitationAnswer = await component('CitationAnswer.tsx', 'CitationAnswer')
  const html = renderToStaticMarkup(createElement(CitationAnswer, {
    rawAnswer: 'See pqac-a.', displayAnswer: '',
    citations: [{ upstream_context_id: 'pqac-a', start: 4, end: 10,
      evidence_id: 'run:pqac-a', mapping_status: 'mapped' }],
    evidenceIds: new Set(), pathname: '/', search: '?answer=run',
  }))
  assert.match(html, /\[1\]（保存证据未载入）<\/span>/)
  assert.doesNotMatch(html, /href=/)
})

test('an invalid saved offset is reported instead of making a guessed link', async () => {
  const CitationAnswer = await component('CitationAnswer.tsx', 'CitationAnswer')
  const html = renderToStaticMarkup(createElement(CitationAnswer, {
    rawAnswer: 'See pqac-a.', displayAnswer: '',
    citations: [{ upstream_context_id: 'pqac-a', start: 3, end: 9,
      evidence_id: 'run:pqac-a', mapping_status: 'mapped' }],
    evidenceIds: new Set(['run:pqac-a']), pathname: '/', search: '?answer=run',
  }))
  assert.match(html, /1 条引用位置无法核对/)
  assert.doesNotMatch(html, /href=/)
})

test('the paper table keeps separate excerpt links and expands only the selected passage', async () => {
  const EvidencePanel = await component('EvidencePanel.tsx', 'EvidencePanel')
  const html = renderToStaticMarkup(createElement(EvidencePanel, {
    answerRunId: 'run', selectedEvidenceId: 'run:b', pathname: '/', search: '?answer=run',
    citationNumbers: new Map([['run:a', 1], ['run:b', 2]]),
    evidence: [
      { evidence_id: 'run:b', upstream_context_id: 'pqac-b', document_id: 'doc-1',
        upstream_dockey: 'paper-v1', summary: 'Chosen summary', original_text: 'Chosen source',
        upstream_locator: null, locator_status: 'unknown', pdf_page: null },
      { evidence_id: 'run:a', upstream_context_id: 'pqac-a', document_id: 'doc-1',
        upstream_dockey: 'paper-v1', summary: 'Other summary', original_text: 'Other source',
        upstream_locator: null, locator_status: 'unknown', pdf_page: null },
    ],
  }))
  assert.match(html, /data-evidence-id="run:a" href="\/\?answer=run&amp;evidence=run%3Aa#evidence"/)
  assert.match(html, /data-evidence-id="run:b" href="\/\?answer=run&amp;evidence=run%3Ab#evidence"/)
  const selected = html.match(/<article[^>]*id="evidence"[\s\S]*?<\/article>/)?.[0] || ''
  assert.match(selected, /引用原文/)
  assert.match(selected, /Chosen source/)
  assert.doesNotMatch(selected, /Other summary|Other source/)
  assert.match(html, /Other source/)
})

test('a long original has a short preview while the unmodified complete passage remains available', async () => {
  const EvidencePanel = await component('EvidencePanel.tsx', 'EvidencePanel')
  const original = 'First line\nSecond line\n\n' + 'Exact saved words 🧭 '.repeat(75) + '\nSaved ending.'
  const before = original
  const html = renderToStaticMarkup(createElement(EvidencePanel, {
    answerRunId: 'run', selectedEvidenceId: 'run:a', citationNumbers: new Map([['run:a', 1]]),
    evidence: [{ evidence_id: 'run:a', upstream_context_id: 'pqac-a', document_id: 'doc-1',
      upstream_dockey: 'paper-v1', summary: 'Summary', original_text: original,
      upstream_locator: 'pages 1-2', locator_status: 'unknown', pdf_page: null }],
  }))
  const preview = html.match(/<p class="original-preview">([\s\S]*?)<\/p>/)?.[1] || ''
  assert.ok(preview.startsWith('First line\nSecond line'))
  assert.ok(preview.length < original.length / 2)
  assert.doesNotMatch(preview, /Saved ending/)
  assert.ok(html.includes(original))
  assert.match(html, /<details class="original-full" open=""/)
  assert.equal(original, before)
})

test('other saved fragments remain separate collapsed records with real document titles', async () => {
  const EvidencePanel = await component('EvidencePanel.tsx', 'EvidencePanel')
  const evidence = ['a', 'b', 'c'].map((id) => ({ evidence_id: `run:${id}`,
    upstream_context_id: id, document_id: 'doc', upstream_dockey: 'paper-v2',
    summary: `Summary ${id}`, original_text: 'Similar but preserved text',
    upstream_locator: null, locator_status: 'unknown', pdf_page: null }))
  const html = renderToStaticMarkup(createElement(EvidencePanel, {
    answerRunId: 'run', selectedEvidenceId: 'run:a', citationNumbers: new Map([['run:a', 1]]),
    evidence, documentTitles: { doc: 'Actual saved title' },
  }))
  assert.match(html, /其他保存的片段（2）/)
  assert.equal((html.match(/<details class="other-fragment">/g) || []).length, 2)
  assert.match(html, /片段 1 · Actual saved title/)
  assert.match(html, /片段 2 · Actual saved title/)
  assert.match(html, /Summary b/)
  assert.match(html, /Summary c/)
  assert.doesNotMatch(html, /<details[^>]* open/)
})

test('diagnostic grouping keeps every occurrence and its saved offsets', async () => {
  const CitationDetails = await component('CitationAnswer.tsx', 'CitationDetails')
  const html = renderToStaticMarkup(createElement(CitationDetails, {
    rawAnswer: 'pqac-a pqac-a', displayAnswer: 'Original formatted answer',
    citations: [0, 7].map((start) => ({ upstream_context_id: 'pqac-a', start,
      end: start + 6, evidence_id: 'run:a', mapping_status: 'mapped' })),
    evidenceIds: new Set(['run:a']), pathname: '/', search: '?answer=run',
  }))
  assert.match(html, /pqac-a · 2 次出现/)
  assert.match(html, /位置 0–6/)
  assert.match(html, /位置 7–13/)
  assert.equal((html.match(/正文 \[1\]/g) || []).length, 2)
  assert.match(html, /Original formatted answer/)
})
