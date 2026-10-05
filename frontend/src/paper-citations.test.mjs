import assert from 'node:assert/strict'
import test from 'node:test'
import { build } from 'esbuild'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { citationReturnId, splitCitationText } from './citations.ts'

async function component(file, exportName) {
  const output = await build({entryPoints: [`src/${file}`], bundle: true, platform: 'node', format: 'esm', write: false, logLevel: 'silent'})
  return (await import(`data:text/javascript;base64,${Buffer.from(output.outputFiles[0].text).toString('base64')}`))[exportName]
}

const evidence = ['a', 'b', 'c'].map((id, index) => ({evidence_id:`run:${id}`, upstream_context_id:`pqac-${id}`,
  document_id: index < 2 ? 'doc-ye' : 'doc-lin', upstream_dockey: index < 2 ? '2401.12794v3' : '2305.19187v3',
  summary: `Summary ${id}`, original_text: `Original passage ${id}`, upstream_locator: null, locator_status:'unknown', pdf_page:null}))
const documentSources = {
  'doc-ye':{document_id:'doc-ye', versioned_id:'2401.12794v3', title:'Fixed Ye paper', authors:['Fanghua Ye','Another Author'], year:2024, abstract:'Full saved Ye abstract',abstract_status:'saved'},
  'doc-lin':{document_id:'doc-lin', versioned_id:'2305.19187v3', title:'Fixed Lin paper', authors:['Zhen Lin','Another Author'], year:2023, abstract:'Full saved Lin abstract',abstract_status:'saved'},
}
function props(raw) {
  const chars = Array.from(raw), citations = []
  for(let i=0;i<chars.length;i++) for(const item of evidence) if(chars.slice(i,i+6).join('')===item.upstream_context_id)
    citations.push({upstream_context_id:item.upstream_context_id,start:i,end:i+6,evidence_id:item.evidence_id,mapping_status:'mapped'})
  return {rawAnswer:raw,displayAnswer:'',citations,evidence,evidenceIds:new Set(evidence.map(e=>e.evidence_id)),documentSources,pathname:'/',search:'?answer=run',hideDetails:true}
}

test('paragraph citations combine one paper once while retaining every original occurrence and evidence identity', async()=>{
  const CitationAnswer = await component('CitationAnswer.tsx','CitationAnswer')
  const input=props('🧭第一句（pqac-a）。第二句（pqac-b, pqac-a）。结论（pqac-c）。\n\n下一段（pqac-b）。')
  const before=structuredClone(input.citations)
  const html=renderToStaticMarkup(createElement(CitationAnswer,input))
  assert.equal((html.match(/class="citation-link paper-citation"/g)||[]).length,3)
  assert.equal((html.match(/Ye et al\., 2024<\/a>/g)||[]).length,2)
  assert.match(html, /Lin et al\., 2023<\/a>/)
  assert.match(html, /结论。[\s\S]*class="paragraph-citations"/)
  const anchors=new Set([...html.matchAll(/id="(citation-\d+)"/g)].map(m=>m[1]))
  assert.equal(anchors.size,input.citations.length)
  const parts=splitCitationText(input.rawAnswer,input.citations,input.evidenceIds)
  assert.equal(citationReturnId(parts,'run:b','#evidence','citation-3',anchors),'citation-3')
  assert.match(html,/data-evidence-ids="run:a run:b"/)
  assert.deepEqual(input.citations,before)
})

test('code and formula citation literals remain literal while lists and table cells get their own source links', async()=>{
  const CitationAnswer = await component('CitationAnswer.tsx','CitationAnswer')
  const input=props('`pqac-a`\n\n- **第一项** (pqac-a)\n- 第二项 (pqac-b)\n\n|方法|依据|\n|---|---|\n|X|结论 (pqac-c)|\n\n$$\n\\text{pqac-b}\n$$')
  const html=renderToStaticMarkup(createElement(CitationAnswer,input))
  assert.match(html,/<code>pqac-a<\/code>/)
  assert.match(html,/class="katex-display"/)
  assert.equal((html.match(/class="citation-link paper-citation"/g)||[]).length,3)
  assert.match(html,/<li>[\s\S]*Ye et al\., 2024/)
  assert.match(html,/<td>结论[\s\S]*Lin et al\., 2023/)
  assert.doesNotMatch(html, /id="citation-0"/)
})

test('a mismatched fixed version cannot lend its author or year to another source', async()=>{
  const CitationAnswer = await component('CitationAnswer.tsx','CitationAnswer')
  const input=props('Supported (pqac-a).')
  input.documentSources={'doc-ye':{...documentSources['doc-ye'], versioned_id:'2401.12794v1',authors:['Wrong Author'],year:2099}}
  const html=renderToStaticMarkup(createElement(CitationAnswer,input))
  assert.doesNotMatch(html,/Wrong Author|2099|Ye et al/)
  assert.match(html,/作者年份未保存/)
  assert.match(html,/evidence=run%3Aa/)
})

test('code, math, and existing links preserve original parentheses, separators and spacing', async()=>{
  const CitationAnswer = await component('CitationAnswer.tsx','CitationAnswer')
  const raw = '`f(pqac-a)` and ``(pqac-a;  pqac-b)``\n\n````txt\n```(pqac-a;  pqac-b)```\n````\n\n~~~~~txt\n(pqac-a;  pqac-b)\n~~~~~\n\n$f(pqac-a)$\n\n$$\nf(pqac-a;  pqac-b)\n$$\n\n[(pqac-a;  pqac-b)](https://example.test/f(pqac-a))\n\nSupported (pqac-c).'
  const input = props(raw)
  const html = renderToStaticMarkup(createElement(CitationAnswer,input))
  assert.match(html, /<code>f\(pqac-a\)<\/code>/)
  assert.match(html, /<code>\(pqac-a;  pqac-b\)<\/code>/)
  assert.match(html, /<code class="language-txt">```\(pqac-a;  pqac-b\)```\n<\/code>/)
  assert.match(html, /<code class="language-txt">\(pqac-a;  pqac-b\)\n<\/code>/)
  assert.match(html, /<annotation encoding="application\/x-tex">f\(pqac-a\)<\/annotation>/)
  assert.match(html, /<annotation encoding="application\/x-tex">f\(pqac-a;  pqac-b\)<\/annotation>/)
  assert.match(html, /href="https:\/\/example.test\/f\(pqac-a\)">\(pqac-a;  pqac-b\)<\/a>/)
  assert.equal((html.match(/class="citation-link paper-citation"/g)||[]).length, 1)
  assert.equal(input.rawAnswer, raw)
})

test('legacy and CJK adjacent bold render in prose while protected literals stay unchanged', async()=>{
  const CitationAnswer = await component('CitationAnswer.tsx','CitationAnswer')
  const raw = '用**集合大小（SS） **表达。**旧答案 **保留。用**集合大小（SS）**表达。\n\n`**代码（SS） **表达`\n\n``**多反引号 **表达``\n\n````\n**围栏 **表达\n````\n\n$\\text{**数学 **表达}$\n\n[**链接（SS） **表达](https://example.test)\n\n\\*\\*字面星号\\*\\*'
  const html = renderToStaticMarkup(createElement(CitationAnswer,props(raw)))
  assert.match(html, /用<strong>集合大小（SS）<\/strong> 表达/)
  assert.match(html, /<strong>旧答案<\/strong> 保留/)
  assert.match(html, /用<strong>集合大小（SS）<\/strong>表达/)
  assert.match(html, /<code>\*\*代码（SS） \*\*表达<\/code>/)
  assert.match(html, /<code>\*\*多反引号 \*\*表达<\/code>/)
  assert.match(html, /<code>\*\*围栏 \*\*表达\n<\/code>/)
  assert.match(html, /<annotation encoding="application\/x-tex">\\text\{\*\*数学 \*\*表达\}<\/annotation>/)
  assert.match(html, /href="https:\/\/example.test">\*\*链接（SS） \*\*表达<\/a>/)
  assert.match(html, /<p>\*\*字面星号\*\*<\/p>/)
  assert.doesNotMatch(html, /<strong>字面星号<\/strong>/)
})

test('paired LaTeX delimiters render math without moving saved citation identities', async()=>{
  const CitationAnswer = await component('CitationAnswer.tsx','CitationAnswer')
  const raw = '🧭统计量\\(T_R^2\\)（pqac-a）。\n\n\\[\nT_R^2=\\frac{nm}{n+m}\\sum_{j=1}^{k}x_j^2\n\\]\n\n公式依据（pqac-b）。'
  const input=props(raw), before=structuredClone(input.citations)
  const html=renderToStaticMarkup(createElement(CitationAnswer,input))
  assert.match(html, /class="katex-display"/)
  assert.match(html, /<annotation encoding="application\/x-tex">T_R\^2<\/annotation>/)
  assert.match(html, /\\frac\{nm\}\{n\+m\}\\sum_\{j=1\}\^\{k\}x_j\^2/)
  assert.doesNotMatch(html, /class="katex-error"/)
  assert.equal((html.match(/class="citation-link paper-citation"/g)||[]).length,2)
  assert.match(html, /id="citation-1"/)
  assert.match(html, /id="citation-3"/)
  assert.equal(input.rawAnswer, raw)
  assert.deepEqual(input.citations,before)
})

test('math compatibility leaves code, existing links and undelimited legacy formulas intact', async()=>{
  const CitationAnswer = await component('CitationAnswer.tsx','CitationAnswer')
  const raw = '`\\(x^2\\)`\n\n````txt\n\\[x+y\\]\n````\n\n[\\(link\\)](https://example.test)\n\nT²_R = XᵀRᵀ(RSRᵀ)⁻¹RX（pqac-a）。\n\n普通圆括号 (x+y)，未闭合 \\(z。\n\n$y^2$'
  const html=renderToStaticMarkup(createElement(CitationAnswer,props(raw)))
  assert.match(html, /<code>\\\(x\^2\\\)<\/code>/)
  assert.match(html, /<code class="language-txt">\\\[x\+y\\\]\n<\/code>/)
  assert.match(html, /href="https:\/\/example.test">\(link\)<\/a>/)
  assert.match(html, /T²_R = XᵀRᵀ\(RSRᵀ\)⁻¹RX/)
  assert.match(html, /普通圆括号 \(x\+y\)，未闭合 \(z/)
  assert.equal((html.match(/class="katex"/g)||[]).length,1)
  assert.equal((html.match(/class="citation-link paper-citation"/g)||[]).length,1)
})

test('citation literals inside compatible math remain literal and return targets stay in prose', async()=>{
  const CitationAnswer = await component('CitationAnswer.tsx','CitationAnswer')
  const raw = '\\(\\text{pqac-a}\\)\n\n\\[\\text{pqac-b}\\]\n\n正文（pqac-b）。'
  const input=props(raw)
  const html=renderToStaticMarkup(createElement(CitationAnswer,input))
  assert.match(html, /<annotation encoding="application\/x-tex">\\text\{pqac-a\}<\/annotation>/)
  assert.match(html, /<annotation encoding="application\/x-tex">\\text\{pqac-b\}<\/annotation>/)
  assert.doesNotMatch(html, /class="katex-error"|\uE000|id="citation-0"|id="citation-2"/)
  assert.match(html, /id="citation-5"/)
  assert.equal((html.match(/class="citation-link paper-citation"/g)||[]).length,1)
})

test('source table groups paper identity once and retains full abstract and separately navigable excerpts',async()=>{
  const EvidencePanel=await component('EvidencePanel.tsx','EvidencePanel')
  const html=renderToStaticMarkup(createElement(EvidencePanel,{answerRunId:'run',evidence,documentSources,
    citationNumbers:new Map([['run:a',1],['run:b',2],['run:c',3]]),selectedEvidenceId:'run:b',returnCitationId:'citation-3',open:true}))
  assert.match(html,/查看 2 篇引用论文/)
  assert.equal((html.match(/class="paper-source-row/g)||[]).length,2)
  assert.equal((html.match(/<h3>Fixed Ye paper<\/h3>/g)||[]).length,1)
  assert.match(html,/Full saved Ye abstract/)
  assert.match(html,/<details class="source-abstract">/)
  assert.match(html,/Original passage a/)
  assert.match(html,/Original passage b/)
  assert.match(html,/data-evidence-id="run:a"/)
  assert.match(html,/data-evidence-id="run:b"/)
  assert.match(html,/colSpan="2"|colspan="2"/)
  assert.match(html,/id="evidence"/)
  assert.match(html,/href="#citation-3"/)
})
