import type { Root } from 'mdast'
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import { answerDisplayParts, type AnswerDisplayPart } from './answer-display'
import type { CitationPart } from './citations'

type Group = Extract<AnswerDisplayPart, { kind: 'references' }> & { role?: 'origin' | 'paragraph' }
type TreeNode = { type: string; value?: string; children?: TreeNode[]; url?: string; data?: { hChildren?: TreeNode[] }; position?: {start:{offset?:number};end:{offset?:number}} }

const displayParser = unified().use(remarkParse).use(remarkGfm).use(remarkMath)

function normalizeMathDelimiters(source: string) {
  const protectedRanges: [number, number][] = []
  const collect = (node: TreeNode) => {
    if (['code','inlineCode','math','inlineMath','link','image','definition','html'].includes(node.type)) {
      const start = node.position?.start.offset, end = node.position?.end.offset
      if (start !== undefined && end !== undefined) protectedRanges.push([start,end])
      return
    }
    node.children?.forEach(collect)
  }
  collect(displayParser.parse(source) as unknown as TreeNode)
  const convert = (value: string) => value.replace(
    /(?<!\\)\\\(([\s\S]*?)(?<!\\)\\\)|(?<!\\)\\\[([\s\S]*?)(?<!\\)\\\]/g,
    (original, inline: string | undefined, block: string | undefined) => {
      const content = (inline ?? block ?? '').trim()
      if (!content) return original
      return inline !== undefined ? '$' + content + '$' : '\n\n$$\n' + content + '\n$$\n\n'
    })
  let cursor = 0, result = ''
  for (const [start,end] of protectedRanges.sort((a,b)=>a[0]-b[0])) {
    if (start < cursor) continue
    result += convert(source.slice(cursor,start)) + source.slice(start,end)
    cursor = end
  }
  return result + convert(source.slice(cursor))
}

function normalizeProseBold(source: string) {
  // Let the same Markdown grammar find protected ranges, including arbitrary
  // backtick/tilde fence lengths, indented code, links and mathematical blocks.
  const protectedRanges: [number,number][] = []
  const collect = (node: TreeNode) => {
    if (['code','inlineCode','math','inlineMath','link','image','definition','html'].includes(node.type)) {
      const start = node.position?.start.offset, end = node.position?.end.offset
      if (start !== undefined && end !== undefined) protectedRanges.push([start,end])
      return
    }
    node.children?.forEach(collect)
  }
  collect(displayParser.parse(source) as unknown as TreeNode)
  let gap = '\u200b\u200b'
  while (source.includes(gap)) gap += '\u200b'
  const gapEntity = '&#x200b;'.repeat(gap.length)
  const repair = (value: string) => value
    .replace(/(?<![\\*])(\*\*(?=\S)(?:(?!\*\*)[^\n])+?)([ \t]+)\*\*(?!\*)/g, '$1**$2')
    // An entity begins with punctuation, allowing CommonMark to close bold
    // after CJK punctuation. Remove only our unique decoded gap in prose AST.
    .replace(/(?<![\\*])(\*\*(?=\S)(?:(?!\*\*)[^\n])+?[\p{P}\p{S}])\*\*(?=[\p{L}\p{N}])/gu, `$1**${gapEntity}`)
  let cursor = 0, markdown = ''
  for (const [start,end] of protectedRanges.sort((a,b)=>a[0]-b[0])) {
    if (start < cursor) continue
    markdown += repair(source.slice(cursor,start)) + source.slice(start,end)
    cursor = end
  }
  markdown += repair(source.slice(cursor))
  return {markdown,gap}
}

// Markers are local to this render and cannot be supplied by the answer. The saved
// code-point offsets are validated before this presentation-only Markdown copy.
export function citationMarkdown(parts: CitationPart[], paragraphEnd = false) {
  const text = parts.map(part => part.text).join('')
  let prefix = 'papertrail-citation-'
  while (text.includes(prefix)) prefix += 'x-'
  const groups = new Map<string, Group>()
  const source = parts.map((part, partIndex) => {
    if (part.kind === 'text') return part.text
    const key = `#${prefix}${groups.size}`
    groups.set(key, {kind:'references',items:[{...part,partIndex}]})
    return `\uE000${key}\uE001`
  }).join('')
  const pattern = /\uE000(#[^\uE001]+)\uE001/g
  const restore = (value: string) => value.replace(pattern, (marker, key: string) =>
    groups.get(key)?.items[0].text || marker)
  const {markdown,gap} = normalizeProseBold(normalizeMathDelimiters(source))
  const remarkCitations = () => (tree: Root) => {
    const proseCitations = (node: TreeNode): TreeNode[] => {
      if (node.value) node.value = node.value.replaceAll(gap,'')
      if (!node.value?.includes('\uE000')) return [node]
      const local: (CitationPart & {partIndex?:number})[] = []
      let offset = 0
      for (const match of node.value.matchAll(pattern)) {
        const item = groups.get(match[1])?.items[0]
        if (!item) continue
        if (match.index! > offset) local.push({kind:'text',text:node.value.slice(offset,match.index)})
        local.push(item)
        offset = match.index! + match[0].length
      }
      if (!offset) return [node]
      if (offset < node.value.length) local.push({kind:'text',text:node.value.slice(offset)})
      // Parentheses and separators may be removed only after the parser has
      // established that this is ordinary prose, preserving protected literals.
      return answerDisplayParts(local).flatMap(part => {
        if (part.kind === 'text') return [{type:'text',value:part.text}]
        const nodes: TreeNode[] = []
        const valid = part.items.filter(item => item.kind === 'link')
        const unavailable = part.items.filter(item => item.kind === 'unavailable')
        if (paragraphEnd && valid.length) {
          const key = `#${prefix}origin-${groups.size}`
          groups.set(key, {kind:'references',items:valid,role:'origin'})
          nodes.push({type:'link',url:key,children:[]})
        }
        const remaining = paragraphEnd ? unavailable : part.items
        if (remaining.length) {
          const key = `#${prefix}references-${groups.size}`
          groups.set(key, {kind:'references',items:remaining})
          nodes.push({type:'link',url:key,children:[]})
        }
        return nodes
      })
    }
    const walk = (node: TreeNode) => {
      if (node.type !== 'text' && node.value) node.value = restore(node.value)
      // remark-math also copies its source into prebuilt HAST children.
      // Restore there before rehype-katex reads it, retaining literal code/math IDs.
      const restoreHast = (child: TreeNode) => { if (child.value) child.value = restore(child.value); child.children?.forEach(restoreHast) }
      node.data?.hChildren?.forEach(restoreHast)
      if (node.url) node.url = restore(node.url)
      if (!node.children) return
      if (node.type === 'link' || node.type === 'image') {
        // A saved literal inside an existing Markdown link must not form nested links.
        const literal = (child: TreeNode) => { if (child.value) child.value = restore(child.value); child.children?.forEach(literal) }
        node.children.forEach(literal)
        return
      }
      node.children = node.children.flatMap(child => {
        if (child.type !== 'text' || !child.value) { walk(child); return [child] }
        return proseCitations(child)
      })
      if (paragraphEnd && ['paragraph', 'tableCell', 'heading'].includes(node.type)) {
        const items: Group['items'] = []
        const collect = (child: TreeNode) => {
          const group = child.url && groups.get(child.url)
          if (group && group.role === 'origin') items.push(...group.items)
          child.children?.forEach(collect)
        }
        node.children.forEach(collect)
        if (items.length) {
          const key = `#${prefix}paragraph-${groups.size}`
          groups.set(key, {kind:'references',items,role:'paragraph'})
          node.children.push({type:'link',url:key,children:[]})
        }
      }
    }
    walk(tree as unknown as TreeNode)
  }
  return { markdown, groups, remarkCitations }
}
