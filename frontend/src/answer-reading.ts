import type { Root } from 'mdast'

type ReadingNode = {
  type: string
  value?: string
  url?: string
  children?: ReadingNode[]
  data?: { hName?: string; hProperties?: Record<string, unknown> }
}
type ReadingGroup = { items: readonly { kind: string; evidenceId?: string }[] }

const constructionPrefix = '参考答案或数据的构建要求（不等同于模型作答规则）：'
const unreadable = /[\u0000-\u0008\u000e-\u001f\u007f\ufffd]/

// An unknown PDF glyph may represent a mathematical symbol. Mark it explicitly;
// never guess the missing symbol or silently remove it. Stored text is untouched.
export function readingText(text: string) {
  return text.replace(/\r\n?/g, '\n').replace(/\f/g, '\n\n').replace(/\v/g, '\n')
    .replace(/[\u0000-\u0008\u000e-\u001f\u007f]/g, '\ufffd')
}

export function hasUnreadableSymbols(text: string) { return unreadable.test(text) }

function prose(node: ReadingNode, groups: ReadonlyMap<string, ReadingGroup>): string {
  if (node.url && groups.has(node.url)) return ''
  return node.value ?? node.children?.map(child => prose(child, groups)).join('') ?? ''
}

function containsLiteral(node: ReadingNode): boolean {
  return ['code', 'inlineCode', 'math', 'inlineMath', 'image', 'html'].includes(node.type) ||
    !!node.children?.some(containsLiteral)
}

function duplicateKey(node: ReadingNode, groups: ReadonlyMap<string, ReadingGroup>) {
  if (node.type !== 'paragraph' || containsLiteral(node)) return null
  const ids = new Set<string>()
  let unavailable = false
  const collect = (item: ReadingNode) => {
    const references = item.url && groups.get(item.url)
    if (references) for (const ref of references.items) {
      if (ref.kind !== 'link' || !ref.evidenceId) unavailable = true
      else ids.add(ref.evidenceId)
    }
    item.children?.forEach(collect)
  }
  collect(node)
  // Fold only identical prose with exactly the same verified sources. Similar
  // statements, different conditions, and different references remain visible.
  if (!ids.size || unavailable) return null
  return JSON.stringify([prose(node, groups).replace(/[ \t\r\n]+/g, ' ').trim(), [...ids].sort()])
}

function disclosure(label: string, children: ReadingNode[]): ReadingNode {
  return { type: 'readingDisclosure', data: { hName: 'details', hProperties: { className: 'answer-reading-fold' } },
    children: [{ type: 'readingSummary', data: { hName: 'summary' }, children: [{ type: 'text', value: label }] }, ...children] }
}

// Runs after raw citation offsets have been checked and every occurrence has its
// own anchor. Only this render tree changes; the saved answer/evidence do not.
export function prepareAnswerReading(tree: Root, groups: ReadonlyMap<string, ReadingGroup>) {
  const root = tree as unknown as ReadingNode
  const seen = new Set<string>()
  const children: ReadingNode[] = []
  for (const node of root.children || []) {
    if (node.type === 'heading') seen.clear()
    const key = duplicateKey(node, groups)
    if (key && seen.has(key)) {
      children.push(disclosure('重复的支持说明（展开查看）', [node]))
      continue
    }
    if (key) seen.add(key)
    const previous = children[children.length - 1]
    const label = previous?.type === 'paragraph' ? prose(previous, groups).trim() : ''
    if (node.type === 'blockquote' && !containsLiteral(node) && Array.from(prose(node, groups)).length > 600 &&
        ['原文判定依据（保留限定条件）：', '参考资料原文（用途与判分条件分开）：'].includes(label)) {
      children.pop()
      children.push(disclosure(label.startsWith('原文判定')
        ? '判定条件原文（展开查看完整条件）' : '参考资料原文（非判分条件，展开查看）', [node]))
    } else children.push(node)
  }
  // Consecutive source quotes share one purpose label. Keep every original
  // quote and occurrence anchor in place; only identical UI labels are grouped.
  const grouped: ReadingNode[] = []
  for (let index = 0; index < children.length; index++) {
    const node = children[index], next = children[index + 1]
    const label = node.type === 'paragraph' ? prose(node, groups).trim() : ''
    if (next?.type === 'blockquote' &&
        ['原文判定依据（保留限定条件）：', '参考资料原文（用途与判分条件分开）：'].includes(label)) {
      const previousGroup = grouped[grouped.length - 1]
      if (previousGroup?.type === 'readingRules' && prose(previousGroup.children![0], groups).trim() === label)
        previousGroup.children!.push(next)
      else {
        const wrapper: ReadingNode = { type: 'readingRules', data: { hName: 'div',
          hProperties: { className: 'answer-rule-group' } }, children: [node,
          { type: 'paragraph', children: [{ type: 'text', value: '以下为论文提取原文；公式与表格请结合引用 PDF 核对。' }] }, next] }
        grouped.push(wrapper)
      }
      index++
    } else grouped.push(node)
  }
  root.children = grouped
  const clean = (node: ReadingNode) => {
    if (['code', 'inlineCode', 'math', 'inlineMath', 'link', 'image', 'html'].includes(node.type)) return
    if (node.type === 'text' && node.value) node.value = readingText(node.value)
    if (node.type === 'paragraph' && node.children?.[0]?.type === 'text' &&
        node.children[0].value?.startsWith(constructionPrefix)) {
      node.children[0].value = node.children[0].value.slice(constructionPrefix.length)
      node.children.unshift({ type: 'readingScope', data: { hName: 'span', hProperties: {
        className: 'answer-scope-note', title: constructionPrefix.slice(0, -1),
      } }, children: [{ type: 'text', value: '资料构建（非作答限制）' }] })
    }
    node.children?.forEach(clean)
  }
  clean(root)
}
