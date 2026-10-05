export type KeywordQuery = { label: string; groups: string[][]; exclude: string[] }
export type MultiQueryPlan = { version: 1; baseline_query_text: string; expansions: KeywordQuery[] }
export type ExpansionSuggestion = {
  suggestion_id: string; question: string; query_text: string; query_plan: MultiQueryPlan
  model: string; prompt_version: string; prompt_sha256: string; elapsed_ms: number
  usage: { total_tokens?: number }; cost_usd: number | null
}

const word = /^[A-Za-z0-9][A-Za-z0-9._+\-]*$/
const reserved = /^(and|or|andnot)$/i
export const normalizedKeywords = (value: string) => value.trim().split(/\s+/).join(' ')

function simpleWords(value: string, maximum: number): string {
  if (typeof value !== 'string' || !value.trim()) throw new Error('请填写有效的英文表达。')
  const normalized = normalizedKeywords(value)
  const terms = normalized.split(' ')
  if (terms.length > maximum || terms.some(t => !word.test(t) || reserved.test(t))) {
    throw new Error('仅填写简单英文表达；同义表达用 | 分隔，不要填写网址或检索语法。')
  }
  return normalized
}

function identity(groups: string[][], exclude: string[]) {
  const concepts = [...new Set(groups.map(g => JSON.stringify(g.map(p => p.toLowerCase()).sort())))].sort()
  return JSON.stringify([concepts, exclude.map(p => p.toLowerCase()).sort()])
}

export function validatedExpansionPlan(baseline: string, expansions: KeywordQuery[]): MultiQueryPlan {
  const query = simpleWords(baseline, 12)
  if (!Array.isArray(expansions) || expansions.length < 1 || expansions.length > 2) throw new Error('原查询之外限 1–2 组扩展。')
  const seen = new Set([identity(query.split(' ').map(t => [t]), [])])
  const result = expansions.map(e => {
    if (!e || Object.keys(e).sort().join(',') !== 'exclude,groups,label' || typeof e.label !== 'string' || !e.label.trim() || e.label.length > 80 ||
      !Array.isArray(e.groups) || e.groups.length < 1 || e.groups.length > 3 || !Array.isArray(e.exclude) || e.exclude.length > 3) throw new Error('每组需要用途说明和 1–3 个概念组。')
    const groups = e.groups.map(g => {
      if (!Array.isArray(g) || g.length < 1 || g.length > 3) throw new Error('每个概念组限 1–3 个表达。')
      const phrases = g.map(p => {
        if (typeof p !== 'string' || p.length > 100) throw new Error('每个表达限 100 字符。')
        return simpleWords(p, 6)
      })
      if (new Set(phrases.map(p => p.toLowerCase())).size !== phrases.length) throw new Error('请删除重复的同义表达。')
      return phrases
    })
    const exclude = e.exclude.map(p => {
      if (typeof p !== 'string' || p.length > 100) throw new Error('每个排除表达限 100 字符。')
      return simpleWords(p, 6)
    })
    if (new Set(groups.map(g => JSON.stringify(g.map(p => p.toLowerCase()).sort()))).size !== groups.length ||
        new Set(exclude.map(p => p.toLowerCase())).size !== exclude.length) throw new Error('概念组或排除表达重复。')
    const key = identity(groups, exclude)
    if (seen.has(key)) throw new Error('扩展与原查询或另一组重复，请修改后采用。')
    seen.add(key)
    return { label: e.label.trim(), groups, exclude }
  })
  return { version: 1, baseline_query_text: query, expansions: result }
}
