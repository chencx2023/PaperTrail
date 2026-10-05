import type { CitationPart } from './citations'

type Reference = Exclude<CitationPart, { kind: 'text' }> & { partIndex: number }
export type AnswerDisplayPart = { kind: 'text'; text: string } | { kind: 'references'; items: Reference[] }
export type FormattedAnswerPart = AnswerDisplayPart | { kind: 'strong'; parts: AnswerDisplayPart[] }

// Only presentation changes here. Saved offsets and citation identities stay in splitCitationText.
export function answerDisplayParts(source: (CitationPart & { partIndex?: number })[]): AnswerDisplayPart[] {
  const parts = source.map(part => ({ ...part }))
  const shown: AnswerDisplayPart[] = []
  for (let index = 0; index < parts.length;) {
    const part = parts[index]
    if (part.kind === 'text') {
      shown.push({ kind: 'text', text: part.text })
      index++
      continue
    }
    const items: Reference[] = [{ ...part, partIndex: part.partIndex ?? index }]
    let next = index + 1
    while (next + 1 < parts.length) {
      const separator = parts[next], reference = parts[next + 1]
      if (separator.kind !== 'text' || !/^[ \t]*[,，;；][ \t]*$/.test(separator.text) || reference.kind === 'text') break
      items.push({ ...reference, partIndex: reference.partIndex ?? next + 1 })
      next += 2
    }
    const before = shown[shown.length - 1], after = parts[next]
    const opening = before?.kind === 'text' ? before.text.match(/[ \t]*([（(])[ \t]*$/) : null
    const closing = after?.kind === 'text' ? after.text.match(/^[ \t]*([）)])/): null
    if (before?.kind === 'text' && after?.kind === 'text' && opening && closing &&
        (opening[1] === '(' ? closing[1] === ')' : closing[1] === '）')) {
      before.text = before.text.slice(0, opening.index)
      after.text = after.text.slice(closing[0].length)
      const punctuation = after.text.match(/^[ \t]*([。.!?！？]+)/)
      if (punctuation && before.text.trim()) {
        before.text += punctuation[1]
        after.text = after.text.slice(punctuation[0].length)
      }
    }
    shown.push({ kind: 'references', items })
    index = next
  }
  return shown
}

// A small, text-only bold formatter; React escapes all source text, including HTML.
export function formatAnswerDisplay(parts: AnswerDisplayPart[]): FormattedAnswerPart[] {
  type Token = AnswerDisplayPart | { kind: 'marker' }
  const tokens: Token[] = parts.flatMap<Token>(part => part.kind === 'references' ? [part] :
    part.text.split(/(\\\*\*|\*\*)/g).filter(Boolean).map(text =>
      text === '**' ? { kind: 'marker' as const } : { kind: 'text' as const, text }))
  const result: FormattedAnswerPart[] = []
  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index]
    if (token.kind !== 'marker') { result.push(token); continue }
    const contents: AnswerDisplayPart[] = []
    let next = index + 1
    for (; next < tokens.length; next++) {
      const inner = tokens[next]
      if (inner.kind === 'marker' || inner.kind === 'text' && /\r?\n[ \t]*\r?\n/.test(inner.text)) break
      contents.push(inner)
    }
    if (tokens[next]?.kind === 'marker' && contents.some(p => p.kind === 'references' || p.text.trim())) {
      result.push({ kind: 'strong', parts: contents })
      index = next
    } else result.push({ kind: 'text', text: '**' })
  }
  return result
}
