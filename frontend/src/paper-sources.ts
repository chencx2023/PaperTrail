import type { DocumentSource, Evidence } from './api'

export function paperKey(item: Evidence) {
  return item.document_id ? `document:${item.document_id}:${item.upstream_dockey}` : `evidence:${item.evidence_id}`
}

export function savedPaperSource(item: Evidence, sources: Record<string, DocumentSource>) {
  const source = sources[item.document_id || '']
  return source?.document_id === item.document_id && source.versioned_id === item.upstream_dockey ? source : undefined
}

// arXiv stores full author names. Shorten conventional names for display only;
// the complete saved list is retained in the source table.
function shortAuthor(name: string) {
  if (name.includes(',')) return name.split(',')[0].trim()
  const tokens = name.trim().split(/\s+/)
  if (tokens.length === 1) return name.trim()
  const particles = new Set(['van', 'von', 'de', 'del', 'der', 'den', 'da', 'di', 'dos', 'du'])
  let start = tokens.length - 1
  while (start > 0 && particles.has(tokens[start - 1].toLowerCase())) start--
  return tokens.slice(start).join(' ')
}

export function paperCitationLabel(source: DocumentSource | undefined, fallback: number, sources: Record<string, DocumentSource> = {}) {
  if (!source?.authors?.length || !source.year) return `[${fallback}]`
  const author = shortAuthor(source.authors[0]) + (source.authors.length > 1 ? ' et al.' : '')
  const collisions = Object.values(sources).filter(s => s.authors?.length && s.year === source.year &&
    shortAuthor(s.authors[0]) + (s.authors.length > 1 ? ' et al.' : '') === author)
    .sort((a, b) => a.versioned_id.localeCompare(b.versioned_id))
  const suffix = collisions.length > 1 ? String.fromCharCode(97 + collisions.findIndex(s => s.document_id === source.document_id)) : ''
  return `${author}, ${source.year}${suffix}`
}

export function paperGroups(items: Evidence[]) {
  const groups = new Map<string, Evidence[]>()
  for (const item of items) groups.set(paperKey(item), [...(groups.get(paperKey(item)) || []), item])
  return [...groups].map(([key, fragments]) => ({ key, fragments }))
}
