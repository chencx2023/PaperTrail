export type SavedCitation = {
  upstream_context_id: string
  start: number
  end: number
  evidence_id: string | null
  mapping_status: string
}

export type CitationPart =
  | { kind: 'text'; text: string }
  | { kind: 'link'; text: string; evidenceId: string; number: number }
  | { kind: 'unavailable'; text: string; status: string; number: number }

export function splitCitationText(
  rawAnswer: string,
  citations: SavedCitation[],
  availableEvidenceIds: ReadonlySet<string>,
): CitationPart[] {
  // Python saved offsets count Unicode code points; JS string slices count UTF-16 units.
  const characters = Array.from(rawAnswer)
  const parts: CitationPart[] = []
  const numbers = new Map<string, number>()
  let cursor = 0
  for (const citation of [...citations].sort((a, b) => a.start - b.start)) {
    const { start, end } = citation
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < cursor ||
        end <= start || end > characters.length ||
        characters.slice(start, end).join('') !== citation.upstream_context_id) continue
    if (start > cursor) parts.push({ kind: 'text', text: characters.slice(cursor, start).join('') })
    const key = citation.evidence_id ? `evidence:${citation.evidence_id}` : `context:${citation.upstream_context_id}`
    if (!numbers.has(key)) numbers.set(key, numbers.size + 1)
    const number = numbers.get(key)!
    if (citation.mapping_status === 'mapped' && citation.evidence_id &&
        availableEvidenceIds.has(citation.evidence_id)) {
      parts.push({ kind: 'link', text: citation.upstream_context_id, evidenceId: citation.evidence_id, number })
    } else {
      parts.push({ kind: 'unavailable', text: citation.upstream_context_id,
        status: citation.mapping_status === 'mapped' ? 'evidence_unavailable' : citation.mapping_status, number })
    }
    cursor = end
  }
  if (cursor < characters.length) parts.push({ kind: 'text', text: characters.slice(cursor).join('') })
  return parts
}

export function evidenceHref(pathname: string, search: string, evidenceId: string, citationId?: string): string {
  const params = new URLSearchParams(search)
  params.set('evidence', evidenceId)
  if (citationId) params.set('citation', citationId)
  else params.delete('citation')
  return `${pathname}?${params.toString()}#evidence`
}

export function citationReturnId(parts: CitationPart[], evidenceId: string, hash: string, preferred = '', renderedIds?: ReadonlySet<string>): string {
  const occurrences = parts.flatMap((part, index) =>
    part.kind === 'link' && part.evidenceId === evidenceId &&
      (!renderedIds || renderedIds.has(`citation-${index}`)) ? [`citation-${index}`] : [])
  const fragment = hash.slice(1)
  return occurrences.includes(fragment) ? fragment : occurrences.includes(preferred) ? preferred : occurrences[0] || ''
}
