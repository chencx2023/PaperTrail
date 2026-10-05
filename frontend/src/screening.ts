import type { Candidate, SearchState } from './api'

export type RelevanceLabel = 'high' | 'partial' | 'low' | 'insufficient'
export type ScreeningView = 'all' | 'selected'
export type PaperAssessment = { paper_id: string; label: RelevanceLabel; reason: string; evidence_quote: string
  evidence_span?: { id: string; start: number; end: number } | null }
export type AbstractScreeningResult = {
  status?: 'succeeded' | 'partial'
  unverified_papers?: { paper_id: string; kind: 'unverified'; message: string }[]
  screening_id: string; search_id: string; snapshot_sha256: string; question: string
  candidate_count: number; interpretation: string; criteria: string[]; papers: PaperAssessment[]
  model: string; prompt_version: string; created_at: string; elapsed_ms: number
  validation_version?: string
  usage: { total_tokens?: number }; cost_usd: number | null; cache_hit: boolean
  screening_scope?: { method: 'rrf_top' | 'all_candidates'; limit: number; source_candidate_count: number; paper_ids: string[] }
}

export type ScreeningPlan = {
  total_count: number; verified_count: number; remaining_count: number
  batch_count: number; wait_timeout_ms: number
}

export function screeningPlanMatches(total: number, value: ScreeningPlan): boolean {
  if (!value || value.total_count !== total || !Number.isInteger(total) || total < 1 || total > 60 ||
      ![value.verified_count, value.remaining_count, value.batch_count, value.wait_timeout_ms].every(Number.isSafeInteger) ||
      value.verified_count < 0 || value.remaining_count < 0 || value.verified_count + value.remaining_count !== total ||
      value.batch_count < Math.ceil(value.remaining_count / 10) || value.batch_count > value.remaining_count ||
      value.wait_timeout_ms !== value.batch_count * 60_000 + 15_000) return false
  return true
}

export const relevanceLabels: Record<RelevanceLabel, string> = {
  high: '优先阅读', partial: '可能相关', low: '关联较弱', insufficient: '摘要信息不足',
}
const priority: Record<RelevanceLabel, number> = { high: 0, partial: 1, low: 2, insufficient: 3 }

export function screeningMatches(search: SearchState, result: AbstractScreeningResult): boolean {
  if (!result || result.search_id !== search.search_id || result.question !== search.question ||
      result.snapshot_sha256 !== search.snapshot_sha256 || !Array.isArray(result.papers) ||
      !Array.isArray(result.criteria) || !result.criteria.every(c => typeof c === 'string') ||
      typeof result.interpretation !== 'string' || !Array.isArray(search.candidates)) return false
  let expected = search.candidates
  if (result.screening_scope) {
    const scope = result.screening_scope
    expected = [...search.candidates].sort((a, b) => a.display_rank - b.display_rank)
    if (scope.method === 'rrf_top') {
      if (search.fusion?.method !== 'rrf' || scope.limit !== 30) return false
      expected = expected.slice(0, 30)
    } else if (scope.method !== 'all_candidates' || scope.limit !== 60 ||
      result.validation_version !== 'abstract-relevance-fullpool-validation-v4') return false
    if (
      scope.source_candidate_count !== search.candidates.length || !Array.isArray(scope.paper_ids) ||
      scope.paper_ids.length !== expected.length || scope.paper_ids.some((id, i) => id !== expected[i].versioned_id)) return false
  } else if (search.fusion?.method === 'rrf') return false
  if (result.candidate_count !== expected.length) return false
  const ids = new Set(expected.map(p => p.versioned_id))
  const seen = new Set<string>()
  for (const paper of result.papers) {
    if (!paper || !ids.has(paper.paper_id) || seen.has(paper.paper_id) || !(paper.label in priority) ||
        typeof paper.reason !== 'string' || typeof paper.evidence_quote !== 'string') return false
    seen.add(paper.paper_id)
  }
  if (result.status === 'partial') {
    if (!Array.isArray(result.unverified_papers) || !result.unverified_papers.length) return false
    for (const paper of result.unverified_papers) {
      if (!paper || !ids.has(paper.paper_id) || seen.has(paper.paper_id) || paper.kind !== 'unverified' ||
          typeof paper.message !== 'string') return false
      seen.add(paper.paper_id)
    }
  } else if (result.unverified_papers?.length) return false
  return seen.size === ids.size
}

export function screenedCandidates(candidates: Candidate[], result: AbstractScreeningResult | null,
  view: ScreeningView, selectedVersions: string[]): Candidate[] {
  const assessments = new Map(result?.papers.map(p => [p.paper_id, priority[p.label]]) || [])
  const original = [...candidates].sort((a, b) => (assessments.get(a.versioned_id) ?? 4) -
    (assessments.get(b.versioned_id) ?? 4) || a.display_rank - b.display_rank)
  const selected = new Set(selectedVersions)
  if (view === 'selected') return original.filter(p => p.versions.some(v => selected.has(v.versioned_id)))
  return original
}

export function candidatePage(candidates: Candidate[], requestedPage: number) {
  const pageSize = 10
  const pages = Math.max(1, Math.ceil(candidates.length / pageSize))
  const page = Math.min(pages, Math.max(1, Math.floor(requestedPage) || 1))
  const offset = (page - 1) * pageSize
  return { page, pages, start: candidates.length ? offset + 1 : 0,
    end: Math.min(offset + pageSize, candidates.length), items: candidates.slice(offset, offset + pageSize) }
}

export function changeVersionSelection(selected: string[], versionId: string, checked: boolean): string[] {
  if (!checked) return selected.filter(id => id !== versionId)
  if (selected.includes(versionId) || selected.length >= 10) return selected
  return [...selected, versionId]
}
