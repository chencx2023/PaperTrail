export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'interrupted'

export type Candidate = {
  arxiv_id?: string
  display_rank: number
  title: string
  abstract: string
  authors: string[]
  versioned_id: string
  versions: { versioned_id: string; version: number | null }[]
}

export type SearchState = {
  search_id: string
  job_id: string
  status: JobStatus
  question: string
  query_text: string
  snapshot_sha256: string | null
  candidates: Candidate[] | null
  query_plan?: import('./query-plan').MultiQueryPlan | null
  fusion?: SearchFusion | null
  failure?: { kind: 'rate_limited' | 'cooldown' | 'network' | 'upstream'
    message: string; http_attempts: number } | null
}

export type SearchFusion = {
  method: 'rrf'; k: number; successful_queries: number; partial: boolean
  candidate_count: number; http_attempts: number; cache_hits: number; elapsed_ms: number
  queries: { index: number; label: string; search_query: string; status: 'success' | 'failed'
    candidate_count: number | null; cache_hit: boolean; fetched_at: string | null; error_kind: string | null
    original_order: { arxiv_id: string; versioned_id: string; rank: number }[] }[]
  ranking: { arxiv_id: string; rank: number; score: number; matches: { query_index: number; rank: number }[] }[]
  merge_order: string[]; relaxation_requests: number
  generation?: { model: string; usage: { total_tokens?: number }; elapsed_ms: number; cost_usd: number | null } | null
}

export type ImportItem = {
  position: number
  versioned_id: string
  title?: string | null
  status: string
  document_id: string | null
  reused_pdf: boolean
  failure_reason: string | null
}

export type CollectionState = {
  collection_id: string
  selection_revision: number
  selections: { search_id: string | null; versioned_id: string; snapshot_sha256: string }[]
  import_job_id: string | null
  import_job_status: JobStatus | null
  import_report: {
    import_run_id: string
    selection_revision: number
    status: string
    items: ImportItem[]
  } | null
}

export type RevisionState = {
  id: string
  collection_id: string
  selection_revision: number
  revision_no: number
  members: { document_id: string; versioned_id: string; sha256: string }[]
  index_job_id: string | null
  index_status: JobStatus | null
  index_artifact_id: string | null
}

export type Evidence = {
  evidence_id: string
  upstream_context_id: string
  document_id: string | null
  upstream_dockey: string
  summary: string
  original_text: string | null
  upstream_locator: string | null
  locator_status: string
  pdf_page: number | null
}

export type ResearchState = {
  research_id: string; parent_id: string | null; title: string; title_is_custom: boolean
  search_id: string | null; collection_id: string | null; pool_revision_id: string | null
  scope_revision_id: string | null; answer_run_id: string | null; question: string | null
  papers: { document_id: string; versioned_id: string; title: string; authors: string[] }[]
  selected_versions: string[]; answer_document_ids: string[]
  preparation: RevisionState | null
  operation: { operation_id: string; kind: 'prepare' | 'answer'; phase: string
    status: 'pending' | 'succeeded' | 'failed'; job_id: string | null; message: string | null; document_ids: string[] } | null
  import_report: CollectionState['import_report']
  created_at: string; updated_at: string
}

export type PdfPageLocation = {
  evidence_id: string
  pdf_sha256: string
  status: 'unique' | 'multiple' | 'unknown'
  matches: { start_page: number; end_page: number }[]
}

export type PdfPageLookup =
  | { status: 'loading' | 'error' }
  | { status: 'ready'; location: PdfPageLocation }

export type SavedContextCoverage = {
  status: 'known' | 'unknown'
  basis: 'saved_paperqa_session_context'
  total_documents: number | null
  included_documents: { document_id: string; versioned_id: string }[] | null
  missing_documents: { document_id: string; versioned_id: string }[] | null
}

export function describeSavedContextCoverage(coverage?: SavedContextCoverage | null) {
  const unknown = { summary: '已保存的生成上下文覆盖未知', included: [], missing: [] } as {
    summary: string; included: string[]; missing: string[]
  }
  if (coverage?.status !== 'known' || !Array.isArray(coverage.included_documents) ||
      !Array.isArray(coverage.missing_documents) || coverage.total_documents == null ||
      coverage.included_documents.length + coverage.missing_documents.length !== coverage.total_documents) {
    return unknown
  }
  return {
    summary: `已保存的生成上下文涉及 ${coverage.included_documents.length}/${coverage.total_documents} 篇固定论文`,
    included: coverage.included_documents.map((document) => document.versioned_id),
    missing: coverage.missing_documents.map((document) => document.versioned_id),
  }
}

export type AnswerBundle = {
  research_context?: { collection_id: string; revision_id: string } | null
  document_titles?: Record<string, string>
  document_sources?: Record<string, DocumentSource>
  question: string
  raw_answer: string
  display_answer: string
  job_status: string
  answer_outcome: string
  upstream_status: string
  termination_reason: string
  allowed_documents: { document_id: string; versioned_id: string }[]
  evidence: Evidence[]
  citations: { upstream_context_id: string; start: number; end: number; evidence_id: string | null; mapping_status: string }[]
  usage: { embedding_tokens: number | null; cost_usd: number | null }
  saved_context_coverage?: SavedContextCoverage
  document_usage?: AnswerDocumentUsage
}

export type DocumentSource = {
  document_id: string; versioned_id: string; title: string; authors: string[]; year: number | null
  published_at?: string; abstract: string | null; abstract_status: 'saved' | 'unavailable' | 'unverified'
}

export type AnswerDocumentUsage = {
  basis: 'saved_answer_records'
  total_documents: number | null
  evidence_documents: { document_id: string; versioned_id: string }[] | null
  evidence_fragments: number | null
  context_documents: { document_id: string; versioned_id: string }[] | null
  context_fragments: number | null
  cited_documents: { document_id: string; versioned_id: string }[] | null
  evidence_fragment_limit: number | null
  context_selection?: { status: 'known' | 'unknown'; budget_tokens?: number; estimated_tokens?: number
    token_count_basis?: string; selected_fragments?: number; selected_documents?: number }
  documents: {
    document_id: string; versioned_id: string
    has_evidence: boolean | null; in_context: boolean | null; is_cited: boolean | null
  }[]
}

export type AnswerState = {
  answer_run_id: string
  job_id: string | null
  attempt_id: string | null
  previous_answer_run_id: string | null
  status: JobStatus
  answer: AnswerBundle | null
}

export type JobState = {
  job_id: string
  kind: string
  resource_id: string
  status: JobStatus
  active_attempt_id: string | null
  created_at: string
  updated_at: string
  attempt: {
    started_at: string
    heartbeat_at: string | null
    lease_until: string | null
    finished_at: string | null
    error_kind: string | null
    error_message: string | null
  } | null
}

export type RuntimeState = {
  shared_workspace?: boolean
  worker_status: 'running' | 'stopped' | 'unknown'
  queued_count: number
  active_job: { job_id: string; kind: string; started_at: string; heartbeat_at: string; lease_until: string } | null
  checked_at: string
}

export class ApiFailure extends Error {
  readonly code: string
  readonly status: number

  constructor(code: string, message: string, status: number) {
    super(message)
    this.code = code
    this.status = status
  }
}

export async function readApi<T>(path: string): Promise<T> {
  const response = await fetch(path, { headers: { Accept: 'application/json' } })
  const payload: unknown = await response.json()
  if (!response.ok) {
    const error = payload as { code?: string; message?: string }
    throw new ApiFailure(error.code || 'REQUEST_FAILED', error.message || '请求失败', response.status)
  }
  return payload as T
}

type Pending = { operation: string; path: string; body: string; key: string }
const pendingPrefix = 'papertrail.pending.'

export function pendingSubmissions(): Pending[] {
  return ['search', 'collection', 'import', 'revision', 'answer', 'prepare', 'research-answer', 'research-retry'].flatMap((operation) => {
    const raw = sessionStorage.getItem(pendingPrefix + operation)
    if (!raw) return []
    try {
      const item = JSON.parse(raw) as Pending
      return item.operation === operation && typeof item.path === 'string' &&
        typeof item.body === 'string' && typeof item.key === 'string' ? [item] : []
    } catch {
      return []
    }
  })
}

export async function replayPending<T>(pending: Pending): Promise<T> {
  return sendPending<T>(pending)
}

export async function submitApi<T>(operation: Pending['operation'], path: string, body: object): Promise<T> {
  const serialized = JSON.stringify(body)
  const storageKey = pendingPrefix + operation
  const original = pendingSubmissions().find((item) => item.operation === operation)
  const pending = original?.path === path && original.body === serialized
    ? original : { operation, path, body: serialized, key: crypto.randomUUID() }
  sessionStorage.setItem(storageKey, JSON.stringify(pending))
  return sendPending<T>(pending)
}

async function sendPending<T>(pending: Pending): Promise<T> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), 15000)
  try {
    const response = await fetch(pending.path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Idempotency-Key': pending.key },
      body: pending.body,
      signal: controller.signal,
    })
    const payload: unknown = await response.json()
    if (!response.ok) {
      const error = payload as { code?: string; message?: string }
      if (response.status >= 400 && response.status < 500) {
        sessionStorage.removeItem(pendingPrefix + pending.operation)
      }
      throw new ApiFailure(error.code || 'REQUEST_FAILED', error.message || '请求失败', response.status)
    }
    sessionStorage.removeItem(pendingPrefix + pending.operation)
    return payload as T
  } finally {
    window.clearTimeout(timeout)
  }
}

export async function mutateApi<T>(path: string, body: object, method: 'PATCH' | 'POST'): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const payload: unknown = await response.json()
  if (!response.ok) {
    const error = payload as { code?: string; message?: string }
    throw new ApiFailure(error.code || 'REQUEST_FAILED', error.message || '请求失败', response.status)
  }
  return payload as T
}
