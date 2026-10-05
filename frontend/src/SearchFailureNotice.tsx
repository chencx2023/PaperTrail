import type { SearchState } from './api'

export function SearchFailureNotice({ search }: { search: SearchState | null }) {
  if (!search || (search.status !== 'failed' && search.status !== 'interrupted')) return null
  return <p className="section-note" role="status">
    {search.failure?.message || '本次搜索未完成，可以稍后显式重试。'}
    {search.failure && ` 本次已发出 ${search.failure.http_attempts} 次 arXiv 请求。`}
    {' 问题和失败记录已保存。'}
  </p>
}
