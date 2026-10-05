import type { SearchState } from './api'

export function SearchFusion({ search }: { search: SearchState }) {
  const fusion = search.fusion
  if (!fusion) return null
  const count = fusion.candidate_count
  return <section className="search-fusion" aria-label="多查询融合结果">
    <h3>多查询融合 · {count} 篇候选</h3>
    <p>{fusion.successful_queries}/{fusion.queries.length} 组查询完成，按 RRF 合并。同一工作合并展示，固定版本保留；按摘要筛选后优先展示阅读建议。</p>
    {fusion.partial && <p className="fusion-warning" role="status">部分查询未完成，当前结果只来自成功查询。可重新发起搜索，旧记录保留。</p>}
    <details className="screening-record"><summary>查看查询来源与原始排名</summary>
      <p className="section-note">本次搜索 {(fusion.elapsed_ms / 1000).toFixed(1)} 秒 · {fusion.http_attempts} 次元数据请求（含重试）· {fusion.cache_hits} 组命中缓存</p>
      {fusion.generation && <p className="section-note">生成扩展：{fusion.generation.model} · {typeof fusion.generation.usage?.total_tokens === 'number' ? `${fusion.generation.usage.total_tokens} token` : '用量未知'} · {fusion.generation.cost_usd == null ? '费用未知' : `$${fusion.generation.cost_usd}`}</p>}
      <p className="section-note">RRF 汇总每组的名次，重复出现或排名靠前的论文更靠前。它不判断摘要是否回答你的问题；相关性由后续摘要筛选和你来核对。</p>
      <div className="fusion-query-list">{fusion.queries.map(query => <details key={query.index} className="fusion-query"><summary>查询 {query.index} · {query.label}<span>{query.status === 'success' ? `${query.candidate_count} 篇${query.cache_hit ? ' · 缓存' : ''}` : '未完成'}</span></summary>
        {query.index === 1 ? <p>{search.query_text}</p> : search.query_plan?.expansions[query.index - 2] && <p>{search.query_plan.expansions[query.index - 2].groups.map(g => g.join(' / ')).join(' + ')}</p>}
        {query.fetched_at && <p className="section-note">原始检索时间：{new Date(query.fetched_at).toLocaleString()}</p>}
        {query.original_order.length > 0 && <ol>{query.original_order.map(paper => <li key={paper.arxiv_id}>{search.candidates?.find(c => c.arxiv_id === paper.arxiv_id)?.title || paper.versioned_id}<small>{paper.versioned_id}</small></li>)}</ol>}
      </details>)}</div>
    </details>
  </section>
}
