export function historyHref(href: string, open: boolean, filter?: { kind: string; cursor: string }): string {
  const url = new URL(href, 'http://papertrail.local')
  if (open) {
    url.searchParams.set('history', '1')
    if (filter) {
      url.searchParams.set('kind', filter.kind)
      if (filter.cursor) url.searchParams.set('cursor', filter.cursor)
      else url.searchParams.delete('cursor')
    }
  } else {
    for (const key of ['history', 'kind', 'cursor']) url.searchParams.delete(key)
  }
  return `${url.pathname}${url.search}${url.hash}`
}
