type Paper = { document_id: string }

export function initialAnswerSelection(papers: Paper[], saved: string[] | null): string[] {
  return papers.map(paper => paper.document_id).filter(id => saved === null || saved.includes(id))
}

export function changeAnswerSelection(papers: Paper[], selected: string[], id: string, checked: boolean): string[] {
  const next = new Set(selected)
  if (checked) next.add(id)
  else next.delete(id)
  return papers.map(paper => paper.document_id).filter(id => next.has(id))
}
