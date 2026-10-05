import { useEffect, useRef, useState } from 'react'
import { ApiFailure } from './api'

export type QuerySuggestion = {
  suggestion_id: string; question: string; query_text: string; model: string
  prompt_version: string; usage: { total_tokens?: number }; cost_usd: number | null
}

export function QueryAssistant({ question, disabled, onAdopt }: {
  question: string; disabled: boolean; onAdopt: (suggestion: QuerySuggestion) => void
}) {
  const [suggestion, setSuggestion] = useState<QuerySuggestion | null>(null)
  const [pending, setPending] = useState(false)
  const [message, setMessage] = useState('')
  const currentQuestion = useRef(question)
  currentQuestion.current = question
  const controller = useRef<AbortController | null>(null)
  useEffect(() => () => controller.current?.abort(), [])
  const generate = async () => {
    if (pending || !question.trim()) return
    setPending(true); setMessage(''); setSuggestion(null)
    const originalQuestion = question
    const request = new AbortController()
    controller.current = request
    const timer = window.setTimeout(() => request.abort(), 25000)
    try {
      const response = await fetch('/api/query-suggestions', { method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: originalQuestion }), signal: request.signal })
      const result = await response.json()
      if (!response.ok) throw new ApiFailure(result.code, result.message || '生成失败，请手动填写或稍后再试。', response.status)
      if (currentQuestion.current !== originalQuestion) {
        setMessage('研究问题已修改，请为新问题重新生成检索词。'); return
      }
      setSuggestion(result as QuerySuggestion)
    } catch (error) {
      setMessage(error instanceof ApiFailure ? error.message : '生成未完成，请稍后再试或手动填写。你的输入仍保留。')
    } finally { window.clearTimeout(timer); setPending(false); controller.current = null }
  }
  const usable = suggestion?.question === question ? suggestion : null
  return <div className="query-assistant">
    <div className="assistant-action"><button className="secondary" disabled={disabled || pending || !question.trim()} onClick={() => void generate()}>{pending ? '正在生成检索词…' : '生成检索词'}</button>
      <span>不知道怎么搜？让模型提炼英文关键词，再由你确认。</span></div>
    {pending && <p className="assistant-message" role="status">只生成建议，不会搜索或下载论文。通常在 20 秒内返回，超时会停止。</p>}
    {message && <p className="assistant-message" role="status">{message}</p>}
    {usable && <div className="query-suggestion" role="status"><strong>{usable.query_text}</strong><button className="secondary" disabled={disabled} onClick={() => onAdopt(usable)}>采用建议</button>
      <p>采用后可编辑上方检索词，再点击“开始搜索”。关键词会同时匹配标题或摘要，词太多可能缩小结果范围。</p>
    </div>}
  </div>
}
