import { useEffect, useState } from 'react'
import { readApi, type RuntimeState } from './api'

export function useRuntime() {
  const [runtime, setRuntime] = useState<RuntimeState | null>(null)
  const [disconnected, setDisconnected] = useState(false)
  useEffect(() => {
    let mounted = true
    const check = async () => {
      try {
        const state = await readApi<RuntimeState>('/api/runtime')
        if (mounted) { setRuntime(state); setDisconnected(false) }
      } catch { if (mounted) { setRuntime(null); setDisconnected(true) } }
    }
    void check()
    const timer = window.setInterval(() => void check(), 5000)
    return () => { mounted = false; window.clearInterval(timer) }
  }, [])
  return { runtime, disconnected, ready: runtime?.worker_status === 'running' && !disconnected }
}

export function RuntimeNotice({ runtime, disconnected, compact = false }: { runtime: RuntimeState | null; disconnected: boolean; compact?: boolean }) {
  if (disconnected) return <p className="runtime-notice unavailable" role="status">无法连接后台。请确认本机服务仍在运行；已有记录和输入不会因此删除。</p>
  if (!runtime) return <p className="runtime-notice" role="status">正在检查后台状态…</p>
  if (runtime.worker_status === 'stopped') return <p className="runtime-notice unavailable" role="status">后台执行器未启动，排队任务尚未开始。请用完整启动入口运行服务后继续。</p>
  if (runtime.worker_status === 'unknown') return <p className="runtime-notice unavailable" role="status">暂时无法确认执行器状态，请检查服务终端后继续。</p>
  if (compact && !runtime.shared_workspace) return <p className="runtime-notice runtime-compact" role="status"><span aria-hidden="true" />{runtime.active_job ? '后台正在处理任务' : '后台就绪'}{runtime.queued_count > 0 ? ` · ${runtime.queued_count} 项排队` : ''}</p>
  return <div className="runtime-notice" role="status">{runtime.shared_workspace && <p className="shared-workspace">共享工作区 · 新记录供持有分享链接的人共同查看，与本机原历史分开保存。</p>}后台已启动{runtime.active_job ? ' · 正在处理任务' : ' · 可以开始研究'}{runtime.queued_count > 0 ? ` · ${runtime.queued_count} 项等待中` : ''}。任务依次执行，耗时取决于论文下载和模型响应。</div>
}
