import { pendingSubmissions, replayPending } from './api.ts'

export async function recoverPending(
  onRecovered: (operation: string, result: Record<string, string>) => void,
  refreshCollection: () => void,
  onError: (cause: unknown) => void,
): Promise<void> {
  for (const item of pendingSubmissions()) {
    try {
      const result = await replayPending<Record<string, string>>(item)
      onRecovered(item.operation, result)
      // Import has no new URL resource ID, so a replay must explicitly re-read the collection.
      if (item.operation === 'import') refreshCollection()
    } catch (cause) {
      onError(cause)
    }
  }
}
