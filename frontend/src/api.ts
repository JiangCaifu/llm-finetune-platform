/**
 * API 调用封装
 */

const API_BASE = import.meta.env.VITE_API_BASE || ''

export interface ChatResponse {
  answer: string
  model: string
  latency_ms: number
  tokens_generated: number
}

export async function sendChat(
  question: string,
  options?: {
    max_new_tokens?: number
    temperature?: number
    top_p?: number
  },
): Promise<ChatResponse> {
  const resp = await fetch(`${API_BASE}/api/v1/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      max_new_tokens: options?.max_new_tokens ?? 512,
      temperature: options?.temperature ?? 0.7,
      top_p: options?.top_p ?? 0.9,
    }),
  })

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }))
    throw new Error(err.detail || `请求失败 (${resp.status})`)
  }

  return resp.json()
}
