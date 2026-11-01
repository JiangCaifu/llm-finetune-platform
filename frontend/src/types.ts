/**
 * 消息类型定义
 */

export type MessageRole = 'user' | 'assistant'

export interface Message {
  id: string
  role: MessageRole
  content: string
  // 助手消息的元信息
  latency_ms?: number
  tokens_generated?: number
  model?: string
  error?: boolean
}
