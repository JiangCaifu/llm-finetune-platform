/**
 * 消息气泡组件：用户和助手消息不同样式
 */

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message } from '../types'

export default function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? 'bg-primary-600 text-white'
            : message.error
              ? 'bg-red-50 text-red-700 border border-red-200'
              : 'bg-white text-gray-800 border border-gray-200 shadow-sm'
        }`}
      >
        {/* 头像标识 */}
        <div className={`mb-1 text-xs ${isUser ? 'text-primary-100' : 'text-gray-400'}`}>
          {isUser ? '我' : '文旅助手'}
        </div>

        {/* 消息内容 */}
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {/* 助手消息的元信息 */}
        {!isUser && !message.error && message.latency_ms && (
          <div className="mt-2 border-t border-gray-100 pt-1 text-xs text-gray-400">
            {message.model} · {message.tokens_generated} tokens · {(message.latency_ms / 1000).toFixed(1)}s
          </div>
        )}
      </div>
    </div>
  )
}
