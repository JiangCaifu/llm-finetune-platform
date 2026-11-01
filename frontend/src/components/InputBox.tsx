/**
 * 输入框组件：文本输入 + 发送按钮 + 加载状态
 */

import { useState, type KeyboardEvent } from 'react'

interface Props {
  onSend: (text: string) => void
  loading: boolean
}

export default function InputBox({ onSend, loading }: Props) {
  const [text, setText] = useState('')

  const handleSend = () => {
    const trimmed = text.trim()
    if (!trimmed || loading) return
    onSend(trimmed)
    setText('')
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter 发送，Shift+Enter 换行
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="border-t border-gray-200 bg-white p-4">
      <div className="flex items-end gap-3">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入你的文旅问题…（Enter 发送，Shift+Enter 换行）"
          rows={1}
          className="flex-1 resize-none rounded-xl border border-gray-300 px-4 py-2.5 text-sm
                     focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500
                     disabled:bg-gray-50"
          disabled={loading}
          style={{ minHeight: '42px', maxHeight: '120px' }}
        />
        <button
          onClick={handleSend}
          disabled={loading || !text.trim()}
          className="rounded-xl bg-primary-600 px-5 py-2.5 text-sm font-medium text-white
                     hover:bg-primary-700 disabled:cursor-not-allowed disabled:bg-gray-300
                     transition-colors whitespace-nowrap"
        >
          {loading ? '推理中…' : '发送'}
        </button>
      </div>
    </div>
  )
}
