/**
 * 主应用组件：聊天界面
 * - 顶部标题栏
 * - 中间消息列表（自动滚动到底部）
 * - 底部输入框
 */

import { useState, useRef, useEffect } from 'react'
import MessageBubble from './components/MessageBubble'
import InputBox from './components/InputBox'
import { sendChat } from './api'
import type { Message } from './types'

// 生成唯一 ID
const genId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

// 欢迎消息
const WELCOME: Message = {
  id: 'welcome',
  role: 'assistant',
  content:
    '你好！我是文旅智能问答助手，可以为你提供景点推荐、交通路线、美食建议、避坑指南等旅游信息。\n\n试试问我：\n- 北京有什么景点？\n- 带家人去九寨沟，关于门票优惠政策有什么建议？\n- 峨眉山的交通路线怎么走？',
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([WELCOME])
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // 自动滚动到底部
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }
  useEffect(scrollToBottom, [messages])

  // 发送消息
  const handleSend = async (text: string) => {
    // 1. 添加用户消息
    const userMsg: Message = { id: genId(), role: 'user', content: text }
    setMessages((prev) => [...prev, userMsg])
    setLoading(true)

    // 2. 添加占位的助手消息（显示加载中）
    const assistantId = genId()
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: 'assistant', content: '' },
    ])

    try {
      // 3. 调用 API
      const resp = await sendChat(text)

      // 4. 更新助手消息
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: resp.answer,
                model: resp.model,
                latency_ms: resp.latency_ms,
                tokens_generated: resp.tokens_generated,
              }
            : m,
        ),
      )
    } catch (err) {
      // 5. 错误处理
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: `请求失败：${err instanceof Error ? err.message : '未知错误'}`,
                error: true,
              }
            : m,
        ),
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-screen flex-col bg-gray-50">
      {/* 顶部标题栏 */}
      <header className="border-b border-gray-200 bg-white px-6 py-3 shadow-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary-600 text-lg text-white">
              🏛
            </div>
            <div>
              <h1 className="text-base font-semibold text-gray-800">文旅智能问答助手</h1>
              <p className="text-xs text-gray-400">Qwen2-7B-Instruct + LoRA 微调</p>
            </div>
          </div>
          <button
            onClick={() => setMessages([WELCOME])}
            className="rounded-lg px-3 py-1.5 text-xs text-gray-500 hover:bg-gray-100"
          >
            清空对话
          </button>
        </div>
      </header>

      {/* 消息列表 */}
      <div className="scrollbar-thin flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto max-w-3xl">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {/* 加载占位 */}
          {loading && (
            <div className="mb-4 flex justify-start">
              <div className="rounded-2xl border border-gray-200 bg-white px-4 py-3 shadow-sm">
                <div className="flex items-center gap-2 text-gray-400">
                  <span className="inline-block h-2 w-2 animate-bounce rounded-full bg-primary-400 [animation-delay:-0.3s]"></span>
                  <span className="inline-block h-2 w-2 animate-bounce rounded-full bg-primary-400 [animation-delay:-0.15s]"></span>
                  <span className="inline-block h-2 w-2 animate-bounce rounded-full bg-primary-400"></span>
                  <span className="text-sm">正在思考…</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* 底部输入 */}
      <InputBox onSend={handleSend} loading={loading} />
    </div>
  )
}
