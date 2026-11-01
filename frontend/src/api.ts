/**
 * API 调用封装
 *
 * 两种模式：
 * - 真实 API：VITE_MOCK=false（默认），请求后端 /api/v1/chat
 * - Mock 模式：VITE_MOCK=true，返回假数据，不需要后端
 */

const API_BASE = import.meta.env.VITE_API_BASE || ''
const USE_MOCK = import.meta.env.VITE_MOCK === 'true'

export interface ChatResponse {
  answer: string
  model: string
  latency_ms: number
  tokens_generated: number
}

// ===== Mock 数据（模拟文旅助手的回答）=====
const MOCK_ANSWERS: Record<string, string> = {
  北京: `北京作为千年古都与现代大都市的融合体，拥有丰富多元的旅游景观：

- **历史文化类**：故宫（世界五大宫之首）、天安门广场、颐和园、八达岭长城、周口店北京人遗址；
- **皇家建筑群**：北海公园、景山公园（万春亭可俯瞰中轴线全貌）、恭王府；
- **现代风貌**：798艺术区、南锣鼓巷、北京坊。

**推荐路线**：第一天故宫+天安门，第二天长城，第三天颐和园+圆明园。`,

  九寨沟: `九寨沟旅游建议：

**门票政策**：
- 旺季（4-11月）：门票169元 + 观光车90元
- 淡季（11-3月）：门票80元 + 观光车80元
- 老人/学生/儿童有优惠

**避坑要点**：
1. 提前在网上预约，现场不售票
2. 海拔2000+，注意高反，带厚外套
3. 景区内只有诺日朗餐厅，价格偏高，建议自备干粮
4. 观光车随上随下，合理规划路线`,

  美食: `各地特色美食推荐：

- **北京**：烤鸭（全聚德/大董）、炸酱面、豆汁焦圈
- **西安**：肉夹馍、凉皮、羊肉泡馍、biangbiang面
- **成都**：火锅、串串香、担担面、龙抄手
- **桂林**：桂林米粉、啤酒鱼、田螺酿
- **三亚**：海鲜大餐、椰子鸡、清补凉`,
}

const DEFAULT_MOCK = `这是一个 Mock 模式的模拟回答（不需要后端服务）。

你问的是关于文旅的问题，在真实环境中，微调后的 Qwen2-7B 模型会给出详细的旅游建议。

**Mock 模式用途**：
- 本地开发调试前端 UI
- 面试演示时 GPU 不可用的备选方案

试试问我"北京有什么景点"或"九寨沟门票政策"。`

function mockSendChat(question: string): Promise<ChatResponse> {
  // 模拟网络延迟（1-2秒）
  const delay = 1000 + Math.random() * 1000

  // 关键词匹配 Mock 回答
  let answer = DEFAULT_MOCK
  for (const [keyword, ans] of Object.entries(MOCK_ANSWERS)) {
    if (question.includes(keyword)) {
      answer = ans
      break
    }
  }

  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        answer,
        model: 'qwen2-7b-lora (mock)',
        latency_ms: delay,
        tokens_generated: Math.floor(answer.length / 2),
      })
    }, delay)
  })
}

// ===== 对外接口 =====
export async function sendChat(
  question: string,
  options?: {
    max_new_tokens?: number
    temperature?: number
    top_p?: number
  },
): Promise<ChatResponse> {
  // Mock 模式
  if (USE_MOCK) {
    return mockSendChat(question)
  }

  // 真实 API
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
