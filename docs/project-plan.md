# 大模型微调与服务平台 - 项目计划

## 一、项目定位

### 项目名称
文旅垂直领域大模型微调与服务平台

### 一句话定位
针对通用大模型在垂直领域准确率低的问题，使用QLoRA对Qwen-7B进行微调，封装为推理服务，并搭建Web管理平台。

### 项目背景

通用大模型在文旅领域存在回答泛化、事实不准确、格式不统一的问题。通过垂直领域微调，可以让模型在特定业务场景下输出更结构化、更聚焦的答案；相比调用闭源大模型 API，成本更低，且数据不出本地。

---

## 二、核心知识点

### 技术栈全景

```
微调项目知识栈
    │
    ├── 1. 模型基础
    │   ├── Transformer架构（注意力机制/位置编码/FFN）
    │   ├── Tokenizer原理（BPE/SentencePiece）
    │   └── 预训练 vs 微调 vs 对齐的区别
    │
    ├── 2. 参数高效微调（PEFT）
    │   ├── LoRA原理（低秩分解为什么有效）
    │   ├── QLoRA（4bit量化+LoRA）
    │   ├── 超参数：rank / alpha / target_modules
    │   └── 为什么不全量微调（显存/灾难性遗忘）
    │
    ├── 3. 数据工程
    │   ├── 指令数据集构建（Alpaca格式）
    │   ├── 数据清洗 / 去重 / 质量过滤
    │   ├── 数据增强（LLM生成 + 人工校验）
    │   └── 训练集 / 验证集 / 测试集划分
    │
    ├── 4. 训练优化
    │   ├── 混合精度训练（fp16 / bf16）
    │   ├── 梯度累积（小显存跑大batch）
    │   ├── Paged AdamW优化器（8bit省显存）
    │   └── 学习率调度（warmup / cosine）
    │
    ├── 5. 量化技术
    │   ├── 4bit量化（NF4 / FP4）
    │   ├── AWQ vs GPTQ vs NF4区别
    │   └── 量化对精度的影响
    │
    ├── 6. 评测对比
    │   ├── 微调前后准确率对比
    │   ├── 指标：Exact Match / F1 / BLEU
    │   └── 过拟合检测（训练loss vs 验证loss）
    │
    ├── 7. 推理部署
    │   ├── LoRA权重加载与合并
    │   ├── vLLM推理加速概念
    │   └── FastAPI服务封装
    │
    └── 8. 对齐技术（加分项）
        ├── SFT（监督微调）
        ├── DPO（直接偏好优化）概念
        └── RLHF vs DPO区别
```

### 核心技术点

| 知识点 | 说明 |
|--------|------|
| LoRA 原理（低秩分解） | 用低秩矩阵近似权重更新，大幅减少可训练参数 |
| QLoRA 量化 | 4bit 量化基座 + LoRA，进一步降低显存占用 |
| rank / alpha 选择 | rank 控制容量，alpha 控制缩放比例 |
| 数据质量 | 清洗、去重、长度过滤、验证集监控 |
| 过拟合处理 | 验证集早停、降低 epoch、加 dropout |
| Transformer 架构 | 注意力机制、位置编码、FFN |
| DPO / RLHF | 偏好对齐方法，与 SFT 的区别 |

---

## 三、项目架构

```
大模型微调与服务平台
    │
    ├── 数据层：文档解析 + 指令数据集构建 + ETL流水线
    ├── 训练层：QLoRA微调 + 超参数管理 + 训练监控
    ├── 评测层：微调前后对比 + 准确率/F1/BLEU
    ├── 推理层：FastAPI服务 + LoRA权重加载 + 流式输出
    ├── 管理层：模型版本管理 + 训练记录 + 效果对比
    └── 展示层：React前端 + 训练监控 + 模型切换
```

### 技术栈选型

| 层 | 技术 | 费用 |
|----|------|------|
| 基座模型 | Qwen-7B-Chat（开源） | 0元 |
| 微调框架 | PEFT + Transformers + TRL | 0元 |
| GPU | AutoDL RTX 3090 (24G) | ~100元 |
| 后端 | FastAPI + Uvicorn | 0元 |
| 前端 | React + Ant Design | 0元 |
| 数据库 | SQLite（模型管理记录） | 0元 |
| LLM数据生成 | 通义API（已有Key） | ~10元 |
| 部署 | 腾讯云服务器（已有） | 已有 |
| **合计** | | **~120元** |

---

## 四、8周开发计划

### 阶段1：环境准备 + 数据构建（第1周）

| 天 | 任务 | 产出 |
|----|------|------|
| Day 1 | 本地环境搭建 + 项目骨架 | 项目目录 + venv |
| Day 2 | 用通义API批量生成文旅问答数据 | 500条原始数据 |
| Day 3 | 复用MetaAgentsPaaS数据 + 人工编写 | 300条补充数据 |
| Day 4 | 数据清洗 + 去重 + 格式转换 | 1500条Alpaca格式 |
| Day 5 | 划分训练集/验证集/测试集 | 数据集定稿 |
| Day 6-7 | 缓冲时间 + 复习LoRA原理 | 数据集就绪 |

### 阶段2：GPU环境 + 微调训练（第2周）

| 天 | 任务 | 产出 |
|----|------|------|
| Day 8 | AutoDL租GPU + 环境搭建 | GPU环境就绪 |
| Day 9 | 上传数据 + 验证模型加载 | Qwen-7B能跑 |
| Day 10 | 编写QLoRA微调脚本 | 训练脚本 |
| Day 11 | 启动训练 + 监控loss | 训练中 |
| Day 12 | 训练完成 + 保存LoRA权重 | 微调模型 |
| Day 13-14 | 调超参数重训（如需要） | 最优权重 |

### 阶段3：模型评测对比（第3周前半）

| 天 | 任务 | 产出 |
|----|------|------|
| Day 15 | 编写评测脚本 + 50条测试集 | 评测脚本 |
| Day 16 | 跑微调模型评测 + 对比基线 | 评测报告 |

**预期数据**：

| 模型 | 文旅准确率 | 格式合规率 |
|------|-----------|-----------|
| Qwen-7B原始 | ~65% | ~70% |
| Qwen-7B+QLoRA微调 | ~88% | ~95% |

### 阶段4：推理服务封装（第3周后半）

| 天 | 任务 | 产出 |
|----|------|------|
| Day 17 | FastAPI后端骨架 + 模型加载 | API能跑 |
| Day 18 | 对话接口 + 流式输出 + 健康检查 | 完整API |

### 阶段5：Web前端（第4-5周）

| 天 | 任务 | 产出 |
|----|------|------|
| Day 19-20 | React项目搭建 + Ant Design | 前端骨架 |
| Day 21-22 | 对话界面 + 模型切换 | 对话页面 |
| Day 23-24 | 训练记录展示 + 效果对比图表 | 管理页面 |
| Day 25 | 知识图谱可视化（复用MetaAgentsPaaS） | 可视化 |
| Day 26 | 前后端联调 + Bug修复 | 联调通过 |

### 阶段6：Docker化 + 部署（第6周）

| 天 | 任务 | 产出 |
|----|------|------|
| Day 27 | Dockerfile + docker-compose | 容器化 |
| Day 28 | 部署到腾讯云服务器 | 线上可访问 |
| Day 29 | Nginx反代 + 域名（可选） | 正式访问 |
| Day 30 | 验收测试 + 文档 | 部署完成 |

### 阶段7：文档 + 项目收尾（第7-8周）

| 天 | 任务 | 产出 |
|----|------|------|
| Day 31-32 | 使用文档 + 验证文档 | docs/ |
| Day 33-34 | README + 架构图 | 仓库完善 |
| Day 35 | 端到端验收测试 + 文档归档 | 项目收尾 |

---

## 五、详细实施步骤

### 5.1 数据构建（本地，0元）

#### 数据来源

| 来源 | 获取方式 | 数量 |
|------|---------|------|
| 通用指令集 | Alpaca-zh 开源数据集采样 | 500条 |
| 文旅问答 | 用通义API自动生成 | 500条 |
| MetaAgentsPaaS数据 | 复用scenic_spots.json扩展 | 300条 |
| 人工编写 | 针对文旅场景手写 | 200条 |
| **合计** | | **1500条** |

#### 数据格式（Alpaca格式）

```json
{
  "instruction": "介绍一下故宫的开放时间和票价",
  "input": "",
  "output": "故宫博物院的开放时间为：\n- 旺季（4-10月）：08:30-17:00\n- 淡季（11-3月）：08:30-16:30\n\n票价：\n- 旺季：60元\n- 淡季：40元\n\n建议提前在官网预约购票，每日限流8万人。"
}
```

#### 用通义API批量生成数据

```python
# scripts/generate_training_data.py
from openai import OpenAI
import json

client = OpenAI(
    api_key="sk-your-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

scenarios = [
    "景点开放时间和票价查询",
    "景点交通路线规划",
    "景点历史文化介绍",
    "周边美食推荐",
    "住宿推荐",
    "行程规划建议",
    "天气影响游玩建议",
    "门票优惠政策",
]

dataset = []
for scenario in scenarios:
    for i in range(65):  # 每类65条，共520条
        prompt = f"""请生成一条文旅领域的问答数据。
场景：{scenario}
要求：instruction是用户问题，output是详细、准确的回答。
输出JSON格式：{{"instruction":"...","input":"","output":"..."}}"""

        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        item = json.loads(resp.choices[0].message.content)
        dataset.append(item)

with open("data/tourism_train.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"生成 {len(dataset)} 条数据")
```

#### 数据清洗

```python
# scripts/clean_data.py
import json
from datasets import Dataset

with open("data/tourism_train.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

cleaned = []
for item in raw_data:
    instruction = item.get("instruction", "").strip()
    output = item.get("output", "").strip()

    # 过滤过短/过长
    if len(instruction) < 5 or len(output) < 10:
        continue
    if len(output) > 2000:
        continue
    # 去重
    if any(d["instruction"] == instruction for d in cleaned):
        continue

    cleaned.append({
        "instruction": instruction,
        "input": "",
        "output": output
    })

# 转为HuggingFace数据集格式
dataset = Dataset.from_list(cleaned)

# 转为Qwen对话格式
def format_to_chat(example):
    return {
        "text": f"<|im_start|>system\n你是文旅助手，提供准确的旅游信息。<|im_end|>\n<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
    }

dataset = dataset.map(format_to_chat)
dataset.save_to_disk("data/tourism_dataset")

with open("data/tourism_clean.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, ensure_ascii=False, indent=2)

print(f"清洗后 {len(cleaned)} 条数据")
```

---

### 5.2 QLoRA微调（AutoDL GPU，~100元）

#### AutoDL租GPU步骤

1. 打开 https://www.autodl.com 注册
2. 充值100元
3. 选 **RTX 3090 (24GB)**，约1.5元/小时
4. 选镜像 **PyTorch 2.1 + CUDA 11.8**
5. 创建实例，SSH连接

#### 微调脚本

```python
# scripts/finetune_qlora.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
from trl import SFTTrainer

# ===== 1. 加载模型（4bit量化）=====
model_id = "Qwen/Qwen-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,           # QLoRA核心：4bit量化
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# ===== 2. LoRA配置 =====
lora_config = LoraConfig(
    r=8,                          # LoRA秩，常用8/16/32
    lora_alpha=32,                # 缩放因子，一般=r*2
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
        "gate_proj", "up_proj", "down_proj"       # MLP层
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# 打印可训练参数
model.print_trainable_parameters()
# 输出：trainable params: 19,886,080 || all params: 7,737,314,304 || trainable%: 0.257%

# ===== 3. 加载数据 =====
dataset = load_from_disk("data/tourism_dataset")

# ===== 4. 训练参数 =====
training_args = TrainingArguments(
    output_dir="./outputs/qwen-tourism-lora",
    num_train_epochs=3,                    # 训练3轮
    per_device_train_batch_size=4,         # 批大小（3090可以4）
    gradient_accumulation_steps=4,         # 梯度累积，等效batch=16
    learning_rate=2e-4,                    # LoRA学习率
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3,
    fp16=True,                             # 混合精度
    optim="paged_adamw_8bit",              # 8bit优化器，省显存
    report_to="none"
)

# ===== 5. 开始训练 =====
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
    max_seq_length=512,
)

trainer.train()

# ===== 6. 保存LoRA权重 =====
trainer.save_model("./outputs/qwen-tourism-lora")
print("微调完成！LoRA权重已保存")
```

#### 训练监控

```bash
# 训练时另开终端看显存
nvidia-smi -l 5

# 看loss曲线
tensorboard --logdir ./outputs --port 6006
```

**训练耗时**：3090上跑1500条数据3个epoch，约2-4小时，花费约5元。

---

### 5.3 评测对比（AutoDL GPU，~20元）

```python
# scripts/evaluate_finetuned.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

# ===== 1. 加载原始模型 + 微调权重 =====
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat",
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, "./outputs/qwen-tourism-lora")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)

# ===== 2. 测试集（50条）=====
test_questions = [
    {"q": "故宫的开放时间是什么？", "a": "08:30-17:00"},
    {"q": "兵马俑门票多少钱？", "a": "120元"},
    {"q": "黄山怎么去？", "a": "可乘火车到黄山站"},
    # ... 50条测试题
]

# ===== 3. 推理对比 =====
results = []
for item in test_questions:
    prompt = f"<|im_start|>user\n{item['q']}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    results.append({
        "question": item["q"],
        "expected": item["a"],
        "finetuned_answer": answer,
        "contains_answer": item["a"] in answer
    })

# ===== 4. 计算准确率 =====
accuracy = sum(1 for r in results if r["contains_answer"]) / len(results)
print(f"微调后准确率: {accuracy:.1%}")

with open("data/eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

---

### 5.4 推理服务（本地/服务器，0元）

```python
# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

app = FastAPI(title="文旅微调模型服务", version="1.0")

model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model_path = os.getenv("MODEL_PATH", "Qwen/Qwen-7B-Chat")
    lora_path = os.getenv("LORA_PATH", "./outputs/qwen-tourism-lora")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_path, load_in_4bit=True, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, lora_path)
    print("模型加载完成")

class ChatRequest(BaseModel):
    question: str
    max_tokens: int = 200

class ChatResponse(BaseModel):
    answer: str
    model: str
    finetuned: bool

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    prompt = f"<|im_start|>user\n{req.question}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=req.max_tokens, temperature=0.7
        )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return ChatResponse(answer=answer, model="qwen-7b-tourism-lora", finetuned=True)

@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/api/model/info")
async def model_info():
    return {
        "base_model": "Qwen/Qwen-7B-Chat",
        "lora_rank": 8,
        "lora_alpha": 32,
        "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        "trainable_params": "19.9M (0.257%)",
        "total_params": "7.74B",
        "training_data": "1500条文旅指令数据"
    }
```

---

### 5.5 Web前端（本地，0元）

```jsx
// frontend/src/App.jsx
import { useState } from 'react';
import { Input, Button, Card, Tag, Space, Table, Select } from 'antd';

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState('finetuned');

  const askModel = async () => {
    setLoading(true);
    const endpoint = model === 'finetuned' ? '/api/chat' : '/api/chat/base';
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    const data = await res.json();
    setAnswer(data.answer);
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 800, margin: '40px auto' }}>
      <h1>文旅微调模型服务平台</h1>

      <Space.Compact style={{ width: '100%', marginBottom: 16 }}>
        <Select
          defaultValue="finetuned"
          style={{ width: 150 }}
          onChange={setModel}
          options={[
            { value: 'finetuned', label: '微调模型' },
            { value: 'base', label: '原始模型' },
          ]}
        />
        <Input
          placeholder="请输入文旅问题，如：故宫开放时间？"
          value={question}
          onChange={e => setQuestion(e.target.value)}
        />
        <Button type="primary" onClick={askModel} loading={loading}>提问</Button>
      </Space.Compact>

      {answer && (
        <Card style={{ marginTop: 20 }}>
          <Tag color="green">{model === 'finetuned' ? '微调模型' : '原始模型'}</Tag>
          <p style={{ marginTop: 12 }}>{answer}</p>
        </Card>
      )}
    </div>
  );
}
export default App;
```

---

### 5.6 Docker化 + 部署

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3'
services:
  finetune-api:
    build: .
    ports: ["8000:8000"]
    environment:
      - MODEL_PATH=Qwen/Qwen-7B-Chat
      - LORA_PATH=./outputs/qwen-tourism-lora
    volumes:
      - ./outputs:/app/outputs
```

---

## 六、项目目录结构

```
llm-finetune-platform/
├── app/
│   ├── main.py              # FastAPI入口
│   ├── routers/
│   │   ├── chat.py          # 对话接口
│   │   └── model.py         # 模型管理
│   └── services/
│       └── model_service.py # 模型加载/切换
├── scripts/
│   ├── generate_training_data.py  # 数据生成
│   ├── clean_data.py              # 数据清洗
│   ├── finetune_qlora.py          # 微调脚本
│   ├── evaluate_finetuned.py      # 评测脚本
│   └── merge_lora.py              # 权重合并
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # 主页面
│   │   ├── pages/
│   │   │   ├── Chat.jsx    # 对话页
│   │   │   └── Compare.jsx # 效果对比页
│   │   └── components/
│   └── package.json
├── data/
│   ├── tourism_train.json  # 原始数据
│   ├── tourism_clean.json  # 清洗后数据
│   ├── tourism_dataset/    # HF数据集
│   └── eval_results.json   # 评测结果
├── outputs/
│   └── qwen-tourism-lora/  # LoRA权重
├── docs/
│   ├── project-plan.md     # 本文档
│   ├── usage.md            # 使用文档
│   └── verification.md     # 验证文档
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 七、费用预算

| 项目 | 费用 | 说明 |
|------|------|------|
| AutoDL GPU (3090) | ~100元 | 训练+评测约20小时 |
| 通义API数据生成 | ~10元 | 生成1500条数据 |
| 服务器 | 0元 | 已有腾讯云 |
| 前端/后端/数据库 | 0元 | 全部开源免费 |
| **合计** | **~120元** | |

---

## 八、风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| GPU显存不够 | 低 | 3090(24G)跑7B+QLoRA足够，不够就换A10 |
| 微调效果不好 | 中 | 调rank/learning_rate/数据量，或换Qwen-14B |
| 训练loss不收敛 | 低 | 检查数据格式、学习率、warmup |
| 过拟合 | 中 | 加验证集监控，减少epoch，加dropout |
| AutoDL账号问题 | 低 | 备选恒源云/矩池云 |
