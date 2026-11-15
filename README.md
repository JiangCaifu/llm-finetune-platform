# 文旅垂直领域大模型微调与服务平台

基于 Qwen2-7B-Instruct 的 LoRA 微调，封装为 FastAPI 推理服务 + React 前端的端到端应用。

## 项目简介

针对通用大模型在文旅领域回答泛化、缺乏结构化的问题，使用 LoRA 对 Qwen2-7B-Instruct 进行垂直领域微调，并构建完整的推理服务平台。

**端到端流程**：数据构建 → LoRA 微调 → 评测对比 → FastAPI 推理服务 → React 前端 → Docker 部署

## 训练效果

| 指标 | 原始模型 | 微调模型 | 变化 |
|------|---------|---------|------|
| Train Loss | — | **1.20** | loss 从 1.96 → 0.86 持续下降 |
| Eval Loss | — | **1.22** | 三轮 eval：1.32 → 1.22 → 1.24 |
| Mean Token Accuracy | 0.50 | **0.75** | +50% |
| 平均回答长度 | 502 字 | 403 字 | 更聚焦、更结构化 |
| 关键词命中率 | 0.0% | 4.8% | 从 0 到有 |
| 格式合规率 | 100% | 100% | 格式保持 |

**训练参数**：3 epoch, 560 条训练数据, LoRA rank=8, target_modules 覆盖 attention + MLP, 可训练参数 20M (0.26%)

## 技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| 基座模型 | Qwen2-7B-Instruct | 7B 参数, ChatML 格式 |
| 微调方法 | LoRA (PEFT) | rank=8, 可训练参数仅 0.26% |
| 训练框架 | Transformers + TRL + PEFT | SFTTrainer, bf16 精度 |
| GPU | Aliyun PAI-DSW (A10 24GB) | 单卡 A10 24GB |
| 后端 | FastAPI + Uvicorn | 异步推理, lifespan 管理模型 |
| 前端 | React + TypeScript + Vite | Tailwind CSS, Markdown 渲染 |
| 部署 | Docker + docker-compose | 多阶段构建, GPU 支持 |

## 项目结构

```
llm-finetune-platform/
├── app/                        # FastAPI 后端
│   ├── main.py                 # 入口（lifespan 加载模型 + 静态文件 serve）
│   ├── config.py               # 环境变量配置
│   ├── schemas.py              # Pydantic 请求/响应模型
│   ├── model_service.py        # 模型加载 + 推理
│   └── routers/
│       ├── chat.py             # POST /api/v1/chat
│       └── health.py           # GET /api/v1/health
├── frontend/                   # React 前端
│   ├── src/
│   │   ├── App.tsx             # 主界面
│   │   ├── components/
│   │   │   ├── MessageBubble.tsx
│   │   │   └── InputBox.tsx
│   │   └── api.ts              # API 调用
│   └── package.json
├── scripts/                    # 训练/评测脚本
│   ├── finetune_qlora.py       # LoRA 微调
│   ├── evaluate_finetuned.py   # 微调前后对比评测
│   ├── merge_lora.py           # LoRA 权重合并
│   └── diagnose_gradient.py    # 梯度诊断工具
├── data/                       # 数据集
│   ├── tourism_train.json      # 560 条训练数据
│   ├── tourism_val.json        # 70 条验证数据
│   └── tourism_test.json       # 70 条测试数据
├── docs/                       # 文档
│   ├── project-plan.md         # 项目计划
│   ├── deployment.md           # Docker 部署指南
│   └── troubleshooting-grad-norm-zero.md  # 训练踩坑记录
├── Dockerfile                  # 多阶段构建
├── docker-compose.yml          # GPU 部署编排
└── requirements-api.txt        # Python 依赖
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/JiangCaifu/llm-finetune-platform.git
cd llm-finetune-platform

# 下载 LoRA 权重（从 GitHub Release）
# 详见 Releases 页面，下载后放入 outputs/qwen-tourism-lora/
```

### 2. Docker 一键部署（GPU）

```bash
# 设置基座模型路径
export MODEL_HOST_PATH=/path/to/Qwen2-7B-Instruct

# 启动
docker compose up -d

# 验证
curl http://localhost:8000/api/v1/health
```

浏览器访问 `http://localhost:8000` 即可使用文旅问答服务。

### 3. 开发模式

```bash
# 后端
pip install -r requirements-api.txt
python -m uvicorn app.main:app --reload

# 前端（Mock 模式，不需要 GPU）
cd frontend
npm install
npm run dev:mock
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 健康检查（模型状态、显存占用） |
| `/api/v1/info` | GET | 模型信息（基座、LoRA 配置） |
| `/api/v1/chat` | POST | 文旅问答推理 |

**请求示例**：

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "北京有什么景点？", "max_new_tokens": 256}'
```

**响应示例**：

```json
{
  "answer": "北京作为千年古都...",
  "model": "qwen2-7b-lora",
  "latency_ms": 10720,
  "tokens_generated": 238
}
```

## 工程设计要点

### 1. 训练问题诊断与修复

训练时遇到 `grad_norm=0`（模型不学习），通过编写诊断脚本定位根因：gradient checkpointing 的 `use_reentrant=True` 模式在 LoRA 下不构建反向图。修复方案：`use_reentrant=False` + `enable_input_require_grads()`。

详见 [docs/troubleshooting-grad-norm-zero.md](docs/troubleshooting-grad-norm-zero.md)

### 2. 多阶段 Docker 构建

前端用 Node 构建静态文件，后端用 nvidia/cuda 运行时，最终镜像不含构建工具，体积小、启动快。

### 3. 模型与代码分离

LoRA 权重通过 GitHub Release 分发，基座模型通过 volume 挂载，镜像只包含代码和依赖。符合"代码与产物分离"的工程原则。

### 4. 前后端一体化部署

FastAPI 同时 serve API 和前端静态文件，单端口部署，无跨域问题，Docker 一键启动。

## 文档

- [项目计划](docs/project-plan.md) - 8 周开发计划与技术栈选型
- [Docker 部署指南](docs/deployment.md) - GPU/CPU 部署方式
- [训练踩坑记录](docs/troubleshooting-grad-norm-zero.md) - grad_norm=0 根因分析与修复
- [AutoDL/DSW 配置](docs/autodl-setup.md) - GPU 环境搭建指南

## License

MIT
