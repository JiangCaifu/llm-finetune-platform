# Docker 部署指南

## 部署形态选择

本项目支持三种运行环境，用途不同，不要混用：

| 环境 | 形态 | 用途 | 理由 |
|------|------|------|------|
| PAI-DSW / AutoDL 等托管 GPU 平台 | **裸机**（venv + uvicorn） | 训练、推理验证 | 平台实例内部无 docker daemon，不支持嵌套 Docker |
| 自有 GPU 服务器 | **Docker** | 标准交付形态 | 环境一致，一次构建到处运行；模型通过卷挂载，不入镜像 |
| 普通 CPU 服务器 / CI | **仅构建验证** | 验证 Dockerfile 可用性 | 无需 GPU 即可完成镜像构建与依赖校验 |

裸机启动方式见 [裸机部署](#五裸机部署训练平台)。

---

## 前置条件

### 硬件要求

| 场景 | 资源 | 说明 |
|------|------|------|
| GPU 推理 | 显存 **16GB+** | Qwen2-7B-Instruct bf16 权重约 15GB，推理需额外 KV cache 空间 |
| CPU 推理 | 内存 **18GB+** | 权重以 bf16 常驻内存，另需运行时开销；速度慢，仅供调试 |
| 镜像构建 | 磁盘 **30GB+** | 最终镜像约 12GB（CUDA 基础镜像 + torch cu121），构建缓存另需约 12GB；若已存在旧镜像需要重建，建议留 40GB |

> 显存与内存不可互换。GPU 机器上模型占用的是显存，CPU 机器上占用的是内存，两者都需要 16GB 以上才能装下 7B 模型。

### 软件要求

| 依赖 | 版本要求 | 说明 |
|------|---------|------|
| Docker | 20.10+ | [安装指南](https://docs.docker.com/get-docker/) |
| Docker Compose | v2.20+ | GPU 部署使用 `docker compose -f` 叠加配置 |
| NVIDIA Container Toolkit | 最新 | 仅 GPU 部署需要，[安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| NVIDIA 驱动 | 支持 CUDA 12.1 | 仅 GPU 部署需要，`nvidia-smi` 可验证 |

---

## 目录结构

```
llm-finetune-platform/
├── Dockerfile                # 多阶段构建：前端 + 后端
├── docker-compose.yml        # 基础编排（CPU 兼容）
├── docker-compose.gpu.yml    # GPU 设备声明（叠加使用）
├── .dockerignore            # 排除大文件
├── app/                      # FastAPI 后端
├── frontend/                 # React 前端源码
├── scripts/                  # 训练/评测脚本
├── outputs/                  # LoRA 权重（训练产物，挂载进容器）
└── requirements-api.txt      # Python 依赖
```

---

## 一、准备模型和权重

模型与代码分离：基座模型和 LoRA 权重都通过卷挂载进容器，不打进镜像。

```bash
# 基座模型（Qwen2-7B-Instruct）
ls /path/to/Qwen2-7B-Instruct/
# 应包含：config.json, model-*.safetensors, tokenizer.json 等

# LoRA 权重（训练产物）
ls outputs/qwen-tourism-lora/
# 应包含：adapter_config.json, adapter_model.safetensors
```

LoRA 权重的获取与校验见 [release-guide.md](release-guide.md)。

---

## 二、GPU 部署

### 1. 启动

```bash
# 指定基座模型在宿主机上的路径
export MODEL_HOST_PATH=/path/to/Qwen2-7B-Instruct

# 叠加 GPU 配置启动
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

首次启动会自动构建镜像，构建完成后立即启动。

### 2. 验证服务

```bash
# 健康检查
curl http://localhost:8000/api/v1/health
# 期望：{"status":"ok","model_loaded":true,"gpu_available":true,"gpu_memory_used_gb":14.26}

# 文旅问答
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "北京有什么景点？", "max_new_tokens": 256}'

# 浏览器访问前端
open http://localhost:8000
```

### 3. 查看日志 / 停止

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
```

---

## 三、无 GPU 环境验证构建

CPU 服务器上不需要模型权重，也能验证 Dockerfile 是否可用。这是 CI 或临时机器上唯一可行的验证方式。

### 1. 构建镜像

```bash
git clone https://github.com/JiangCaifu/llm-finetune-platform.git
cd llm-finetune-platform
docker build -t tourism-llm:latest .
```

构建耗时主要在下载依赖（国内网络约 10-30 分钟）：

| 阶段 | 内容 | 体积 |
|------|------|------|
| 阶段 1 | `node:20-slim` + npm ci + vite build | 约 400MB（构建层，不进最终镜像） |
| 阶段 2 | `nvidia/cuda:12.1.0-runtime-ubuntu22.04` + torch cu121 | 约 8GB |

构建完成后清理构建缓存，回收磁盘：

```bash
docker builder prune -f
```

注意不要用 `docker system prune -a`：它会连没有容器引用的**镜像本身**一起删掉（此时 `tourism-llm:latest` 还没被任何容器使用，会被误删）。

### 2. 校验镜像内容

```bash
# torch 必须是 GPU 版：版本号带 +cu121 后缀
docker run --rm tourism-llm:latest python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
# 期望：2.3.1+cu121 / 12.1
# 若输出 2.3.1（无后缀），说明装成了 CPU wheel，镜像不可用于 GPU 推理

# 推理依赖版本必须与训练环境一致（上界已由 requirements-api.txt 钉死）
docker run --rm tourism-llm:latest python -c \
  "import transformers, peft, accelerate, numpy; \
   print('transformers', transformers.__version__); \
   print('peft', peft.__version__); \
   print('accelerate', accelerate.__version__); \
   print('numpy', numpy.__version__)"
# 期望：transformers 4.55.4 / peft 0.18.x / accelerate 0.34.x / numpy 1.x
# 若 transformers 是 5.x → LoRA 适配代码里的 torch_dtype 会直接报错
# 若 numpy 是 2.x → import torch 时 ABI 冲突（Failed to initialize NumPy）

# 前端产物与后端代码应已就位
docker run --rm tourism-llm:latest ls /app/frontend/dist/ /app/app/
```

### 3. 启动容器（预期失败）

CPU 机器内存不足时容器无法加载 7B 模型，日志会出现 OOM 或模型路径缺失：

```bash
docker run -d --name tourism-test -p 8000:8000 \
  -v /path/to/Qwen2-7B-Instruct:/models/Qwen2-7B-Instruct:ro \
  -v $PWD/outputs/qwen-tourism-lora:/app/outputs/qwen-tourism-lora:ro \
  tourism-llm:latest

docker logs -f tourism-test
```

这是预期行为，不代表部署有误。**完整的推理验证需要 GPU 环境。**

### 4. 本项目的实际验证状态

以下结论来自一台无 GPU、内存 3.6GB 的 ECS（Docker 29.3.1，可用磁盘 22GB）：

| 验证项 | 状态 | 证据 |
|--------|------|------|
| Dockerfile 20 步完整执行 | ✅ 已验证 | 多阶段构建成功，产出 12.1GB 镜像 |
| torch cu121 wheel 安装正确 | ✅ 已验证 | `torch 2.3.1+cu121`、`torch.version.cuda == 12.1` |
| numpy 与 torch ABI 兼容 | ✅ 已验证 | `numpy 1.26.4` 下 `import torch` 无报错 |
| 前端产物 / 后端代码进镜像 | ✅ 已验证 | `frontend/dist/index.html`、`app/main.py` 均在镜像内 |
| 依赖版本与训练环境一致 | ⚠️ 部分 | peft 0.18.1 / accelerate 0.34.2 / numpy 1.26.4 一致；transformers 实际解析为 4.57.6（约束已收到 `<4.56.0`，该机器磁盘不足未能重建复验） |
| 模型加载 + 真实推理 | ❌ 未验证 | 该机器无 GPU、内存 3.6GB，7B 模型无法加载 |
| Docker GPU 运行时路径 | ❌ 未验证 | 无支持 NVIDIA Container Toolkit 的机器，`docker-compose.gpu.yml` 未实测 |

> 结论：**镜像可构建、内容正确，但「起服务 → 加载模型 → 返回回答」这条链路没有在 GPU 上实测过。** 要完整闭环，请在 GPU 机器上执行第二节，或改用第五节的裸机部署。

---

## 四、配置项

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_HOST_PATH` | `/models/Qwen2-7B-Instruct` | 宿主机上的基座模型目录 |
| `BASE_MODEL_PATH` | `/models/Qwen2-7B-Instruct` | 容器内的基座模型路径 |
| `LORA_PATH` | `/app/outputs/qwen-tourism-lora` | 容器内的 LoRA 权重路径 |

也可以在项目根目录创建 `.env` 文件统一管理：

```env
MODEL_HOST_PATH=/your/model/path
BASE_MODEL_PATH=/models/Qwen2-7B-Instruct
```

---

## 五、裸机部署（训练平台）

PAI-DSW、AutoDL 等托管平台的实例内部没有 docker daemon，无法在容器里再跑容器。这类环境用裸机方式运行。

```bash
# 1. 安装依赖
pip install -r requirements-api.txt

# 2. 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. 验证
curl http://localhost:8000/api/v1/health
```

前端需要单独构建并由 FastAPI 一并 serve：

```bash
cd frontend && npm install && npm run build && cd ..
```

裸机与 Docker 的区别只在于依赖装在哪、进程怎么起，代码路径完全一致。

---

## Dockerfile 设计说明

### 多阶段构建

```
阶段 1: node:20-slim                → npm ci → npm run build → frontend/dist/
阶段 2: nvidia/cuda:12.1.0-runtime  → pip install → copy app + frontend/dist
```

**为什么多阶段？**
- 前端构建工具（Node、npm、TypeScript）不进入最终镜像，减小镜像体积
- 最终镜像只包含运行时必需的 Python + 编译好的静态文件

### 依赖来源

torch 必须从 PyTorch 官方索引安装：

```dockerfile
RUN pip3 install --no-cache-dir torch==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

第三方镜像源同步的是默认 PyPI 版本，不带 `+cu121` 后缀，装出来是 CPU wheel，容器内即使有 GPU 也无法使用。其余依赖走清华源加速。

### 镜像分层缓存

Dockerfile 先复制 `package.json` 和 `requirements-api.txt`，再复制源码。这样修改代码后重新 build，依赖安装步骤会被缓存跳过，构建更快。

### 模型不入镜像

Qwen2-7B 基座模型约 15GB，LoRA 权重约 80MB。如果把模型打入镜像：
- 镜像体积 15GB+，推拉极慢
- 每次模型更新都要重新 build

因此模型通过 `volumes` 挂载进容器，镜像只包含代码和依赖。

### 健康检查

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3
```

`start-period` 给到 180 秒，因为 7B 模型加载需要 1-2 分钟，过短会让容器在模型就绪前就被判定为 `unhealthy`。
