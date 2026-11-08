# Docker 部署指南

## 前置条件

| 依赖 | 版本要求 | 说明 |
|------|---------|------|
| Docker | 20.10+ | [安装指南](https://docs.docker.com/get-docker/) |
| NVIDIA Container Toolkit | 最新 | GPU 部署需要，[安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| GPU 显存 | 16GB+ | Qwen2-7B-Instruct bf16 推理约 14GB |

---

## 目录结构

```
llm-finetune-platform/
├── Dockerfile              # 多阶段构建：前端 + 后端
├── docker-compose.yml      # GPU 部署编排
├── .dockerignore           # 排除大文件
├── app/                    # FastAPI 后端
├── frontend/               # React 前端源码
├── scripts/                # 训练/评测脚本
├── outputs/                # LoRA 权重（训练产物，挂载进容器）
└── requirements-api.txt    # Python 依赖
```

---

## 快速部署（GPU）

### 1. 准备模型和权重

确保以下两个目录存在：

```bash
# 基座模型（Qwen2-7B-Instruct）
ls /mnt/workspace/models/models/qwen--Qwen2-7B-Instruct/snapshots/master/
# 应包含：config.json, model-*.safetensors, tokenizer.json 等

# LoRA 权重（训练产物）
ls outputs/qwen-tourism-lora/
# 应包含：adapter_config.json, adapter_model.safetensors, tokenizer 文件等
```

### 2. 一键启动

```bash
docker compose up -d
```

首次启动会自动构建镜像（约 5-10 分钟），后续启动直接复用缓存。

### 3. 验证服务

```bash
# 健康检查
curl http://localhost:8000/api/v1/health

# 文旅问答
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "北京有什么景点？"}'

# 浏览器访问前端
open http://localhost:8000
```

### 4. 查看日志 / 停止

```bash
docker compose logs -f        # 实时日志
docker compose down           # 停止并清理
docker compose restart        # 重启
```

---

## 自定义配置

### 修改模型路径

在项目根目录创建 `.env` 文件：

```env
MODEL_HOST_PATH=/your/model/path
BASE_MODEL_PATH=/models/Qwen2-7B-Instruct
```

或直接在 `docker-compose.yml` 的 `volumes` 中修改挂载路径。

### CPU 模式（仅供调试，速度很慢）

修改 `docker-compose.yml`，注释掉 `deploy` 段：

```yaml
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
```

---

## Dockerfile 设计说明

### 多阶段构建

```
阶段 1: node:20-slim     → npm ci → npm run build → frontend/dist/
阶段 2: nvidia/cuda:12.1 → pip install → copy app + frontend/dist
```

**为什么多阶段？**
- 前端构建工具（Node、npm、TypeScript）不进入最终镜像，减小镜像体积
- 最终镜像只包含运行时必需的 Python + 编译好的静态文件

### 镜像分层缓存

Dockerfile 先复制 `package.json` 和 `requirements-api.txt`，再复制源码。这样修改代码后重新 build，依赖安装步骤会被缓存跳过，构建更快。

### 模型不入镜像

Qwen2-7B 基座模型约 14GB，LoRA 权重约 80MB。如果把模型打入镜像：
- 镜像体积 14GB+，推拉极慢
- 每次模型更新都要重新 build

因此模型通过 `volumes` 挂载进容器，镜像只包含代码和依赖。
