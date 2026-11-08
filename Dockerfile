# ============================================================
# Dockerfile：文旅问答服务（多阶段构建）
# ============================================================
# 构建：docker build -t tourism-llm:latest .
# 运行：docker compose up（见 docker-compose.yml）
# ============================================================

# ===== 阶段 1：构建前端静态文件 =====
FROM node:20-slim AS frontend-builder

WORKDIR /build

# 先复制 package.json，利用 Docker 缓存层
COPY frontend/package.json frontend/package-lock.json* ./

RUN npm config set registry https://registry.npmmirror.com && \
    npm ci || npm install

# 复制前端源码，构建静态文件
COPY frontend/ ./

RUN npm run build

# ===== 阶段 2：Python 运行时（GPU 推理） =====
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 Python 3.11 + 基础工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 创建 Python 软链
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# 安装 Python 依赖（先复制 requirements，利用缓存层）
COPY requirements-api.txt ./

RUN pip3 install --no-cache-dir -r requirements-api.txt

# 复制后端代码
COPY app/ ./app/
COPY scripts/ ./scripts/

# 从阶段 1 复制前端构建产物
COPY --from=frontend-builder /build/dist ./frontend/dist

# 环境变量默认值（可在 docker-compose.yml 或 docker run 时覆盖）
ENV BASE_MODEL_PATH=/models/Qwen2-7B-Instruct \
    LORA_PATH=/app/outputs/qwen-tourism-lora \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# 健康检查（容器启动后 60 秒开始检测，每次间隔 30 秒）
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
