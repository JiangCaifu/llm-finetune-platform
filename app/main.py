"""
FastAPI 应用入口
================

启动方式（在 DSW / GPU 实例上）：
  cd ~/llm-finetune-platform
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

API 文档（启动后访问）：
  - Swagger UI:  http://localhost:8000/docs
  - ReDoc:       http://localhost:8000/redoc

接口列表：
  POST /api/v1/chat     文旅问答
  GET  /api/v1/health   健康检查
  GET  /api/v1/info     模型信息
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from app.config import settings
from app.model_service import model_service
from app.routers import chat, health

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载模型，关闭时清理"""
    # 启动
    logger.info("=" * 60)
    logger.info("文旅问答服务启动中...")
    logger.info(f"基座模型: {settings.base_model_path}")
    logger.info(f"LoRA 权重: {settings.lora_path}")
    logger.info("=" * 60)

    model_service.load()

    logger.info("服务就绪，等待请求...")
    logger.info(f"API 文档: http://localhost:{settings.port}/docs")

    yield

    # 关闭
    logger.info("服务关闭中...")


app = FastAPI(
    title="文旅问答 API",
    description="基于 Qwen2-7B-Instruct + LoRA 微调的文旅问答推理服务",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router)
app.include_router(health.router)


# ===== 前端静态文件（生产模式）=====
# 前端 build 后产物在 frontend/dist/，FastAPI 直接 serve
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    # 挂载静态资源（js/css/图片等）
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    # 其他所有路径返回 index.html（SPA 路由）
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # 先尝试匹配静态文件
        file_path = frontend_dist / full_path
        if full_path and file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        # 否则返回 index.html（前端路由）
        return FileResponse(frontend_dist / "index.html")

    logger.info(f"前端静态文件已挂载: {frontend_dist}")
else:
    logger.info("前端未 build（frontend/dist 不存在），仅提供 API 服务")


# ===== 根路径处理 =====
# 前端未 build 时，返回 API 简介（开发模式）
if not frontend_dist.exists():
    @app.get("/")
    async def root():
        """根路径：返回服务简介（仅 API 模式）"""
        return {
            "service": "文旅问答 API",
            "version": "1.0.0",
            "docs": "/docs",
            "endpoints": {
                "chat": "POST /api/v1/chat",
                "health": "GET /api/v1/health",
                "info": "GET /api/v1/info",
            },
        }
