"""
健康检查 & 模型信息接口
"""

import logging

from fastapi import APIRouter

from app.config import settings
from app.schemas import HealthResponse, ModelInfoResponse
from app.model_service import model_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health():
    """
    健康检查

    - 用于容器编排（K8s/Docker）的存活探针
    - 返回模型加载状态和 GPU 信息
    """
    gpu_available, gpu_mem_gb = model_service.gpu_info()
    return HealthResponse(
        status="ok" if model_service.is_loaded() else "loading",
        model_loaded=model_service.is_loaded(),
        gpu_available=gpu_available,
        gpu_memory_used_gb=round(gpu_mem_gb, 2),
    )


@router.get("/info", response_model=ModelInfoResponse)
async def info():
    """
    模型信息

    - 返回基座模型、LoRA 路径、系统提示词等元信息
    """
    return ModelInfoResponse(
        base_model=settings.base_model_path,
        lora_path=settings.lora_path,
        lora_loaded=model_service.lora_loaded,
        system_prompt=settings.system_prompt,
    )
