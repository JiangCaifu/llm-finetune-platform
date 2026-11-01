"""
文旅问答接口：POST /api/v1/chat
"""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.schemas import ChatRequest, ChatResponse
from app.model_service import model_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    文旅问答接口

    - 接收用户问题，返回微调模型的回答
    - 推理在线程池中执行，不阻塞事件循环
    """
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="模型正在加载中，请稍后重试")

    try:
        # 推理是 CPU/GPU 密集型任务，放到线程池避免阻塞 FastAPI 事件循环
        answer, latency_ms, tokens_generated = await run_in_threadpool(
            model_service.generate,
            question=request.question,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        logger.info(
            f"chat: question={request.question[:40]}... "
            f"tokens={tokens_generated} latency={latency_ms:.0f}ms"
        )

        model_name = "qwen2-7b-lora" if model_service.lora_loaded else "qwen2-7b-base"
        return ChatResponse(
            answer=answer,
            model=model_name,
            latency_ms=latency_ms,
            tokens_generated=tokens_generated,
        )
    except Exception as e:
        logger.error(f"chat 推理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")
