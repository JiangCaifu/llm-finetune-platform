"""
Pydantic 请求/响应模型：定义 API 的输入输出结构
"""

from pydantic import BaseModel, Field


# ===== 请求模型 =====


class ChatRequest(BaseModel):
    """文旅问答请求"""

    question: str = Field(..., min_length=1, max_length=500, description="用户问题")
    max_new_tokens: int = Field(
        default=512, ge=64, le=2048, description="生成的最大 token 数"
    )
    temperature: float = Field(
        default=0.7, ge=0.1, le=2.0, description="采样温度，越高越随机"
    )
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="核采样阈值")


# ===== 响应模型 =====


class ChatResponse(BaseModel):
    """文旅问答回复"""

    answer: str = Field(..., description="模型生成的回答")
    model: str = Field(..., description="模型标识")
    latency_ms: float = Field(..., description="推理耗时（毫秒）")
    tokens_generated: int = Field(..., description="生成的 token 数")


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = Field(..., description="服务状态")
    model_loaded: bool = Field(..., description="模型是否已加载")
    gpu_available: bool = Field(..., description="GPU 是否可用")
    gpu_memory_used_gb: float = Field(0.0, description="GPU 显存占用（GB）")


class ModelInfoResponse(BaseModel):
    """模型信息响应"""

    base_model: str = Field(..., description="基座模型路径")
    lora_path: str = Field(..., description="LoRA 权重路径")
    lora_loaded: bool = Field(..., description="是否成功加载 LoRA")
    system_prompt: str = Field(..., description="系统提示词")


class ErrorResponse(BaseModel):
    """错误响应"""

    detail: str = Field(..., description="错误信息")
    error_type: str = Field(..., description="错误类型")
