"""
应用配置：模型路径、生成参数、服务配置
所有参数可通过环境变量覆盖，方便 Docker 部署时调整
"""

import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Settings:
    """应用配置（可通过环境变量覆盖）"""

    # ===== 模型配置 =====
    base_model_path: str = field(
        default_factory=lambda: os.getenv(
            "BASE_MODEL_PATH",
            "/mnt/workspace/models/models/qwen--Qwen2-7B-Instruct/snapshots/master",
        )
    )
    lora_path: str = field(
        default_factory=lambda: os.getenv(
            "LORA_PATH",
            str(Path(__file__).parent.parent / "outputs" / "qwen-tourism-lora"),
        )
    )

    # ===== 生成参数（默认值，可被请求参数覆盖）=====
    default_max_new_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    max_max_new_tokens: int = 2048  # 上限，防止滥用

    # ===== 服务配置 =====
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list = field(default_factory=lambda: ["*"])  # 开发环境允许所有来源

    # ===== 系统提示词 =====
    system_prompt: str = "你是文旅助手，提供准确、详细的旅游信息和建议。"


settings = Settings()
