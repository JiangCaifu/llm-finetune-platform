"""
模型服务：加载基座模型 + LoRA 权重，提供推理能力
核心组件，在 FastAPI 启动时加载一次，后续请求复用
"""

import time
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from app.config import settings

logger = logging.getLogger(__name__)


class ModelService:
    """模型加载与推理服务（单例，FastAPI 启动时初始化）"""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lora_loaded = False
        self._loaded = False

    def load(self):
        """加载模型（启动时调用一次）"""
        logger.info(f"开始加载模型: base={settings.base_model_path}")

        # 1. 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.base_model_path,
            padding_side="left",  # 推理用 left padding
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. 加载基座模型（bf16 全精度，与训练保持一致）
        base_model = AutoModelForCausalLM.from_pretrained(
            settings.base_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # 3. 加载 LoRA 权重（如果存在）
        lora_path = Path(settings.lora_path)
        if lora_path.exists() and (lora_path / "adapter_model.safetensors").exists():
            self.model = PeftModel.from_pretrained(base_model, str(lora_path))
            self.lora_loaded = True
            logger.info(f"LoRA 权重已加载: {lora_path}")
        else:
            self.model = base_model
            self.lora_loaded = False
            logger.warning(f"LoRA 权重不存在: {lora_path}，仅使用基座模型")

        self.model.eval()
        self._loaded = True

        if self.device == "cuda":
            mem_gb = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"模型加载完成，显存占用: {mem_gb:.2f} GB")
        else:
            logger.info("模型加载完成（CPU 模式）")

    def generate(
        self,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> tuple[str, float, int]:
        """
        生成回答

        Returns:
            (answer, latency_ms, tokens_generated)
        """
        if not self._loaded:
            raise RuntimeError("模型未加载，请等待服务启动完成")

        start_time = time.time()

        # 构造 ChatML 输入（与训练/评测脚本一致）
        messages = [
            {"role": "system", "content": settings.system_prompt},
            {"role": "user", "content": question},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 只取新生成的部分
        generated_tokens = outputs[0][input_len:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        latency_ms = (time.time() - start_time) * 1000
        tokens_generated = len(generated_tokens)

        return answer, latency_ms, tokens_generated

    def is_loaded(self) -> bool:
        return self._loaded

    def gpu_info(self) -> tuple[bool, float]:
        """返回 (gpu_available, gpu_memory_used_gb)"""
        if torch.cuda.is_available():
            return True, torch.cuda.memory_allocated() / 1024**3
        return False, 0.0


# 全局单例
model_service = ModelService()
