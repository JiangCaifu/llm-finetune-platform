"""
合并 LoRA 权重到基座模型
========================
将 LoRA adapter 合并到 Qwen2-7B-Instruct 基座，导出完整模型权重。

用途：
  - 推理时无需再加载 LoRA，直接加载合并后的模型
  - 部署到 vLLM 等推理引擎时必须先合并

运行（在阿里云 DSW / AutoDL GPU 实例上）：
  python scripts/merge_lora.py
  python scripts/merge_lora.py --output ./outputs/qwen-tourism-merged
"""

import os
import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--lora_path", type=str, default="outputs/qwen-tourism-lora")
    parser.add_argument("--output", type=str, default="outputs/qwen-tourism-merged")
    parser.add_argument("--quantize", action="store_true", help="合并后量化为 4bit (GPTQ)")
    args = parser.parse_args()

    print(f"[1/4] 加载 Tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"[2/4] 加载基座模型: {args.model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"[3/4] 加载并合并 LoRA: {args.lora_path}")
    model = PeftModel.from_pretrained(base, args.lora_path)
    model = model.merge_and_unload()

    print(f"[4/4] 保存合并后的模型: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        shutil.rmtree(output_path)

    model.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(str(output_path))

    size_gb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1024**3
    print(f"\n✓ 合并完成")
    print(f"  路径: {output_path}")
    print(f"  大小: {size_gb:.2f} GB")
    print(f"\n下一步：")
    print(f"  - 用合并后的模型启动推理服务：")
    print(f"    MODEL_PATH={output_path} python app/main.py")
    print(f"  - 或上传到 HuggingFace Hub")


if __name__ == "__main__":
    main()
