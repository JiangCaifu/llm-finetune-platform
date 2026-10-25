"""
QLoRA 微调脚本（在 AutoDL GPU 实例上运行）
============================================

功能：
  1. 加载 Qwen-7B-Chat 基座（4bit 量化）
  2. 应用 LoRA 配置（rank=8, alpha=32）
  3. 加载文旅数据集（Alpaca 格式）
  4. 使用 SFTTrainer 训练
  5. 保存 LoRA 权重

前置条件：
  1. AutoDL 实例：RTX 3090 (24GB) 或 A10 (24GB)
  2. 镜像：PyTorch 2.1 + CUDA 11.8
  3. 安装依赖：pip install -r requirements-gpu.txt
  4. 数据集已上传到 ~/llm-finetune-platform/data/

运行：
  cd ~/llm-finetune-platform
  python scripts/finetune_qlora.py
  python scripts/finetune_qlora.py --epochs 5  # 调整 epoch
  python scripts/finetune_qlora.py --rank 16   # 调整 LoRA rank

预计耗时（RTX 3090）：
  - 700条数据，3个epoch，约 30-60 分钟
  - 显存占用 ~16GB（4bit + LoRA + batch=4 + 梯度累积=4）

输出：
  outputs/qwen-tourism-lora/
    - adapter_model.safetensors  (LoRA 权重)
    - adapter_config.json        (LoRA 配置)
    - tokenizer.json
"""

import os
import json
import argparse
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import Dataset
from trl import SFTTrainer, SFTConfig


# ============ Qwen ChatML 模板 ============
SYSTEM_PROMPT = "你是文旅助手，提供准确、详细的旅游信息和建议。"

# Qwen-7B-Chat 的 ChatML 模板（手动构造，避免依赖 tokenizer.chat_template）
CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n{assistant}<|im_end|>"
)


def format_to_chat(example: dict, tokenizer=None) -> str:
    """把 Alpaca 数据格式化为 Qwen ChatML（手动构造，不依赖 tokenizer.chat_template）"""
    text = CHATML_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        user=example["instruction"],
        assistant=example["output"],
    )
    return {"text": text}


def load_dataset_split(tokenizer, data_dir: str = "data"):
    """加载训练集和验证集"""
    train_path = Path(data_dir) / "tourism_train.json"
    val_path = Path(data_dir) / "tourism_validation.json"

    if not train_path.exists():
        raise FileNotFoundError(
            f"训练集不存在: {train_path}\n"
            f"请先在本地运行 scripts/clean_data.py 生成数据集，"
            f"然后上传 data/ 目录到 AutoDL"
        )

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    train_ds = Dataset.from_list(train_data).map(
        lambda x: format_to_chat(x, tokenizer),
        remove_columns=train_data[0].keys() if train_data else ["instruction", "input", "output"],
    )
    val_ds = Dataset.from_list(val_data).map(
        lambda x: format_to_chat(x, tokenizer),
        remove_columns=val_data[0].keys() if val_data else ["instruction", "input", "output"],
    )

    print(f"训练集: {len(train_ds)} 条")
    print(f"验证集: {len(val_ds)} 条")

    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description="QLoRA 微调 Qwen-7B-Chat 文旅模型")
    parser.add_argument("--model", type=str, default="Qwen/Qwen-7B-Chat",
                        help="基座模型路径")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="数据目录")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen-tourism-lora",
                        help="输出目录")
    parser.add_argument("--rank", type=int, default=8, help="LoRA 秩")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批大小")
    parser.add_argument("--grad_accum", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--save_steps", type=int, default=100, help="保存间隔步数")
    args = parser.parse_args()

    print("=" * 60)
    print("QLoRA 微调 Qwen-7B-Chat 文旅模型")
    print("=" * 60)
    print(f"基座模型: {args.model}")
    print(f"LoRA rank: {args.rank}, alpha: {args.alpha}")
    print(f"训练轮数: {args.epochs}")
    print(f"批大小: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"学习率: {args.lr}")
    print(f"最大序列长度: {args.max_seq_length}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    # ===== 1. 加载 Tokenizer =====
    print("\n[1/6] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer 加载完成，vocab_size = {len(tokenizer)}")

    # ===== 2. 加载数据 =====
    print("\n[2/6] 加载数据集...")
    train_ds, val_ds = load_dataset_split(tokenizer, args.data_dir)
    print(f"数据集加载完成")

    # ===== 3. 加载基座模型（4bit 量化）=====
    print("\n[3/6] 加载基座模型（4bit 量化）...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    print("基座模型加载完成")

    # ===== 4. 应用 LoRA 配置 =====
    print("\n[4/6] 应用 LoRA 配置...")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention
            "gate_proj", "up_proj", "down_proj",     # MLP
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ===== 5. 训练配置 =====
    print("\n[5/6] 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    sft_config = SFTConfig(
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    # ===== 6. 开始训练 =====
    print("\n[6/6] 开始训练...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        **sft_config.to_dict(),
    )

    # 打印显存
    if torch.cuda.is_available():
        print(f"\n训练前显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    train_result = trainer.train()

    if torch.cuda.is_available():
        print(f"训练后显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # ===== 保存 LoRA 权重 =====
    print("\n[保存] LoRA 权重...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("\n" + "=" * 60)
    print(f"✓ 微调完成！")
    print(f"  LoRA 权重: {args.output_dir}")
    print(f"  Train loss: {metrics.get('train_loss', 'N/A')}")
    print(f"  Train steps: {metrics.get('train_steps', 'N/A')}")
    print("=" * 60)
    print(f"\n下一步：运行 scripts/evaluate_finetuned.py 评测")


if __name__ == "__main__":
    main()
