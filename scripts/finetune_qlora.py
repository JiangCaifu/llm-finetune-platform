"""
QLoRA 微调脚本（在阿里云 DSW / AutoDL GPU 实例上运行）
======================================================

功能：
  1. 加载 Qwen-7B-Chat 基座（4bit 量化）
  2. 应用 LoRA 配置（rank=8, alpha=32）
  3. 加载文旅数据集（Alpaca 格式）
  4. 使用 SFTTrainer 训练
  5. 保存 LoRA 权重

前置条件：
  1. GPU 实例：RTX 3090 (24GB) 或 A10 (24GB)
  2. 镜像：PyTorch 2.3 + CUDA 12.1 + Python 3.11
  3. 安装依赖：pip install -r requirements-gpu.txt
  4. 数据集已上传到 data/ 目录
  5. Qwen-7B-Chat 模型已下载到本地

运行：
  cd ~/llm-finetune-platform
  python scripts/finetune_qlora.py
  python scripts/finetune_qlora.py --epochs 5  # 调整 epoch
  python scripts/finetune_qlora.py --rank 16   # 调整 LoRA rank

预计耗时（A10 GPU）：
  - 560条训练数据，3个epoch，约 30-50 分钟
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
            f"然后上传 data/ 目录到 GPU 实例"
        )

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    train_ds = Dataset.from_list(train_data).map(
        lambda x: format_to_chat(x, tokenizer),
        remove_columns=list(train_data[0].keys()) if train_data else ["instruction", "input", "output"],
    )
    val_ds = Dataset.from_list(val_data).map(
        lambda x: format_to_chat(x, tokenizer),
        remove_columns=list(val_data[0].keys()) if val_data else ["instruction", "input", "output"],
    )

    print(f"训练集: {len(train_ds)} 条")
    print(f"验证集: {len(val_ds)} 条")

    return train_ds, val_ds


def setup_pad_token(tokenizer):
    """
    为 Qwen-7B-Chat 旧版 tokenizer 设置 pad_token。

    背景：
      Qwen-7B-Chat（2023 年初版）的 tokenizer_config.json 仅 174 字节，
      未显式配置 pad_token/eos_token/unk_token 的字符串映射，
      导致 tokenizer.pad_token 属性返回 None，
      trl 0.22 的 SFTTrainer 初始化时调用
        pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
      会传 None 给 tokenization_qwen.py 的 convert_tokens_to_ids，
      触发 'NoneType' object is not iterable 错误。

    工程标准方案（按优先级）：
      1. 优先从 generation_config 读 pad_token_id（最规范，适配不同模型版本）
      2. 配置缺失时用已知常量兜底（Qwen-7B-Chat 的 たくさん token id=151643）
      3. 同步设置 pad_token 字符串，避免字符串查询返回 None
      4. 不新增 token，不 resize embedding，避免与 LoRA 训练冲突
    """
    if tokenizer.pad_token is not None:
        return

    # ===== 步骤 1：优先尝试直接用 eos_token（HuggingFace 官方推荐写法）=====
    try:
        eos = tokenizer.eos_token
        if eos is not None:
            tokenizer.pad_token = eos
            return
    except Exception:
        pass

    # ===== 步骤 2：从 generation_config 读 pad_token_id（最规范的工程做法）=====
    pad_token_id = None
    try:
        gen_config = getattr(tokenizer, "generation_config", None)
        if gen_config is not None:
            # 优先用 pad_token_id，其次 eos_token_id（很多模型共用同一个 id）
            pad_token_id = getattr(gen_config, "pad_token_id", None) or \
                           getattr(gen_config, "eos_token_id", None)
    except Exception:
        pass

    # ===== 步骤 3：兜底用已知常量（Qwen-7B-Chat 的 たくさん token id）=====
    # 已通过 vocab 探查验证：id=151643 对应 たくさん token
    if pad_token_id is None:
        pad_token_id = 151643  # Qwen-7B-Chat 的 たくさん token id

    # ===== 步骤 4：设置 pad_token_id + 同步 pad_token 字符串 =====
    tokenizer.pad_token_id = pad_token_id
    # 从 vocab 反查字符串，保证 pad_token 不为 None
    try:
        pad_token_str = tokenizer.convert_ids_to_tokens(pad_token_id)
        if pad_token_str is None:
            # 兜底：用 decode 拿到字符串
            pad_token_str = tokenizer.decode([pad_token_id])
        if pad_token_str is None:
            # 最后兜底：构造字符串（chr(0x3C) 是 '<'）
            pad_token_str = chr(0x3C) + "|endoftext|"
        tokenizer.pad_token = pad_token_str
    except Exception:
        tokenizer.pad_token = chr(0x3C) + "|endoftext|"

    print(f"  设置 pad_token_id={tokenizer.pad_token_id}, pad_token={tokenizer.pad_token!r}")


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
    # 显式设置 pad_token，避免 trl 0.22 内部传 None 报错
    setup_pad_token(tokenizer)
    print(f"Tokenizer 加载完成，vocab_size = {len(tokenizer)}")
    print(f"pad_token = {tokenizer.pad_token!r}, pad_token_id = {tokenizer.pad_token_id}")

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
    # Qwen-7B-Chat（2023 初版）使用 GPT 风格命名：
    #   c_attn = 合并的 QKV 矩阵
    #   c_proj = attention 输出投影 + MLP 输出投影
    #   w1, w2 = MLP 上下投影（对应新版的 gate_proj/up_proj）
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["c_attn", "c_proj", "w1", "w2"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ===== 5. 训练配置 =====
    print("\n[5/6] 配置训练参数...")
    # trl 0.22+ 的 SFTConfig 继承自 TrainingArguments，所有参数都直接在 SFTConfig 里设置
    # 旧版的 max_seq_length 改名为 max_length；不再需要单独的 TrainingArguments
    sft_config = SFTConfig(
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
        # SFT 专属参数（trl 0.22+ 新 API）
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
    )

    # ===== 6. 开始训练 =====
    print("\n[6/6] 开始训练...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
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
