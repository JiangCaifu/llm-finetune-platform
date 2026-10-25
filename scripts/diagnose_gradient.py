"""
梯度诊断脚本：找出 grad_norm=0 的真正根因
================================================

在 DSW 上运行：
  python scripts/diagnose_gradient.py --model $MODEL_PATH

作用：
  对同一个样本，分别用 bf16 全精度 和 4-bit 量化 跑一次 forward+backward，
  直接打印 LoRA 参数的实际梯度范数。2 分钟内确定性回答"4-bit 是不是根因"。

判读：
  - bf16 有梯度、4-bit 无梯度 → 确认 4-bit 是根因，用 bf16 全精度训练
  - 两者都无梯度 → 根因在更深处（LoRA / SFTTrainer / 数据 / tokenizer）
"""

import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


SAMPLE_TEXT = (
    "<|im_start|>system\n你是文旅助手，提供准确、详细的旅游信息和建议。<|im_end|>\n"
    "<|im_start|>user\n北京有什么景点？<|im_end|>\n"
    "<|im_start|>assistant\n北京有故宫、长城、颐和园等著名景点。<|im_end|>"
)


def test_config(model, tokenizer, config_name: str):
    """对给定模型跑一次 forward+backward，打印 LoRA 梯度"""
    print(f"\n{'=' * 60}")
    print(f"测试配置: {config_name}")
    print(f"{'=' * 60}")

    # 应用 LoRA（与训练脚本完全一致的配置）
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.train()  # 训练模式（关键！eval 模式下 dropout 等行为不同）

    # 准备输入
    inputs = tokenizer(
        SAMPLE_TEXT,
        return_tensors="pt",
        max_length=128,
        truncation=True,
    ).to("cuda")

    # Forward
    outputs = model(**inputs, labels=inputs["input_ids"])
    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Loss requires_grad: {outputs.loss.requires_grad}")

    # Backward
    outputs.loss.backward()

    # 检查每个 LoRA 参数的梯度
    total_grad_norm_sq = 0.0
    trainable_count = 0
    zero_or_none_count = 0
    sample_grads = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_count += 1
            if param.grad is not None:
                gn = param.grad.norm().item()
                total_grad_norm_sq += gn ** 2
                if gn == 0.0:
                    zero_or_none_count += 1
                if len(sample_grads) < 3:  # 只打印前 3 个，避免刷屏
                    sample_grads.append((name, gn))
            else:
                zero_or_none_count += 1
                if len(sample_grads) < 3:
                    sample_grads.append((name, "None"))

    total_grad_norm = total_grad_norm_sq ** 0.5
    print(f"可训练参数数: {trainable_count}")
    print(f"梯度为 None 或 0 的参数数: {zero_or_none_count}")
    print(f"总 grad_norm: {total_grad_norm:.6f}")
    print(f"前 3 个 LoRA 参数梯度示例:")
    for name, gn in sample_grads:
        print(f"  {name}: {gn}")

    if total_grad_norm > 1e-8:
        print(f"✓ {config_name}: 梯度正常流动，模型可以学习")
    else:
        print(f"✗ {config_name}: 梯度为 0，模型不会学习")

    # 清理显存
    del model, outputs
    torch.cuda.empty_cache()
    return total_grad_norm


def main():
    parser = argparse.ArgumentParser(description="梯度诊断：找出 grad_norm=0 的根因")
    parser.add_argument("--model", type=str, required=True, help="基座模型路径")
    args = parser.parse_args()

    print(f"模型路径: {args.model}")
    print(f"样本: {SAMPLE_TEXT[:80]}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ===== 配置 1: bf16 全精度 =====
    print("\n" + "#" * 60)
    print("# 配置 1: bf16 全精度 + gradient checkpointing")
    print("#" * 60)
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model_bf16.config.use_cache = False
    model_bf16.gradient_checkpointing_enable()
    model_bf16.enable_input_require_grads()
    gn_bf16 = test_config(model_bf16, tokenizer, "bf16 全精度")

    # ===== 配置 2: 4-bit 量化 =====
    print("\n" + "#" * 60)
    print("# 配置 2: 4-bit nf4 量化（与训练脚本失败的配置一致）")
    print("#" * 60)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model_4bit = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model_4bit = prepare_model_for_kbit_training(model_4bit)
    gn_4bit = test_config(model_4bit, tokenizer, "4-bit nf4 量化")

    # ===== 结论 =====
    print("\n" + "=" * 60)
    print("诊断结论")
    print("=" * 60)
    print(f"bf16 全精度 grad_norm: {gn_bf16:.6f}")
    print(f"4-bit 量化  grad_norm: {gn_4bit:.6f}")
    print()
    if gn_bf16 > 1e-8 and gn_4bit < 1e-8:
        print("→ 确认：4-bit 量化是根因。用 bf16 全精度训练即可。")
    elif gn_bf16 < 1e-8 and gn_4bit < 1e-8:
        print("→ 两者都无梯度：根因不在量化，在更深处。")
        print("  需要进一步排查：LoRA target_modules / SFTTrainer / 数据 / tokenizer")
    elif gn_bf16 > 1e-8 and gn_4bit > 1e-8:
        print("→ 两者都有梯度：训练失败可能是别的环节（数据/优化器/学习率）。")
    else:
        print("→ 异常情况，需要人工分析。")
    print("=" * 60)


if __name__ == "__main__":
    main()
