"""
梯度诊断脚本（4 配置对比版）：找出 grad_norm=0 的真正根因
==========================================================

在 DSW 上运行：
  python scripts/diagnose_gradient.py --model $MODEL_PATH

4 个配置对比（每个独立加载模型，避免状态污染）：
  A. bf16 + use_reentrant=False + enable_input_require_grads   ← 标准修复
  B. bf16 + use_reentrant=True  + enable_input_require_grads   ← 隔离 use_reentrant 影响
  C. bf16 + use_reentrant=True  + 不调 enable_input_require_grads ← 复现 bug
  D. 4-bit + prepare_model_for_kbit_training                    ← 量化路径

判读（对照结论表）：
  - A 有梯度、C 无梯度 → 根因确认：reentrant 模式 + 缺 enable_input_require_grads
  - A 有梯度、B 无梯度 → use_reentrant 是关键开关（不是 enable_input_require_grads）
  - D 有梯度 → 4-bit 路径在当前 PEFT 版本下已被 prepare_model_for_kbit_training 自动修复
  - D 无梯度 → 4-bit 路径仍有问题
  - A 无梯度 → 根因不在 gradient checkpointing，在更深处（LoRA/数据/tokenizer）

正常训练的 grad_norm 应在 0.1~10 之间（参考 axolotl 文档）。
我们的失败训练 grad_norm=0.0 = 铁定异常。
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

# 与训练脚本完全一致的 LoRA 配置
LORA_CONFIG = LoraConfig(
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


def run_forward_backward(model, tokenizer, config_name: str) -> float:
    """对已加载+LoRA 包装的模型跑一次 forward+backward，返回总 grad_norm"""
    print(f"\n{'=' * 60}")
    print(f"测试配置: {config_name}")
    print(f"{'=' * 60}")

    model.train()  # 训练模式

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

    # 统计 LoRA 梯度
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
                if len(sample_grads) < 3:
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

    del model, outputs
    torch.cuda.empty_cache()
    return total_grad_norm


def load_bf16(model_path: str, use_reentrant: bool, enable_input_grads: bool):
    """加载 bf16 全精度模型，按参数配置 gradient checkpointing"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
    )
    if enable_input_grads:
        model.enable_input_require_grads()
    return get_peft_model(model, LORA_CONFIG)


def load_4bit(model_path: str):
    """加载 4-bit nf4 量化模型，走 prepare_model_for_kbit_training 路径"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    return get_peft_model(model, LORA_CONFIG)


def main():
    parser = argparse.ArgumentParser(description="梯度诊断（4 配置对比）：找出 grad_norm=0 根因")
    parser.add_argument("--model", type=str, required=True, help="基座模型路径")
    args = parser.parse_args()

    print(f"模型路径: {args.model}")
    print(f"样本: {SAMPLE_TEXT[:80]}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    results = {}

    # ===== 配置 A: bf16 + use_reentrant=False + enable_input_require_grads（标准修复） =====
    print("\n" + "#" * 60)
    print("# 配置 A: bf16 + use_reentrant=False + enable_input_require_grads")
    print("#" * 60)
    model_a = load_bf16(args.model, use_reentrant=False, enable_input_grads=True)
    results["A"] = run_forward_backward(model_a, tokenizer, "A: bf16 + reentrant=False + input_grads")

    # ===== 配置 B: bf16 + use_reentrant=True + enable_input_require_grads =====
    print("\n" + "#" * 60)
    print("# 配置 B: bf16 + use_reentrant=True + enable_input_require_grads")
    print("#" * 60)
    model_b = load_bf16(args.model, use_reentrant=True, enable_input_grads=True)
    results["B"] = run_forward_backward(model_b, tokenizer, "B: bf16 + reentrant=True + input_grads")

    # ===== 配置 C: bf16 + use_reentrant=True + 不调 enable_input_require_grads（复现 bug） =====
    print("\n" + "#" * 60)
    print("# 配置 C: bf16 + use_reentrant=True + 不调 enable_input_require_grads")
    print("#" * 60)
    model_c = load_bf16(args.model, use_reentrant=True, enable_input_grads=False)
    results["C"] = run_forward_backward(model_c, tokenizer, "C: bf16 + reentrant=True + 无 input_grads")

    # ===== 配置 D: 4-bit + prepare_model_for_kbit_training =====
    print("\n" + "#" * 60)
    print("# 配置 D: 4-bit nf4 + prepare_model_for_kbit_training")
    print("#" * 60)
    model_d = load_4bit(args.model)
    results["D"] = run_forward_backward(model_d, tokenizer, "D: 4-bit + prepare_model_for_kbit_training")

    # ===== 结论 =====
    print("\n" + "=" * 60)
    print("诊断结论")
    print("=" * 60)
    print(f"{'配置':<50} {'grad_norm':<12} {'状态'}")
    print("-" * 80)
    for key, desc in [
        ("A", "bf16 + reentrant=False + input_grads（标准修复）"),
        ("B", "bf16 + reentrant=True  + input_grads"),
        ("C", "bf16 + reentrant=True  + 无 input_grads（复现 bug）"),
        ("D", "4-bit + prepare_model_for_kbit_training"),
    ]:
        gn = results[key]
        status = "✓ 有梯度" if gn > 1e-8 else "✗ 无梯度"
        print(f"{desc:<50} {gn:<12.6f} {status}")

    print()
    # 根因判定
    has = lambda k: results[k] > 1e-8
    if has("A") and not has("C"):
        print("→ 根因确认：reentrant 模式 + 缺 enable_input_require_grads 导致反向图未构建")
        print("  修复：训练脚本加 model.enable_input_require_grads() + use_reentrant=False")
    elif has("A") and has("C"):
        print("→ A 和 C 都有梯度：根因不在 gradient checkpointing 配置")
        print("  需要排查：SFTTrainer / 数据格式 / dataset_text_field")
    elif not has("A"):
        print("→ 标准修复配置 A 也无梯度：根因在更深处（LoRA / tokenizer / 模型本身）")
        print("  需要进一步排查")
    else:
        print("→ 其他组合，请把上表贴出来人工分析")

    # 4-bit 路径单独说明
    if has("D"):
        print("→ 4-bit 路径在当前 PEFT 版本下已被 prepare_model_for_kbit_training 自动修复，可用")
    else:
        print("→ 4-bit 路径仍有问题：prepare_model_for_kbit_training 未正确启用 input_require_grads")
        print("  若选 4-bit 训练，需手动加 model.enable_input_require_grads()")
    print("=" * 60)


if __name__ == "__main__":
    main()
