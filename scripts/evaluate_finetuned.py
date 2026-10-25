"""
模型评测脚本（对比微调前后效果）
================================

功能：
  1. 加载原始 Qwen2-7B-Instruct
  2. 加载微调后的 Qwen2-7B-Instruct + LoRA
  3. 在 70 条测试集上对比推理结果
  4. 计算准确率（关键词命中）+ 长度 + 格式合规率
  5. 输出对比报告

运行（在阿里云 DSW / AutoDL GPU 实例上）：
  python scripts/evaluate_finetuned.py
  python scripts/evaluate_finetuned.py --samples 30  # 只评30条
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


SYSTEM_PROMPT = "你是文旅助手，提供准确、详细的旅游信息和建议。"


def load_model(model_path: str, lora_path: str = None, load_in_4bit: bool = True):
    """加载模型 + tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=load_in_4bit,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # 与训练保持一致
    )

    if lora_path and Path(lora_path).exists():
        model = PeftModel.from_pretrained(base, lora_path)
        print(f"[加载] LoRA 权重: {lora_path}")
    else:
        model = base
        print(f"[加载] 仅基座模型（无 LoRA）")

    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 300) -> str:
    """生成回答"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 只取新生成的部分
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer


def evaluate_quality(answer: str, instruction: str, expected_keywords: List[str] = None) -> Dict:
    """质量评估：长度、关键词命中、格式合规"""
    result = {
        "answer_length": len(answer),
        "has_list": "- " in answer or "1." in answer or "*" in answer,
        "has_section": "##" in answer or "##" in answer,
        "keyword_hits": 0,
        "keyword_total": 0,
    }

    if expected_keywords:
        result["keyword_total"] = len(expected_keywords)
        for kw in expected_keywords:
            if kw in answer:
                result["keyword_hits"] += 1

    # 格式合规：含列表/分节 + 长度 > 100
    result["format_ok"] = (result["has_list"] or result["has_section"]) and result["answer_length"] > 100
    return result


def extract_keywords(instruction: str, output: str) -> List[str]:
    """从测试集的预期 output 中提取关键名词"""
    # 简化：提取 output 中的关键词
    keywords = []
    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue
        # 找带 - 或 * 开头的列表项中的关键词
        if line.startswith(("- ", "* ", "1. ", "2. ")):
            kw = line.lstrip("- *1234567890. ").strip()
            # 取前8个字作为关键词
            if len(kw) > 4:
                keywords.append(kw[:8])
    return keywords[:5]  # 最多5个


def main():
    parser = argparse.ArgumentParser(description="评测微调前后模型效果")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--lora_path", type=str, default="outputs/qwen-tourism-lora")
    parser.add_argument("--test_data", type=str, default="data/tourism_test.json")
    parser.add_argument("--output", type=str, default="data/eval_results.json")
    parser.add_argument("--samples", type=int, default=50, help="评测样本数")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    args = parser.parse_args()

    print("=" * 60)
    print("模型评测：微调前 vs 微调后")
    print("=" * 60)

    # 加载测试集
    with open(args.test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    if args.samples < len(test_data):
        test_data = test_data[:args.samples]
    print(f"测试样本数: {len(test_data)}")

    # 1. 评测原始模型
    print(f"\n[1/2] 加载原始模型: {args.model}")
    base_model, tokenizer = load_model(args.model, lora_path=None)

    base_results = []
    for i, item in enumerate(test_data):
        question = item["instruction"]
        expected_output = item.get("output", "")
        keywords = extract_keywords(question, expected_output)

        print(f"  [{i+1}/{len(test_data)}] {question[:40]}...", end="", flush=True)
        answer = generate_answer(base_model, tokenizer, question, args.max_new_tokens)
        quality = evaluate_quality(answer, question, keywords)
        base_results.append({
            "question": question,
            "expected": expected_output[:200],
            "answer": answer,
            "quality": quality,
        })
        print(f" len={quality['answer_length']} kw={quality['keyword_hits']}/{quality['keyword_total']}")

    # 释放显存
    del base_model
    torch.cuda.empty_cache()

    # 2. 评测微调模型
    if not Path(args.lora_path).exists():
        print(f"\n⚠️  LoRA 权重不存在: {args.lora_path}")
        print(f"跳过微调模型评测")
        report = {
            "base_model_results": base_results,
            "finetuned_model_results": [],
            "summary": {"base_only": True},
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return

    print(f"\n[2/2] 加载微调模型: {args.model} + {args.lora_path}")
    ft_model, tokenizer = load_model(args.model, lora_path=args.lora_path)

    ft_results = []
    for i, item in enumerate(test_data):
        question = item["instruction"]
        expected_output = item.get("output", "")
        keywords = extract_keywords(question, expected_output)

        print(f"  [{i+1}/{len(test_data)}] {question[:40]}...", end="", flush=True)
        answer = generate_answer(ft_model, tokenizer, question, args.max_new_tokens)
        quality = evaluate_quality(answer, question, keywords)
        ft_results.append({
            "question": question,
            "expected": expected_output[:200],
            "answer": answer,
            "quality": quality,
        })
        print(f" len={quality['answer_length']} kw={quality['keyword_hits']}/{quality['keyword_total']}")

    # ===== 对比报告 =====
    print("\n" + "=" * 60)
    print("评测报告")
    print("=" * 60)

    def avg(lst, key):
        vals = [r["quality"][key] for r in lst]
        return sum(vals) / len(vals) if vals else 0

    base_kw_rate = avg(base_results, "keyword_hits") / max(avg(base_results, "keyword_total"), 1)
    ft_kw_rate = avg(ft_results, "keyword_hits") / max(avg(ft_results, "keyword_total"), 1)
    base_format_rate = avg(base_results, "format_ok")
    ft_format_rate = avg(ft_results, "format_ok")
    base_avg_len = avg(base_results, "answer_length")
    ft_avg_len = avg(ft_results, "answer_length")

    print(f"\n{'指标':<20} {'原始模型':<15} {'微调模型':<15} {'变化':<15}")
    print("-" * 65)
    print(f"{'平均长度':<20} {base_avg_len:<15.0f} {ft_avg_len:<15.0f} {ft_avg_len-base_avg_len:+.0f}")
    print(f"{'关键词命中率':<20} {base_kw_rate*100:<15.1f}% {ft_kw_rate*100:<15.1f}% {(ft_kw_rate-base_kw_rate)*100:+.1f}%")
    print(f"{'格式合规率':<20} {base_format_rate*100:<15.1f}% {ft_format_rate*100:<15.1f}% {(ft_format_rate-base_format_rate)*100:+.1f}%")

    # 保存
    report = {
        "base_model_results": base_results,
        "finetuned_model_results": ft_results,
        "summary": {
            "samples": len(test_data),
            "base": {
                "avg_length": base_avg_len,
                "keyword_hit_rate": base_kw_rate,
                "format_compliance_rate": base_format_rate,
            },
            "finetuned": {
                "avg_length": ft_avg_len,
                "keyword_hit_rate": ft_kw_rate,
                "format_compliance_rate": ft_format_rate,
            },
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {args.output}")


if __name__ == "__main__":
    main()
