"""
数据清洗 + 格式转换
=====================
将 data/tourism_raw.json 处理为：
  1. data/tourism_clean.json   - Alpaca 格式（清洗去重后）
  2. data/tourism_dataset/     - HuggingFace Dataset 格式（Qwen 对话格式）

清洗规则：
  - 去除 instruction/output 为空或过短的样本
  - 去除 output 过长的样本（>2000字）
  - 去重（基于 instruction 哈希）
  - 统计输出长度分布

使用方法：
  python scripts/clean_data.py
"""

import json
import hashlib
from pathlib import Path
from collections import Counter
from datasets import Dataset


def load_raw(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_data(raw: list) -> list:
    """清洗 + 去重"""
    seen_hashes = set()
    cleaned = []
    stats = {"too_short": 0, "too_long": 0, "duplicate": 0, "missing_field": 0}

    for item in raw:
        instruction = str(item.get("instruction", "")).strip()
        output = str(item.get("output", "")).strip()

        # 字段完整性
        if not instruction or not output:
            stats["missing_field"] += 1
            continue
        # 长度过滤
        if len(instruction) < 5 or len(output) < 30:
            stats["too_short"] += 1
            continue
        if len(output) > 2000:
            stats["too_long"] += 1
            continue
        # 去重（基于 instruction 哈希，宽松判定）
        h = hashlib.md5(instruction.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            stats["duplicate"] += 1
            continue
        seen_hashes.add(h)

        cleaned.append({
            "instruction": instruction,
            "input": "",
            "output": output,
            "source": item.get("source", "unknown"),
            "spot": item.get("spot", "通用"),
            "topic": item.get("topic", "general"),
        })

    return cleaned, stats


def to_qwen_chat(example: dict) -> dict:
    """转为 Qwen ChatML 对话格式"""
    system_prompt = "你是文旅助手，提供准确、详细的旅游信息和建议。"
    text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    )
    return {"text": text, **example}


def main():
    raw_path = Path("data/tourism_raw.json")
    clean_path = Path("data/tourism_clean.json")
    hf_dataset_path = Path("data/tourism_dataset")

    if not raw_path.exists():
        raise FileNotFoundError(f"原始数据不存在: {raw_path}")

    # 1. 加载
    raw = load_raw(raw_path)
    print(f"[1/4] 加载原始数据: {len(raw)} 条")

    # 2. 清洗
    cleaned, stats = clean_data(raw)
    print(f"[2/4] 清洗后: {len(cleaned)} 条")
    print(f"     过滤统计: {stats}")

    # 3. 保存 Alpaca 格式
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    with open(clean_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"[3/4] 保存 Alpaca 格式: {clean_path.absolute()}")

    # 4. 转 HuggingFace Dataset（Qwen ChatML 格式）
    dataset = Dataset.from_list(cleaned)
    dataset = dataset.map(to_qwen_chat)
    dataset.save_to_disk(str(hf_dataset_path))
    print(f"[4/4] 保存 HF Dataset: {hf_dataset_path.absolute()}")

    # === 详细统计 ===
    print(f"\n=== 数据集统计 ===")
    print(f"总数: {len(cleaned)} 条")

    print(f"\n=== 来源分布 ===")
    src_dist = Counter(d["source"] for d in cleaned)
    for src, cnt in sorted(src_dist.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt} 条 ({cnt/len(cleaned)*100:.1f}%)")

    print(f"\n=== 主题分布 ===")
    topic_dist = Counter(d["topic"] for d in cleaned)
    for topic, cnt in sorted(topic_dist.items(), key=lambda x: -x[1]):
        print(f"  {topic}: {cnt} 条 ({cnt/len(cleaned)*100:.1f}%)")

    print(f"\n=== 景点分布 ===")
    spot_dist = Counter(d["spot"] for d in cleaned)
    for spot, cnt in sorted(spot_dist.items(), key=lambda x: -x[1]):
        print(f"  {spot}: {cnt} 条")

    lens = [len(d["output"]) for d in cleaned]
    print(f"\n=== 输出长度 ===")
    print(f"  最短: {min(lens)} 字")
    print(f"  最长: {max(lens)} 字")
    print(f"  平均: {sum(lens)//len(lens)} 字")
    print(f"  总字数: {sum(lens)} 字")

    # === 划分训练集/验证集/测试集 ===
    print(f"\n=== 数据集划分（8:1:1）===")
    import random
    random.seed(42)
    shuffled = cleaned.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(5, n // 10)
    n_val = max(5, n // 10)
    n_train = n - n_test - n_val

    train_set = shuffled[:n_train]
    val_set = shuffled[n_train:n_train + n_val]
    test_set = shuffled[n_train + n_val:]

    splits = {"train": train_set, "validation": val_set, "test": test_set}
    for name, data in splits.items():
        out = Path(f"data/tourism_{name}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  {name}: {len(data)} 条 → {out}")

    print(f"\n=== 样例（验证集第1条）===")
    if val_set:
        sample = val_set[0]
        print(f"instruction: {sample['instruction']}")
        print(f"output (前200字): {sample['output'][:200]}...")


if __name__ == "__main__":
    main()
