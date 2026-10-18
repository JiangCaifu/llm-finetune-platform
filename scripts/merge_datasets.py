"""
合并多源数据集
================
将本地手写数据 + API 生成数据 合并为统一的数据集，去重后输出。

使用方法：
  python scripts/merge_datasets.py
"""
import json
from pathlib import Path
from collections import Counter
import hashlib


def main():
    sources = [
        ("local", Path("data/tourism_raw_local.json")),
        ("api", Path("data/tourism_raw_api.json")),
    ]

    all_data = []
    for src_name, path in sources:
        if not path.exists():
            print(f"[跳过] {src_name}: {path} 不存在")
            continue
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            item["source"] = src_name
            all_data.append(item)
        print(f"[加载] {src_name}: {len(items)} 条 ← {path}")

    print(f"\n合并前总数: {len(all_data)} 条")

    # 去重（基于 instruction 哈希）
    seen = set()
    deduped = []
    dup_count = 0
    for item in all_data:
        inst = str(item.get("instruction", "")).strip()
        h = hashlib.md5(inst.encode("utf-8")).hexdigest()
        if h in seen:
            dup_count += 1
            continue
        seen.add(h)
        deduped.append(item)

    print(f"去重: 移除 {dup_count} 条重复")
    print(f"合并后总数: {len(deduped)} 条")

    # 输出
    out_path = Path("data/tourism_raw.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 备份原文件
    if out_path.exists():
        backup = Path("data/tourism_raw.bak.json")
        if not backup.exists():
            import shutil
            shutil.copy2(out_path, backup)
            print(f"[备份] 原数据已备份到 {backup}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)
    print(f"[输出] {out_path}")

    # 统计
    print(f"\n=== 来源分布 ===")
    src_dist = Counter(d["source"] for d in deduped)
    for src, cnt in sorted(src_dist.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt} 条 ({cnt/len(deduped)*100:.1f}%)")

    print(f"\n=== 场景分布 ===")
    scene_dist = Counter(d.get("scenario", "通用") for d in deduped)
    for scene, cnt in sorted(scene_dist.items(), key=lambda x: -x[1]):
        print(f"  {scene}: {cnt} 条 ({cnt/len(deduped)*100:.1f}%)")

    print(f"\n=== 景点分布 ===")
    spot_dist = Counter(d.get("spot", "通用") for d in deduped)
    for spot, cnt in sorted(spot_dist.items(), key=lambda x: -x[1]):
        print(f"  {spot}: {cnt} 条")

    lens = [len(str(d.get("output", ""))) for d in deduped]
    print(f"\n=== 输出长度 ===")
    print(f"  最短: {min(lens)} 字")
    print(f"  最长: {max(lens)} 字")
    print(f"  平均: {sum(lens)//len(lens)} 字")
    print(f"  总字数: {sum(lens)} 字")


if __name__ == "__main__":
    main()
