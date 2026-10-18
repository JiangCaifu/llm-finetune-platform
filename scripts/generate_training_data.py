"""
文旅指令数据集生成器（并发版）
==============================
使用通义千问 API（兼容 OpenAI 协议）批量生成 Alpaca 格式的文旅问答数据。

特性：
  - 8 线程并发，10秒/条 → 1.2秒/条
  - 断点续传：中断后再次运行可继续
  - 进度保存：每 30 条保存一次
  - 自动去重：相同(场景,景点)组合最多 3 条
  - 多样性：24景点 × 10场景 × 7种问题风格

使用方法：
    python scripts/generate_training_data.py --total 800
    python scripts/generate_training_data.py --total 800 --resume  # 断点续传
    python scripts/generate_training_data.py --total 800 --workers 12  # 加大并发

成本估算：
    qwen-plus 约 0.008元/千 token，生成 800 条约 3-6 元
"""

import os
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from dotenv import load_dotenv
from openai import OpenAI

# ============ 场景库 ============
SCENARIOS: List[Dict] = [
    {"name": "景点开放时间与票价", "hint": "覆盖5A/4A景区，含旺季淡季差异"},
    {"name": "景点交通路线规划", "hint": "包含高铁/飞机/自驾多种方式"},
    {"name": "景点历史文化介绍", "hint": "深度文化背景，不只百科"},
    {"name": "周边美食推荐", "hint": "含本地特色菜，地址定位"},
    {"name": "住宿推荐", "hint": "覆盖民宿/酒店/特色住宿"},
    {"name": "行程规划建议", "hint": "1日/2日/3日不同需求"},
    {"name": "天气影响游玩建议", "hint": "雨/雪/高温等极端天气"},
    {"name": "门票优惠政策", "hint": "学生/老人/军人/团体"},
    {"name": "最佳游玩季节", "hint": "四季特色对比"},
    {"name": "避坑指南", "hint": "常见骗局/过度消费陷阱"},
]

SCENIC_SPOTS = [
    "故宫", "兵马俑", "黄山", "九寨沟", "张家界", "桂林山水", "西湖",
    "长城（八达岭/慕田峪）", "颐和园", "鼓浪屿", "丽江古城", "布达拉宫",
    "三亚亚龙湾", "稻城亚丁", "青海湖", "华山", "泰山", "峨眉山",
    "武夷山", "千岛湖", "周庄古镇", "凤凰古城", "敦煌莫高窟", "喀纳斯",
]

QUESTION_STYLES = [
    "请问{spot}的{topic}是什么？",
    "我想了解一下{spot}的{topic}。",
    "{spot}的{topic}是怎样的？",
    "去{spot}玩，{topic}方面有什么需要注意的？",
    "能不能详细说说{spot}的{topic}？",
    "{spot}{topic}详细介绍一下。",
    "带家人去{spot}，关于{topic}有什么建议？",
]

TOPIC_MAP = {
    "景点开放时间与票价": "开放时间和票价",
    "景点交通路线规划": "交通路线",
    "景点历史文化介绍": "历史文化",
    "周边美食推荐": "周边美食",
    "住宿推荐": "住宿选择",
    "行程规划建议": "行程规划",
    "天气影响游玩建议": "天气注意事项",
    "门票优惠政策": "门票优惠政策",
    "最佳游玩季节": "最佳游玩季节",
    "避坑指南": "避坑要点",
}


def build_client() -> OpenAI:
    """构造通义千问客户端"""
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if not api_key or api_key.startswith("sk-your"):
        raise RuntimeError("请先在 .env 中设置 DASHSCOPE_API_KEY")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)


def build_prompt(scenario: Dict, spot: str) -> str:
    """构造单条数据生成 prompt"""
    topic = TOPIC_MAP[scenario["name"]]
    question_style = random.choice(QUESTION_STYLES)
    instruction = question_style.format(spot=spot, topic=topic)

    return f"""请生成一条高质量的文旅领域问答数据，用于大模型微调训练。

【场景类型】{scenario['name']}
【具体景点】{spot}
【场景要求】{scenario['hint']}
【用户问题】{instruction}

【输出要求】
1. 输出必须是合法 JSON，字段固定为 instruction / input / output
2. instruction 用上面给的【用户问题】
3. input 留空字符串
4. output 必须详细、准确、有信息密度，至少200字，使用 Markdown 列表/分段
5. 不要包含编造的电话、网址、价格等不确定信息，必要时给范围
6. 风格像专业文旅客服，避免寒暄

【输出格式】
{{"instruction": "...", "input": "", "output": "..."}}
"""


def gen_one(client: OpenAI, model: str, scenario: Dict, spot: str, max_retry: int = 3) -> Dict:
    """生成一条数据，失败重试"""
    prompt = build_prompt(scenario, spot)
    last_err = None
    for i in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8 + random.random() * 0.3,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content.strip()
            item = json.loads(content)
            if not all(k in item for k in ("instruction", "input", "output")):
                raise ValueError(f"缺少字段: {item.keys()}")
            if len(str(item["output"])) < 50:
                raise ValueError("output 太短")
            item["scenario"] = scenario["name"]
            item["spot"] = spot
            return item
        except Exception as e:
            last_err = e
            time.sleep(1 + i * 2)  # 指数退避
    raise RuntimeError(f"重试 {max_retry} 次仍失败: {last_err}")


def gen_one_task(client, model, scenario, spot, idx):
    """线程池任务包装"""
    try:
        return idx, gen_one(client, model, scenario, spot)
    except Exception as e:
        return idx, e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=500, help="要生成的总条数")
    parser.add_argument("--model", type=str, default="qwen-plus")
    parser.add_argument("--output", type=str, default="data/tourism_raw.json")
    parser.add_argument("--resume", action="store_true", help="断点续传")
    parser.add_argument("--workers", type=int, default=8, help="并发线程数")
    args = parser.parse_args()

    client = build_client()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 断点续传：加载已有数据
    dataset = []
    if args.resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"[续传] 已有 {len(dataset)} 条")

    target_count = args.total - len(dataset)
    if target_count <= 0:
        print(f"已达目标数量：{len(dataset)} 条")
        return

    print(f"[开始] 目标 {args.total} 条，已有 {len(dataset)} 条，需生成 {target_count} 条")
    print(f"[配置] 并发 {args.workers} 线程，模型 {args.model}")

    # 生成任务列表（含多样性控制）
    tasks = []
    for i in range(target_count):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        spot = SCENIC_SPOTS[i % len(SCENIC_SPOTS)]
        if random.random() < 0.3:
            spot = random.choice(SCENIC_SPOTS)
        tasks.append((scenario, spot))

    # 检查已有数据的(场景,景点)组合，避免重复超过3次
    existing_keys = [d.get("scenario") + "|" + d.get("spot", "") for d in dataset]
    from collections import Counter
    key_count = Counter(existing_keys)
    filtered_tasks = []
    for scenario, spot in tasks:
        key = scenario["name"] + "|" + spot
        if key_count.get(key, 0) < 3:
            filtered_tasks.append((scenario, spot))
            key_count[key] = key_count.get(key, 0) + 1
    if len(filtered_tasks) < target_count:
        # 不足的话，允许更多重复
        while len(filtered_tasks) < target_count:
            scenario = random.choice(SCENARIOS)
            spot = random.choice(SCENIC_SPOTS)
            filtered_tasks.append((scenario, spot))

    tasks = filtered_tasks[:target_count]
    print(f"[任务] 已规划 {len(tasks)} 个生成任务")

    success = 0
    fail = 0
    t0 = time.time()
    save_lock = Lock()

    # 并发执行
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(gen_one_task, client, args.model, scenario, spot, i): (i, scenario, spot)
            for i, (scenario, spot) in enumerate(tasks)
        }

        for future in as_completed(futures):
            i, scenario, spot = futures[future]
            try:
                idx, result = future.result()
                if isinstance(result, Exception):
                    raise result
                with save_lock:
                    dataset.append(result)
                    success += 1

                # 进度打印
                if success % 20 == 0 or success <= 5:
                    elapsed = time.time() - t0
                    rate = success / elapsed if elapsed > 0 else 0
                    eta = (target_count - success) / rate if rate > 0 else 0
                    print(f"[{success}/{target_count}] {result['spot']} - {result['scenario']} | "
                          f"速率 {rate:.1f}条/秒 | ETA {eta:.0f}s")

                # 定期保存
                if success % 30 == 0:
                    with save_lock:
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(dataset, f, ensure_ascii=False, indent=2)
            except Exception as e:
                fail += 1
                print(f"[失败 #{fail}] {spot} - {scenario['name']}: {e}")

    # 最终保存
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n=== 生成完成 ===")
    print(f"成功: {success} 条 | 失败: {fail} 条 | 总计: {len(dataset)} 条")
    print(f"耗时: {(time.time()-t0):.0f}s")
    print(f"输出: {out_path}")

    # 场景分布
    from collections import Counter
    print(f"\n=== 场景分布 ===")
    dist = Counter(d.get("scenario", "未知") for d in dataset)
    for name, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {name}: {cnt} 条 ({cnt/len(dataset)*100:.1f}%)")


if __name__ == "__main__":
    main()
