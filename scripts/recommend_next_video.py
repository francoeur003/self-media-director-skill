#!/usr/bin/env python3
"""Rank next-video candidates against creator history and emit one shootable brief."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

from common import clamp, compact_number, parse_number, read_history, write_json, write_text


CANDIDATE_ALIASES = {
    "主题": "topic",
    "选题": "topic",
    "角度": "angle",
    "切入角度": "angle",
    "形式": "format",
    "首句": "hook",
    "钩子": "hook",
    "结果": "payoff",
    "价值兑现": "payoff",
    "趋势分": "trend_score",
    "战略匹配": "strategic_fit",
    "制作成本": "effort",
    "来源": "source",
}


def read_candidates(path: str) -> List[Dict[str, Any]]:
    candidates = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, raw in enumerate(reader, start=2):
            row = {CANDIDATE_ALIASES.get(key.strip(), key.strip()): value.strip() for key, value in raw.items()}
            if not row.get("topic") or not row.get("angle"):
                continue
            for field, default in (("trend_score", 50), ("strategic_fit", 60), ("effort", 3)):
                row[field] = parse_number(row.get(field)) if row.get(field) else float(default)
            row["_row"] = row_number
            candidates.append(row)
    if not candidates:
        raise ValueError("候选表没有有效的 topic + angle 行")
    return candidates


def derive_candidates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[float]] = {}
    for row in rows:
        topic = str(row.get("topic") or "").strip()
        if topic:
            buckets.setdefault(topic, []).append(float(row["views"]))
    if not buckets:
        raise ValueError("历史表缺少 topic/主题；请补主题标签或提供候选表")
    ranked = sorted(buckets, key=lambda topic: median(buckets[topic]), reverse=True)[:3]
    angles = [
        ("最容易犯的 3 个错误，以及最快修正方法", "误区拆解"),
        ("我实测了常见做法，最后只保留这一种", "实测对比"),
        ("从原始状态到结果的完整过程", "前后对比"),
    ]
    return [
        {
            "topic": topic,
            "angle": angles[index % len(angles)][0],
            "format": angles[index % len(angles)][1],
            "trend_score": 50.0,
            "strategic_fit": 70.0,
            "effort": 2.0,
            "source": "历史高表现主题自动派生",
        }
        for index, topic in enumerate(ranked)
    ]


def interaction_rate(row: Dict[str, Any]) -> Optional[float]:
    views = float(row["views"])
    if views <= 0:
        return None
    interactions = (
        float(row.get("likes") or 0)
        + 2 * float(row.get("comments") or 0)
        + 3 * float(row.get("shares") or 0)
        + 2 * float(row.get("saves") or 0)
    )
    return interactions / views if interactions else None


def goal_weights(goal: str) -> Dict[str, float]:
    if goal == "conversion":
        return {"proven": 0.25, "engagement": 0.22, "trend": 0.12, "novelty": 0.10, "fit": 0.26, "effort": 0.05}
    if goal == "authority":
        return {"proven": 0.24, "engagement": 0.16, "trend": 0.12, "novelty": 0.14, "fit": 0.29, "effort": 0.05}
    return {"proven": 0.34, "engagement": 0.16, "trend": 0.22, "novelty": 0.13, "fit": 0.10, "effort": 0.05}


def score_candidates(
    rows: List[Dict[str, Any]], candidates: List[Dict[str, Any]], goal: str
) -> List[Dict[str, Any]]:
    overall_log = median(math.log1p(float(row["views"])) for row in rows)
    overall_engagement_values = [value for value in (interaction_rate(row) for row in rows) if value is not None]
    overall_engagement = median(overall_engagement_values) if overall_engagement_values else None
    recent_topics = [str(row.get("topic") or "").strip() for row in rows[-5:]]
    weights = goal_weights(goal)
    results = []
    for candidate in candidates:
        topic = str(candidate["topic"]).strip()
        topic_rows = [row for row in rows if str(row.get("topic") or "").strip() == topic]
        if topic_rows:
            topic_log = median(math.log1p(float(row["views"])) for row in topic_rows)
            proven = clamp(50 + 32 * (topic_log - overall_log), 10, 95)
        else:
            proven = 52.0
        topic_engagement_values = [
            value for value in (interaction_rate(row) for row in topic_rows) if value is not None
        ]
        if topic_engagement_values and overall_engagement and overall_engagement > 0:
            ratio = median(topic_engagement_values) / overall_engagement
            engagement = clamp(50 + 35 * math.log(max(0.1, ratio)), 10, 95)
        else:
            engagement = 50.0
        if topic not in recent_topics:
            novelty = 78.0 if topic_rows else 66.0
        else:
            last_distance = len(recent_topics) - 1 - max(
                index for index, value in enumerate(recent_topics) if value == topic
            )
            novelty = 25.0 + min(3, last_distance) * 12.0
        trend = clamp(float(candidate.get("trend_score") or 50), 0, 100)
        fit = clamp(float(candidate.get("strategic_fit") or 60), 0, 100)
        effort_raw = clamp(float(candidate.get("effort") or 3), 1, 5)
        effort_score = 100 - (effort_raw - 1) * 20
        components = {
            "proven_performance": round(proven, 1),
            "engagement_quality": round(engagement, 1),
            "trend_strength": round(trend, 1),
            "novelty_without_fatigue": round(novelty, 1),
            "strategic_fit": round(fit, 1),
            "production_efficiency": round(effort_score, 1),
        }
        total = (
            proven * weights["proven"]
            + engagement * weights["engagement"]
            + trend * weights["trend"]
            + novelty * weights["novelty"]
            + fit * weights["fit"]
            + effort_score * weights["effort"]
        )
        result = dict(candidate)
        result["score"] = round(total, 1)
        result["components"] = components
        result["historical_topic_posts"] = len(topic_rows)
        results.append(result)
    return sorted(results, key=lambda item: item["score"], reverse=True)


def load_forecast(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "next_post" not in payload:
        raise ValueError("forecast JSON 缺少 next_post")
    return payload


def default_hook(topic: str, angle: str) -> str:
    return f"如果你正在做{topic}，先别急着开始——{angle}。"


def build_payload(
    ranked: List[Dict[str, Any]],
    forecast: Optional[Dict[str, Any]],
    goal: str,
) -> Dict[str, Any]:
    winner = ranked[0]
    hook = winner.get("hook") or default_hook(winner["topic"], winner["angle"])
    payoff = winner.get("payoff") or f"现场展示“{winner['topic']}”从错误做法到正确结果的对比"
    predicted = None
    if forecast:
        base = forecast["next_post"]
        factor = clamp(0.65 + winner["score"] / 160.0, 0.75, 1.30)
        predicted = {
            "p20": round(float(base["p20"]) * factor),
            "p50": round(float(base["p50"]) * factor),
            "p80": round(float(base["p80"]) * factor),
            "confidence": forecast.get("confidence", "unknown"),
            "note": "以账号基线区间乘候选相对分修正，不是结果承诺",
        }
    return {
        "goal": goal,
        "recommendation": {
            "topic": winner["topic"],
            "angle": winner["angle"],
            "format": winner.get("format") or "结果先行的口播 + 演示",
            "score": winner["score"],
            "why_now": [
                "兼顾历史表现、互动质量、近期疲劳和制作成本",
                f"该主题历史样本 {winner['historical_topic_posts']} 条",
                f"证据来源：{winner.get('source') or '账号历史数据'}",
            ],
            "predicted_traffic": predicted,
            "shooting_brief": {
                "hook_0_3s": hook,
                "beat_3_8s": "点出观众正在犯的具体错误，只讲一个核心冲突。",
                "beat_8_25s": f"用屏幕、实物或过程演示：{payoff}。",
                "beat_25_35s": "给出可复用的三步方法，并再次展示结果。",
                "cta": "让观众评论自己的具体问题；发布后优先回复高意向问题，为下一条选题采样。",
                "must_capture": payoff,
                "success_metric": "发布 24 小时后的播放、3 秒留存、完播、分享/收藏率与评论问题数",
            },
            "score_components": winner["components"],
        },
        "ranked_candidates": [
            {
                "topic": item["topic"],
                "angle": item["angle"],
                "score": item["score"],
            }
            for item in ranked
        ],
    }


def markdown_report(payload: Dict[str, Any]) -> str:
    item = payload["recommendation"]
    brief = item["shooting_brief"]
    predicted = item["predicted_traffic"]
    if predicted:
        traffic_line = (
            f"{compact_number(predicted['p20'])}–{compact_number(predicted['p80'])}，"
            f"P50 {compact_number(predicted['p50'])}，置信度 {predicted['confidence']}"
        )
    else:
        traffic_line = "未提供账号预测基线，本次不编造播放量"
    return f"""# 下一条视频拍摄 Brief

## 唯一主推荐

**{item['topic']}｜{item['angle']}**

- 推荐分：{item['score']}/100
- 形式：{item['format']}
- 预测流量：{traffic_line}

## 为什么现在拍

- {item['why_now'][0]}
- {item['why_now'][1]}
- {item['why_now'][2]}

## 直接开拍

- 0–3 秒：{brief['hook_0_3s']}
- 3–8 秒：{brief['beat_3_8s']}
- 8–25 秒：{brief['beat_8_25s']}
- 25–35 秒：{brief['beat_25_35s']}
- 必拍画面：{brief['must_capture']}
- CTA：{brief['cta']}

## 发布后只看这些

{brief['success_metric']}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_csv")
    parser.add_argument("--candidates")
    parser.add_argument("--forecast")
    parser.add_argument("--goal", choices=("growth", "conversion", "authority"), default="growth")
    parser.add_argument("--newest-first", action="store_true")
    parser.add_argument("--out-json")
    parser.add_argument("--out-md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = read_history(args.history_csv, newest_first=args.newest_first)
        candidates = read_candidates(args.candidates) if args.candidates else derive_candidates(rows)
        ranked = score_candidates(rows, candidates, args.goal)
        payload = build_payload(ranked, load_forecast(args.forecast), args.goal)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"选题失败：{exc}")
    write_json(args.out_json, payload)
    write_text(args.out_md, markdown_report(payload))
    if args.out_json:
        print(f"recommendation_json={Path(args.out_json).resolve()}")
    if args.out_md:
        print(f"recommendation_md={Path(args.out_md).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
