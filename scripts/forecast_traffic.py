#!/usr/bin/env python3
"""Robust short-horizon traffic forecast from a creator's post history."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from common import (
    clamp,
    compact_number,
    percentile,
    read_history,
    robust_sigma,
    weighted_mean,
    weighted_median,
    write_json,
    write_text,
)


def recency_weights(count: int) -> List[float]:
    half_life = max(4.0, min(12.0, count / 2.0))
    return [2 ** ((index - (count - 1)) / half_life) for index in range(count)]


def linear_fit(values: List[float], weights: List[float]) -> Tuple[float, float]:
    xs = [float(index) for index in range(len(values))]
    x_bar = weighted_mean(xs, weights)
    y_bar = weighted_mean(values, weights)
    denominator = sum(w * (x - x_bar) ** 2 for x, w in zip(xs, weights))
    if denominator <= 1e-9:
        return y_bar, 0.0
    slope = sum(
        w * (x - x_bar) * (y - y_bar)
        for x, y, w in zip(xs, values, weights)
    ) / denominator
    slope = clamp(slope, -0.25, 0.25)
    return y_bar - slope * x_bar, slope


def predict_core(log_views: List[float]) -> Dict[str, float]:
    weights = recency_weights(len(log_views))
    intercept, slope = linear_fit(log_views, weights)
    trend_next = intercept + slope * len(log_views)
    recent_start = max(0, len(log_views) - 12)
    recent_logs = log_views[recent_start:]
    recent_weights = weights[recent_start:]
    recent_level = weighted_median(recent_logs, recent_weights)
    sample_shrink = min(1.0, len(log_views) / 20.0)
    prediction = 0.58 * recent_level + 0.42 * trend_next
    prediction = recent_level + (prediction - recent_level) * sample_shrink
    residuals = [
        value - (intercept + slope * index)
        for index, value in enumerate(log_views)
    ]
    return {
        "prediction_log": prediction,
        "slope": slope,
        "sigma": robust_sigma(residuals),
        "recent_level_log": recent_level,
    }


def inferred_post_gap(rows: List[Dict[str, Any]]) -> Optional[float]:
    dates = [row.get("_published_dt") for row in rows if row.get("_published_dt")]
    if len(dates) < 3:
        return None
    gaps = [
        (later - earlier).total_seconds() / 86400.0
        for earlier, later in zip(dates, dates[1:])
        if later > earlier
    ]
    return median(gaps) if gaps else None


def target_weekday_adjustment(
    rows: List[Dict[str, Any]], log_views: List[float], target: Optional[datetime]
) -> float:
    if target is None:
        return 0.0
    indexed = [
        (row["_published_dt"].weekday(), value)
        for row, value in zip(rows, log_views)
        if row.get("_published_dt") is not None
    ]
    matching = [value for weekday, value in indexed if weekday == target.weekday()]
    if len(matching) < 3:
        return 0.0
    raw = median(matching) - median(value for _, value in indexed)
    return clamp(raw * min(1.0, len(matching) / 8.0), -0.25, 0.25)


def backtest(log_views: List[float]) -> Optional[float]:
    if len(log_views) < 10:
        return None
    holdout = min(5, max(2, len(log_views) // 4))
    errors = []
    for index in range(len(log_views) - holdout, len(log_views)):
        trained = log_views[:index]
        predicted = math.expm1(predict_core(trained)["prediction_log"])
        actual = math.expm1(log_views[index])
        if actual > 0:
            errors.append(abs(predicted - actual) / actual)
    return median(errors) if errors else None


def topic_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[float]] = {}
    for row in rows:
        topic = str(row.get("topic") or "").strip()
        if topic:
            buckets.setdefault(topic, []).append(float(row["views"]))
    ranked = sorted(
        (
            {"topic": topic, "posts": len(values), "median_views": round(median(values))}
            for topic, values in buckets.items()
        ),
        key=lambda item: item["median_views"],
        reverse=True,
    )
    return ranked[:5]


def forecast(
    rows: List[Dict[str, Any]],
    future_posts: int,
    publish_date: Optional[datetime],
    simulations: int,
) -> Dict[str, Any]:
    log_views = [math.log1p(float(row["views"])) for row in rows]
    model = predict_core(log_views)
    day_adjust = target_weekday_adjustment(rows, log_views, publish_date)
    first_log = model["prediction_log"] + day_adjust
    sigma = model["sigma"]
    gap_days = inferred_post_gap(rows)
    forecasts = []
    for step in range(future_posts):
        predicted_log = first_log + model["slope"] * step * min(1.0, len(rows) / 20.0)
        p50 = max(0.0, math.expm1(predicted_log))
        p20 = max(0.0, math.expm1(predicted_log - 0.841621 * sigma))
        p80 = max(0.0, math.expm1(predicted_log + 0.841621 * sigma))
        date_value = None
        if publish_date:
            date_value = publish_date + timedelta(days=(gap_days or 0.0) * step)
        forecasts.append(
            {
                "post": step + 1,
                "publish_at": date_value.isoformat(timespec="minutes") if date_value else None,
                "p20": round(p20),
                "p50": round(p50),
                "p80": round(p80),
            }
        )

    rng = random.Random(20260729)
    totals = []
    for _ in range(simulations):
        common_shock = rng.gauss(0.0, sigma * 0.55)
        total = 0.0
        for step in range(future_posts):
            independent = rng.gauss(0.0, sigma * math.sqrt(1 - 0.55 ** 2))
            predicted_log = (
                first_log
                + model["slope"] * step * min(1.0, len(rows) / 20.0)
                + common_shock
                + independent
            )
            total += max(0.0, math.expm1(predicted_log))
        totals.append(total)
    totals.sort()

    bt_error = backtest(log_views)
    if len(rows) >= 30 and sigma < 0.55 and (bt_error is None or bt_error < 0.6):
        confidence = "high"
    elif len(rows) >= 12 and sigma < 0.85 and (bt_error is None or bt_error < 1.0):
        confidence = "medium"
    else:
        confidence = "low"

    engagement_values = []
    for row in rows:
        views = float(row["views"])
        if views <= 0:
            continue
        interactions = (
            float(row.get("likes") or 0)
            + 2 * float(row.get("comments") or 0)
            + 3 * float(row.get("shares") or 0)
            + 2 * float(row.get("saves") or 0)
        )
        if interactions:
            engagement_values.append(interactions / views)

    date_coverage = sum(row.get("_published_dt") is not None for row in rows) / len(rows)
    return {
        "method": "recency-weighted robust log trend with deterministic simulation",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sample": {
            "valid_posts": len(rows),
            "date_coverage": round(date_coverage, 3),
            "recommended_minimum_met": len(rows) >= 12,
        },
        "next_post": forecasts[0],
        "future_posts": forecasts,
        "cumulative": {
            "posts": future_posts,
            "p20": round(percentile(totals, 0.20)),
            "p50": round(percentile(totals, 0.50)),
            "p80": round(percentile(totals, 0.80)),
        },
        "diagnostics": {
            "recent_median_views": round(math.expm1(model["recent_level_log"])),
            "trend_per_post_pct": round((math.exp(model["slope"]) - 1) * 100, 1),
            "log_volatility_sigma": round(sigma, 3),
            "weekday_adjustment_pct": round((math.exp(day_adjust) - 1) * 100, 1),
            "backtest_median_absolute_pct_error": (
                round(bt_error * 100, 1) if bt_error is not None else None
            ),
            "weighted_engagement_rate": (
                round(median(engagement_values), 4) if engagement_values else None
            ),
            "inferred_post_gap_days": round(gap_days, 2) if gap_days else None,
            "top_topics": topic_summary(rows),
        },
        "confidence": confidence,
        "limits": [
            "区间描述自然流量波动，不保证单条结果",
            "换赛道、投流、平台分发规则变化和突发热点会使预测失效",
            "不同平台或不同流量口径不得混合建模",
        ],
    }


def markdown_report(payload: Dict[str, Any]) -> str:
    next_post = payload["next_post"]
    cumulative = payload["cumulative"]
    diagnostics = payload["diagnostics"]
    backtest_value = diagnostics["backtest_median_absolute_pct_error"]
    backtest_text = f"{backtest_value:.1f}%" if backtest_value is not None else "样本不足"
    return f"""# 账号流量预测

## 结论

- 下一条 P50：{compact_number(next_post['p50'])}
- 下一条 P20–P80：{compact_number(next_post['p20'])}–{compact_number(next_post['p80'])}
- 未来 {cumulative['posts']} 条累计 P50：{compact_number(cumulative['p50'])}
- 累计 P20–P80：{compact_number(cumulative['p20'])}–{compact_number(cumulative['p80'])}
- 置信等级：{payload['confidence']}

## 诊断

- 有效样本：{payload['sample']['valid_posts']} 条
- 近期中位播放：{compact_number(diagnostics['recent_median_views'])}
- 每条趋势：{diagnostics['trend_per_post_pct']:+.1f}%
- 波动 σ：{diagnostics['log_volatility_sigma']:.3f}
- 回测中位绝对误差：{backtest_text}

## 使用边界

- 这是自然流量概率区间，不是播放量承诺。
- 换赛道、投流、平台规则变化或异常热点会使区间失效。
- 发布后把真实数据追加到历史表，再滚动更新预测。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_csv")
    parser.add_argument("--future-posts", type=int, default=5)
    parser.add_argument("--publish-date", help="下一条计划发布时间，ISO 格式")
    parser.add_argument("--newest-first", action="store_true")
    parser.add_argument("--simulations", type=int, default=5000)
    parser.add_argument("--out-json")
    parser.add_argument("--out-md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.future_posts <= 30:
        raise SystemExit("--future-posts 必须在 1–30 之间")
    if not 500 <= args.simulations <= 50000:
        raise SystemExit("--simulations 必须在 500–50000 之间")
    publish_date = None
    if args.publish_date:
        try:
            publish_date = datetime.fromisoformat(args.publish_date.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SystemExit(f"--publish-date 不是有效 ISO 日期：{exc}")
    try:
        rows = read_history(args.history_csv, newest_first=args.newest_first)
        payload = forecast(rows, args.future_posts, publish_date, args.simulations)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"预测失败：{exc}")
    write_json(args.out_json, payload)
    write_text(args.out_md, markdown_report(payload))
    if args.out_json:
        print(f"forecast_json={Path(args.out_json).resolve()}")
    if args.out_md:
        print(f"forecast_md={Path(args.out_md).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
