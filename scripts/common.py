#!/usr/bin/env python3
"""Shared CSV and numeric helpers for self-media-director."""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional


ALIASES = {
    "published_at": ["published_at", "publish_time", "published", "date", "发布时间", "发布日期", "日期"],
    "views": ["views", "view_count", "播放量", "浏览量", "观看次数"],
    "title": ["title", "video_title", "标题", "视频标题"],
    "topic": ["topic", "content_pillar", "主题", "选题", "内容支柱"],
    "format": ["format", "video_format", "形式", "视频形式"],
    "likes": ["likes", "like_count", "点赞", "点赞量"],
    "comments": ["comments", "comment_count", "评论", "评论量"],
    "shares": ["shares", "share_count", "分享", "转发"],
    "saves": ["saves", "save_count", "收藏", "收藏量"],
    "followers": ["followers", "follower_count", "发布时粉丝数", "粉丝数"],
    "duration_sec": ["duration_sec", "duration", "时长", "视频时长"],
    "completion_rate": ["completion_rate", "完播率"],
}

NUMERIC_FIELDS = {
    "views",
    "likes",
    "comments",
    "shares",
    "saves",
    "followers",
    "duration_sec",
    "completion_rate",
}


def normalized_header(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(value).strip().lower())


ALIAS_LOOKUP = {
    normalized_header(alias): canonical
    for canonical, aliases in ALIASES.items()
    for alias in aliases
}


def parse_number(value: Any, percent_as_ratio: bool = False) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        if percent_as_ratio and number > 1:
            return number / 100.0
        return number
    text = str(value).strip().lower().replace(",", "").replace("，", "")
    if not text or text in {"-", "—", "na", "n/a", "null", "none"}:
        return None
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1]
    multiplier = 1.0
    suffixes = {
        "万": 10000.0,
        "w": 10000.0,
        "k": 1000.0,
        "m": 1000000.0,
        "b": 1000000000.0,
    }
    if text and text[-1] in suffixes:
        multiplier = suffixes[text[-1]]
        text = text[:-1]
    try:
        number = float(text) * multiplier
    except ValueError:
        return None
    if is_percent or (percent_as_ratio and number > 1):
        number /= 100.0
    return number if math.isfinite(number) else None


def parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("/", "-")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%m-%d-%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def read_history(path: str, newest_first: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV 缺少表头")
        header_map = {
            field: ALIAS_LOOKUP.get(normalized_header(field), field.strip())
            for field in reader.fieldnames
        }
        for raw_index, raw in enumerate(reader):
            row: Dict[str, Any] = {"_row": raw_index + 2}
            for field, value in raw.items():
                canonical = header_map.get(field, field)
                row[canonical] = value.strip() if isinstance(value, str) else value
            for field in NUMERIC_FIELDS:
                if field in row:
                    row[field] = parse_number(
                        row[field], percent_as_ratio=(field == "completion_rate")
                    )
            row["_published_dt"] = parse_datetime(row.get("published_at"))
            if row.get("views") is not None and row["views"] >= 0:
                rows.append(row)
    if newest_first:
        rows.reverse()
    elif rows and all(row.get("_published_dt") is not None for row in rows):
        rows.sort(key=lambda item: item["_published_dt"])
    if len(rows) < 5:
        raise ValueError(f"有效历史视频只有 {len(rows)} 条；数值预测至少需要 5 条")
    return rows


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    value_list = list(values)
    weight_list = list(weights)
    total = sum(weight_list)
    if not value_list or total <= 0:
        raise ValueError("无法计算加权均值")
    return sum(v * w for v, w in zip(value_list, weight_list)) / total


def weighted_median(values: Iterable[float], weights: Iterable[float]) -> float:
    pairs = sorted(zip(values, weights), key=lambda pair: pair[0])
    total = sum(weight for _, weight in pairs)
    cursor = 0.0
    for value, weight in pairs:
        cursor += weight
        if cursor >= total / 2.0:
            return value
    return pairs[-1][0]


def robust_sigma(residuals: List[float]) -> float:
    if not residuals:
        return 0.35
    center = median(residuals)
    mad = median(abs(value - center) for value in residuals)
    return min(1.2, max(0.15, 1.4826 * mad))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def percentile(sorted_values: List[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("空序列没有分位数")
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def write_json(path: Optional[str], payload: Dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def write_text(path: Optional[str], text: str) -> None:
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.rstrip() + "\n", encoding="utf-8")


def compact_number(value: float) -> str:
    if value >= 100000000:
        return f"{value / 100000000:.1f}亿"
    if value >= 10000:
        return f"{value / 10000:.1f}万"
    if value >= 1000:
        return f"{value / 1000:.1f}千"
    return str(int(round(value)))
