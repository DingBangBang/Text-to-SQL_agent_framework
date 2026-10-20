"""Attribution node: deterministic multi-dimensional decomposition.

Attribution never asks the LLM to write SQL. It reuses metrics.yaml templates
to compare the current window against the previous equal-length window, split
by every configured category dimension (symbol, chain_type, ...).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from ..db import execute_query
from ..semantic import TimeWindow
from ..state import AgentContext, trace_entry


def _window(data: dict[str, Any]) -> TimeWindow:
    return TimeWindow(
        key=data["key"],
        label=data["label"],
        start=date.fromisoformat(data["start"]),
        end=date.fromisoformat(data["end"]),
    )


def _scalar(db_path: Any, sql: str) -> float:
    frame = execute_query(db_path, sql, max_rows=1).df
    if frame.empty:
        return 0.0
    value = frame.iloc[0, 0]
    return float(value) if value is not None else 0.0


def attribution_node(state: dict[str, Any], *, ctx: AgentContext) -> dict[str, Any]:
    semantic = state["semantic"]
    metric = ctx.registry.get(semantic["key"])
    window = _window(semantic["time_window"])
    previous = _window(semantic["previous_window"])

    current_total = _scalar(
        ctx.settings.db_path, ctx.registry.build_total_sql(metric, window)
    )
    previous_total = _scalar(
        ctx.settings.db_path, ctx.registry.build_total_sql(metric, previous)
    )
    change_value = current_total - previous_total
    change_pct = (
        (current_total / previous_total - 1) * 100 if previous_total else None
    )

    breakdowns: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    for dimension in ctx.registry.attribution_dimensions(metric)[:3]:
        try:
            result = execute_query(
                ctx.settings.db_path,
                ctx.registry.build_attribution_sql(metric, window, dimension),
                max_rows=50,
            )
            breakdowns[dimension] = result.df
        except Exception as exc:
            warnings.append(f"{dimension}: {type(exc).__name__}: {exc}")

    attribution = {
        "window": semantic["time_window"],
        "previous_window": semantic["previous_window"],
        "current_total": current_total,
        "previous_total": previous_total,
        "change_value": change_value,
        "change_pct": change_pct,
        "breakdowns": breakdowns,
        "warnings": warnings,
    }

    top_driver = ""
    if breakdowns:
        frames = []
        for dimension, frame in breakdowns.items():
            if frame is None or frame.empty:
                continue
            candidate = frame.copy()
            candidate["_dimension"] = dimension
            frames.append(candidate)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            negative = combined[combined["change_value"] < 0]
            if not negative.empty:
                worst = negative.sort_values("change_value").iloc[0]
                top_driver = (
                    f"，最大拖累 {worst['_dimension']}={worst['dimension_value']}"
                    f"（{worst['change_value']:,.2f}）"
                )

    pct_text = f"{change_pct:+.2f}%" if change_pct is not None else "N/A"
    detail = (
        f"{window.label}: {current_total:,.2f} vs 上一周期 {previous_total:,.2f}"
        f"（{pct_text}）；拆解维度：{', '.join(breakdowns) or '无'}{top_driver}"
    )
    if warnings:
        detail += f"；{len(warnings)} 个维度失败已跳过"

    return {
        "attribution": attribution,
        "status": "attributed",
        "trace": [trace_entry("Attribution", detail)],
    }
