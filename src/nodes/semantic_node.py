"""Semantic node: resolve metric keys into governed expressions and windows."""

from __future__ import annotations

from typing import Any

from ..i18n import tr
from ..state import AgentContext, trace_entry


def semantic_node(state: dict[str, Any], *, ctx: AgentContext) -> dict[str, Any]:
    metric = ctx.registry.get(state["metric"])
    window = ctx.registry.resolve_time_range(
        state.get("time_range"), ctx.today, ctx.language
    )
    if window is None:
        raise ValueError(
            tr(
                ctx.language,
                f"无法解析时间窗口：{state.get('time_range')}",
                f"Unable to resolve time window: {state.get('time_range')}",
            )
        )

    previous = ctx.registry.previous_window(window, ctx.language)
    time_dimension = next(
        (dim.key for dim in metric.dimensions.values() if dim.type == "time"),
        metric.time_column,
    )
    semantic = {
        "key": metric.key,
        "label": metric.display_label(ctx.language),
        "label_zh": metric.label,
        "label_en": metric.display_label("en"),
        "description": metric.display_description(ctx.language),
        "expression": metric.expression,
        "table": metric.table,
        "alias": metric.alias,
        "time_column": metric.time_expr,
        "time_dimension": time_dimension,
        "value_column": metric.value_column,
        "unit": metric.unit,
        "filters": list(metric.filters),
        "dimensions": [
            {"key": dim.key, "label": dim.label, "type": dim.type}
            for dim in metric.dimensions.values()
        ],
        "time_window": {
            "key": window.key,
            "label": window.label,
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            "days": window.days,
        },
        "previous_window": {
            "key": previous.key,
            "label": previous.label,
            "start": previous.start.isoformat(),
            "end": previous.end.isoformat(),
            "days": previous.days,
        },
    }
    detail = (
        tr(
            ctx.language,
            f"{metric.label} → `{metric.expression}`；",
            f"{metric.display_label('en')} → `{metric.expression}`; ",
        )
        + tr(
            ctx.language,
            f"窗口 {window.start} ~ {window.end}（{window.days} 天）；",
            f"window {window.start} ~ {window.end} ({window.days} days); ",
        )
        + tr(
            ctx.language,
            f"固定过滤 {', '.join(metric.filters) or '无'}",
            f"fixed filters: {', '.join(metric.filters) or 'none'}",
        )
    )
    return {
        "semantic": semantic,
        "status": "semantic_ready",
        "trace": [trace_entry("Semantic", detail)],
    }
