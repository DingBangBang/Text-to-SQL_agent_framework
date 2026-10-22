"""Insight node: conclusion-first narrative grounded in executed numbers."""

from __future__ import annotations

from typing import Any

from ..db import execute_query
from ..i18n import tr
from ..llm import LLMError
from ..prompts import insight_messages
from ..reporting import (
    attribution_tables_markdown,
    build_insight_context,
    df_to_markdown,
)
from ..semantic import TimeWindow
from ..state import AgentContext, llm_entry, trace_entry


def _fallback_insight(
    payload: dict[str, Any], reason: str, language: str = "zh"
) -> str:
    driver = payload.get("top_driver")
    if language == "en":
        lines = [
            "**Conclusion**",
            "",
            f"{payload['window_label']} {payload['metric_label']}: "
            f"{payload['current_value']}, versus {payload['previous_value']} "
            f"in the previous period ({payload['change_pct']}).",
            "",
            "**Evidence**",
            "",
            f"- Current: {payload['current_value']}; previous: "
            f"{payload['previous_value']}.",
        ]
        if driver:
            lines.append(
                f"- Largest drag: {driver['name']}, change {driver['change']} "
                f"({driver['share']} of the total change)."
            )
        lines += [
            "- All figures come from aggregated queries in the local DuckDB sandbox.",
            "",
            "**Recommendations**",
            "",
            "1. Investigate the primary declining dimension (market depth, on-chain congestion, campaign timing).",
            "2. Add a period-over-period alert that triggers attribution when the change falls below -10%.",
            "",
            f"> LLM unavailable; this section was generated locally ({reason}).",
        ]
        return "\n".join(lines)

    lines = [
        "**结论**",
        "",
        f"{payload['window_label']} {payload['metric_label']}为 "
        f"{payload['current_value']}，上一周期 {payload['previous_value']}，"
        f"环比 {payload['change_pct']}。",
        "",
        "**数据支撑**",
        "",
        f"- 当前值：{payload['current_value']}；上一周期：{payload['previous_value']}。",
    ]
    if driver:
        lines.append(
            f"- 最大拖累：{driver['name']}，变化 {driver['change']}"
            f"（占比 {driver['share']}）。"
        )
    lines += [
        "- 以上数字均来自本地 DuckDB 只读沙箱的聚合结果。",
        "",
        "**业务建议**",
        "",
        "1. 对主要下滑维度做专项排查（做市深度、链上拥堵、活动节奏）。",
        "2. 为该指标配置环比告警，跌破 -10% 自动触发归因。",
        "",
        f"> LLM 不可用，本段由本地规则生成（{reason}）。",
    ]
    return "\n".join(lines)


def _window(data: dict[str, Any]) -> TimeWindow:
    from datetime import date

    return TimeWindow(
        key=data["key"],
        label=data["label"],
        start=date.fromisoformat(data["start"]),
        end=date.fromisoformat(data["end"]),
    )


def _ensure_totals(
    state: dict[str, Any], ctx: AgentContext
) -> dict[str, Any]:
    """Query path has no attribution yet; add accurate totals for the KPI cards."""
    if state.get("attribution"):
        return state
    semantic = state["semantic"]
    metric = ctx.registry.get(semantic["key"])
    window = _window(semantic["time_window"])
    previous = _window(semantic["previous_window"])

    def scalar(sql: str) -> float:
        frame = execute_query(ctx.settings.db_path, sql, max_rows=1).df
        if frame.empty or frame.iloc[0, 0] is None:
            return 0.0
        return float(frame.iloc[0, 0])

    current = scalar(ctx.registry.build_total_sql(metric, window))
    prior = scalar(ctx.registry.build_total_sql(metric, previous))
    enriched = dict(state)
    enriched["attribution"] = {
        "window": semantic["time_window"],
        "previous_window": semantic["previous_window"],
        "current_total": current,
        "previous_total": prior,
        "change_value": current - prior,
        "change_pct": (current / prior - 1) * 100 if prior else None,
        "breakdowns": {},
        "warnings": [],
    }
    return enriched


def insight_node(state: dict[str, Any], *, ctx: AgentContext) -> dict[str, Any]:
    semantic = state["semantic"]
    metric = ctx.registry.get(semantic["key"])
    unit = semantic.get("unit", "")
    state = _ensure_totals(state, ctx)
    payload = build_insight_context(state, unit, ctx.language)
    window = _window(semantic["time_window"])

    data_table = df_to_markdown(state.get("result_df"), max_rows=20, unit=unit)
    attribution_table = attribution_tables_markdown(
        state, unit=unit, language=ctx.language
    )
    calls: list[dict[str, Any]] = []

    try:
        response = ctx.llm.complete(
            insight_messages(
                question=state.get("question", ""),
                metric=metric,
                window=window,
                data_table=data_table,
                attribution_table=attribution_table,
                current_value=payload["current_value"],
                previous_value=payload["previous_value"],
                change_pct=payload["change_pct"],
                language=ctx.language,
            ),
            task="insight",
            context={
                **payload,
                "row_count": state.get("row_count", 0),
                "sql_error": state.get("sql_error"),
                "language": ctx.language,
            },
        )
        calls.append(llm_entry(response, "insight"))
        insight = response.text.strip()
        if ctx.is_english:
            note = (
                f"Insight generated ({'mock' if response.mock else 'deepseek'}, "
                f"{response.input_tokens}+{response.output_tokens} tokens, "
                f"${ctx.settings.cost_usd(response.cost_cny):.6f})"
            )
        else:
            note = (
                f"洞察生成完成（{'mock' if response.mock else 'deepseek'}，"
                f"{response.input_tokens}+{response.output_tokens} tokens，"
                f"¥{response.cost_cny:.4f}）"
            )
        source = "mock" if response.mock else "deepseek"
    except LLMError as exc:
        insight = _fallback_insight(payload, type(exc).__name__, ctx.language)
        note = tr(
            ctx.language,
            f"LLM 不可用，已使用本地规则生成洞察：{type(exc).__name__}",
            f"LLM unavailable; insight generated locally: {type(exc).__name__}",
        )
        source = "local-fallback"

    return {
        "attribution": state.get("attribution"),
        "insight": insight,
        "status": "completed",
        "trace": [
            trace_entry(
                "Insight",
                tr(ctx.language, f"{note}；来源={source}", f"{note}; source={source}"),
            )
        ],
        "llm_calls": calls,
    }


def failure_node(state: dict[str, Any], *, ctx: AgentContext) -> dict[str, Any]:
    """Terminal node after exhausting the repair budget."""
    if ctx.is_english:
        insight = (
            "Execution failed, human intervention required.\n\n"
            f"Made {int(state.get('sql_attempts', 0))} SQL attempts "
            f"(first attempt + {ctx.settings.max_sql_retries} self-corrections) "
            "without success.\n\n"
            f"Last error: `{state.get('sql_error', 'unknown error')}`\n\n"
            f"Last generated SQL:\n```sql\n{state.get('sql', '')}\n```"
        )
    else:
        insight = (
            "执行失败，请人工介入。\n\n"
            f"已进行 {int(state.get('sql_attempts', 0))} 次 SQL 尝试"
            f"（首次 + {ctx.settings.max_sql_retries} 次自纠错），仍无法执行。\n\n"
            f"最后一次错误：`{state.get('sql_error', '未知错误')}`\n\n"
            f"最后生成的 SQL：\n```sql\n{state.get('sql', '')}\n```"
        )
    return {
        "insight": insight,
        "status": "failed",
        "trace": [
            trace_entry(
                "Failure",
                tr(
                    ctx.language,
                    f"达到最大自纠错次数 {ctx.settings.max_sql_retries}，停止重试",
                    f"Reached the maximum of {ctx.settings.max_sql_retries} "
                    "self-corrections; stopped retrying",
                ),
            )
        ],
    }
