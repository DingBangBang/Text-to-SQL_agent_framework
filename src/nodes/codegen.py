"""CodeGen node: LLM writes DuckDB SQL, with a deterministic SQL fallback."""

from __future__ import annotations

from datetime import date
from typing import Any

from ..db import clean_sql
from ..llm import LLMError
from ..prompts import codegen_messages
from ..semantic import TimeWindow
from ..state import AgentContext, llm_entry, trace_entry


def _window_from_semantic(semantic: dict[str, Any]) -> TimeWindow:
    data = semantic["time_window"]
    return TimeWindow(
        key=data["key"],
        label=data["label"],
        start=date.fromisoformat(data["start"]),
        end=date.fromisoformat(data["end"]),
    )


def codegen_node(state: dict[str, Any], *, ctx: AgentContext) -> dict[str, Any]:
    semantic = state["semantic"]
    metric = ctx.registry.get(semantic["key"])
    window = _window_from_semantic(semantic)
    dimension = semantic.get("time_dimension", metric.time_column)

    # Deterministic, governance-safe SQL assembled from metrics.yaml. It is
    # both the MockLLM answer and the fallback if the API is unavailable.
    fallback_sql = ctx.registry.build_metric_sql(metric, window, dimension=dimension)
    attempt = int(state.get("sql_attempts", 0)) + 1
    previous_sql = state.get("sql") if state.get("sql_error") else None
    error = state.get("sql_error") if state.get("sql_error") else None

    calls: list[dict[str, Any]] = []
    source = "semantic-template"
    sql = fallback_sql
    note = ""
    try:
        response = ctx.llm.complete(
            codegen_messages(
                question=state.get("question", ""),
                metric=metric,
                window=window,
                schema=ctx.schema,
                dimension=dimension,
                registry=ctx.registry,
                previous_sql=previous_sql,
                error=error,
            ),
            task="codegen",
            context={
                "fallback_sql": fallback_sql,
                "metric_key": metric.key,
                "window": window,
                "attempt": attempt,
            },
        )
        calls.append(llm_entry(response, "codegen"))
        candidate = clean_sql(response.text)
        if candidate:
            sql = candidate
            source = "mock-rules" if response.mock else "deepseek"
        else:
            note = "（模型返回空 SQL，已回退到语义模板）"
    except LLMError as exc:
        note = f"（LLM 不可用，已回退到语义模板：{type(exc).__name__}）"

    detail = f"第 {attempt}/{ctx.max_attempts} 次生成 SQL，来源={source}"
    if error:
        detail += f"；根据报错修复：{error[:120]}"
    if note:
        detail += f" {note}"

    return {
        "sql": sql,
        "sql_source": source,
        "sql_attempts": attempt,
        "retry_count": max(attempt - 1, 0),
        "status": "sql_ready",
        "trace": [trace_entry("CodeGen", detail)],
        "llm_calls": calls,
    }
