"""Execution node: run SQL in the local read-only DuckDB sandbox."""

from __future__ import annotations

from typing import Any

from ..db import execute_query
from ..state import AgentContext, trace_entry


def execution_node(state: dict[str, Any], *, ctx: AgentContext) -> dict[str, Any]:
    attempt = int(state.get("sql_attempts", 1))
    try:
        result = execute_query(
            ctx.settings.db_path,
            state.get("sql", ""),
            max_rows=ctx.settings.max_result_rows,
        )
    except Exception as exc:  # execution errors are designed to trigger repair
        message = f"{type(exc).__name__}: {exc}"
        detail = f"第 {attempt} 次执行失败：{message[:220]}"
        return {
            "sql_error": message,
            "status": "execution_error",
            "trace": [trace_entry("Execution", detail)],
        }

    detail = (
        f"执行成功：{result.row_count} 行，{result.elapsed_ms} ms"
        + ("（已截断）" if result.truncated else "")
    )
    return {
        "result_df": result.df,
        "row_count": result.row_count,
        "exec_ms": result.elapsed_ms,
        "truncated": result.truncated,
        "sql_error": None,
        "status": "executed",
        "trace": [trace_entry("Execution", detail)],
    }


def route_after_execution(state: dict[str, Any], *, ctx: AgentContext) -> str:
    """Conditional edge: repair (max 3), attribution, insight, or fail."""
    if state.get("sql_error"):
        attempts = int(state.get("sql_attempts", 1))
        if attempts < ctx.max_attempts:
            return "codegen"
        return "failure"
    if state.get("action") == "attribution":
        return "attribution"
    return "insight"
