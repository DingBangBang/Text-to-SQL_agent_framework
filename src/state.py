"""Shared LangGraph state definition."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated, Any, TypedDict

from .config import Settings
from .llm import BaseLLM
from .semantic import SemanticRegistry


class AgentState(TypedDict, total=False):
    question: str
    route: dict[str, Any]
    metric: str
    time_range: str
    action: str
    needs_clarification: bool
    clarification: str
    semantic: dict[str, Any]
    assumption: str

    sql: str
    sql_source: str
    sql_error: str | None
    sql_attempts: int
    retry_count: int

    result_df: Any
    row_count: int
    exec_ms: int
    truncated: bool

    attribution: dict[str, Any]
    insight: str
    status: str

    trace: Annotated[list[dict[str, Any]], operator.add]
    llm_calls: Annotated[list[dict[str, Any]], operator.add]


@dataclass
class AgentContext:
    """Dependencies shared by every node (kept out of serialisable state)."""

    settings: Settings
    registry: SemanticRegistry
    llm: BaseLLM
    schema: str = ""
    today: date = field(default_factory=date.today)

    @property
    def max_attempts(self) -> int:
        """First attempt + ``max_sql_retries`` self-corrections."""
        return self.settings.max_sql_retries + 1


def trace_entry(node: str, detail: str) -> dict[str, str]:
    return {"node": node, "detail": detail}


def llm_entry(response: Any, task: str) -> dict[str, Any]:
    return {
        "task": task,
        "mock": bool(getattr(response, "mock", False)),
        "model": getattr(response, "model", "unknown"),
        "input_tokens": int(getattr(response, "input_tokens", 0)),
        "output_tokens": int(getattr(response, "output_tokens", 0)),
        "cost_cny": float(getattr(response, "cost_cny", 0.0)),
        "truncated": bool(getattr(response, "input_truncated", False)),
    }
