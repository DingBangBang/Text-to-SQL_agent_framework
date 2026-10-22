"""LangGraph assembly: Router → Semantic → CodeGen → Execution → (Attribution) → Insight."""

from __future__ import annotations

import time
from datetime import date
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from .config import Settings
from .db import schema_ddl
from .i18n import normalize_language
from .llm import build_llm
from .nodes.attribution import attribution_node
from .nodes.codegen import codegen_node
from .nodes.execution import execution_node, route_after_execution
from .nodes.insight import failure_node, insight_node
from .nodes.router import router_node
from .nodes.semantic_node import semantic_node
from .semantic import SemanticRegistry
from .state import AgentContext, AgentState

NODE_LABELS = {
    "router": "Router · 意图路由",
    "semantic": "Semantic · 指标口径",
    "codegen": "CodeGen · Text-to-SQL",
    "execution": "Execution · 沙箱执行",
    "attribution": "Attribution · 多维归因",
    "insight": "Insight · 洞察生成",
    "failure": "Failure · 人工介入",
}


def build_context(
    settings: Settings | None = None,
    *,
    llm: Any | None = None,
    registry: SemanticRegistry | None = None,
    today: date | None = None,
    language: str = "zh",
) -> AgentContext:
    settings = settings or Settings.from_env()
    registry = registry or SemanticRegistry.load(settings.metrics_path)
    llm = llm or build_llm(settings)
    schema = schema_ddl(settings.db_path) if settings.db_path.exists() else ""
    return AgentContext(
        settings=settings,
        registry=registry,
        llm=llm,
        schema=schema,
        today=today or date.today(),
        language=normalize_language(language),
    )


def build_graph(
    settings: Settings | None = None,
    *,
    llm: Any | None = None,
    registry: SemanticRegistry | None = None,
    today: date | None = None,
    checkpointer: Any | None = None,
    language: str = "zh",
):
    """Compile the state machine. Tests inject MockLLM here to stay offline."""
    ctx = build_context(
        settings, llm=llm, registry=registry, today=today, language=language
    )

    builder = StateGraph(AgentState)
    builder.add_node("router", partial(router_node, ctx=ctx))
    builder.add_node("semantic", partial(semantic_node, ctx=ctx))
    builder.add_node("codegen", partial(codegen_node, ctx=ctx))
    builder.add_node("execution", partial(execution_node, ctx=ctx))
    builder.add_node("attribution", partial(attribution_node, ctx=ctx))
    builder.add_node("insight", partial(insight_node, ctx=ctx))
    builder.add_node("failure", partial(failure_node, ctx=ctx))

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        lambda state: "end" if state.get("needs_clarification") else "semantic",
        {"semantic": "semantic", "end": END},
    )
    builder.add_edge("semantic", "codegen")
    builder.add_edge("codegen", "execution")
    builder.add_conditional_edges(
        "execution",
        lambda state: route_after_execution(state, ctx=ctx),
        {
            "codegen": "codegen",
            "attribution": "attribution",
            "insight": "insight",
            "failure": "failure",
        },
    )
    builder.add_edge("attribution", "insight")
    builder.add_edge("insight", END)
    builder.add_edge("failure", END)

    return builder.compile(checkpointer=checkpointer)


def run_agent(
    question: str,
    *,
    settings: Settings | None = None,
    llm: Any | None = None,
    registry: SemanticRegistry | None = None,
    today: date | None = None,
    graph: Any | None = None,
    language: str = "zh",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute one full agent run and return ``(state, run_meta)``."""
    compiled = graph or build_graph(
        settings,
        llm=llm,
        registry=registry,
        today=today,
        language=language,
    )
    started = time.perf_counter()
    state = compiled.invoke(
        {"question": question, "trace": [], "llm_calls": []},
        config={"recursion_limit": 30},
    )
    elapsed = time.perf_counter() - started

    calls = list(state.get("llm_calls") or [])
    total_in = sum(int(call.get("input_tokens", 0)) for call in calls)
    total_out = sum(int(call.get("output_tokens", 0)) for call in calls)
    cost = sum(float(call.get("cost_cny", 0.0)) for call in calls)
    meta = {
        "elapsed_s": round(elapsed, 3),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_cny": round(cost, 6),
        "calls": calls,
        "mock": all(bool(call.get("mock")) for call in calls) if calls else True,
        "model": calls[0].get("model", "mock") if calls else "mock",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return dict(state), meta
