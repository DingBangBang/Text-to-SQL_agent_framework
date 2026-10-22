"""Router node: natural language -> governed intent JSON."""

from __future__ import annotations

from typing import Any

from ..i18n import tr
from ..llm import LLMError, extract_json
from ..prompts import router_messages
from ..routing_rules import detect_time_range, is_attribution
from ..state import AgentContext, llm_entry, trace_entry


def _clarification(metric_missing: bool, time_missing: bool, language: str) -> str:
    if metric_missing and time_missing:
        return tr(
            language,
            "请问您想分析哪个指标（如交易量、链上交易量、活跃用户）"
            "以及哪个时间范围（如过去7天、上周）？",
            "Which metric (e.g. trading volume, on-chain volume, active users) "
            "and which time range (e.g. last 7 days, last week) should I analyze?",
        )
    if metric_missing:
        return tr(
            language,
            "请问您想分析哪个指标？目前支持交易量、链上交易量、活跃用户、手续费收入等。",
            "Which metric should I analyze? Available options include trading volume, "
            "on-chain volume, active users, and fee revenue.",
        )
    if time_missing:
        return tr(
            language,
            "请问您想看哪个时间范围的数据？（如过去7天、上周、过去30天）",
            "Which time range should I use? (e.g. last 7 days, last week, last 30 days)",
        )
    return ""


def router_node(state: dict[str, Any], *, ctx: AgentContext) -> dict[str, Any]:
    question = str(state.get("question", "")).strip()
    calls: list[dict[str, Any]] = []
    source = "fallback-rules"

    payload: dict[str, Any] = {}
    try:
        response = ctx.llm.complete(
            router_messages(question, ctx.registry, ctx.language),
            json_mode=True,
            task="router",
            context={
                "question": question,
                "registry": ctx.registry,
                "today": ctx.today,
                "language": ctx.language,
            },
        )
        calls.append(llm_entry(response, "router"))
        payload = extract_json(response.text)
        source = "mock-rules" if response.mock else "deepseek"
    except LLMError:
        # One deterministic fallback, no extra API call: routing must never
        # stall the pipeline or burn budget on retries.
        payload = {}

    metric = None
    metric_key = payload.get("metric")
    if isinstance(metric_key, str) and metric_key in ctx.registry.metrics:
        metric = ctx.registry.get(metric_key)
    if metric is None:
        metric = ctx.registry.find_metric(question)
    if metric is None and ("链上" in question or "onchain" in question.lower()):
        metric = ctx.registry.metrics.get("onchain_volume")

    window = ctx.registry.resolve_time_range(
        payload.get("time_range"), ctx.today, ctx.language
    )
    if window is None:
        detected = detect_time_range(question, ctx.registry)
        window = ctx.registry.resolve_time_range(detected, ctx.today, ctx.language)

    action = payload.get("action")
    if action not in ("query", "attribution"):
        action = "attribution" if is_attribution(question) else "query"

    metric_missing = metric is None
    time_missing = window is None
    clarification = str(payload.get("clarification") or "").strip()
    if not clarification:
        clarification = _clarification(metric_missing, time_missing, ctx.language)

    route = {
        "metric": metric.key if metric else None,
        "time_range": window.key if window else None,
        "action": action,
        "source": source,
    }
    if ctx.is_english:
        detail = (
            f"metric={route['metric']}, window={route['time_range']}, "
            f"action={action}, source={source}"
        )
        if metric_missing or time_missing:
            detail += "; missing information, asking a clarification question"
    else:
        detail = (
            f"指标={route['metric']}，时间窗口={route['time_range']}，"
            f"动作={action}，来源={source}"
        )
        if metric_missing or time_missing:
            detail += "；信息不足，生成反问"

    return {
        "route": route,
        "metric": route["metric"] or "",
        "time_range": route["time_range"] or "",
        "action": action,
        "needs_clarification": metric_missing or time_missing,
        "clarification": clarification,
        "status": "needs_clarification" if (metric_missing or time_missing) else "routed",
        "trace": [trace_entry("Router", detail)],
        "llm_calls": calls,
    }
