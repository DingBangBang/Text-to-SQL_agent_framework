"""Deterministic MockLLM: full agent behaviour with zero API cost.

MockLLM is *not* a stub that returns fixed strings.  It performs the same
contract as the real model — routing JSON, SQL emission and a structured
narrative — using local rules.  That makes the entire LangGraph pipeline,
including retries and attribution, testable offline.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from .config import Settings
from .i18n import tr
from .llm import LLMResponse
from .routing_rules import detect_time_range, is_attribution
from .semantic import SemanticRegistry
from .token_budget import TokenBudget, count_tokens


class MockLLM:
    model = "mock-local"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        budget: TokenBudget | None = None,
        scripted: dict[str, list[str]] | None = None,
        **_: Any,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.budget = budget or TokenBudget(self.settings.max_input_tokens)
        self.scripted = {key: list(values) for key, values in (scripted or {}).items()}
        self.calls: list[dict[str, Any]] = []
        self.spent_cny = 0.0

    def reset_meter(self) -> None:
        self.calls = []
        self.spent_cny = 0.0

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        task: str = "generic",
        context: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        context = context or {}
        _safe_messages, report = self.budget.enforce(messages)
        prompt_tokens = count_tokens(
            json.dumps(_safe_messages, ensure_ascii=False)
        )
        queue = self.scripted.get(task, [])
        if queue:
            text = queue.pop(0)
        elif task == "router":
            text = self._route(context)
        elif task == "codegen":
            text = self._codegen(context)
        elif task == "insight":
            text = self._insight(context)
        else:
            text = "{}" if json_mode else "OK"
        self.calls.append(
            {
                "task": task,
                "model": self.model,
                "input_tokens": prompt_tokens,
                "output_tokens": count_tokens(text),
                "cost_cny": 0.0,
                "latency_ms": 0,
            }
        )
        return LLMResponse(
            text=text,
            input_tokens=prompt_tokens,
            output_tokens=count_tokens(text),
            cost_cny=0.0,
            model=self.model,
            mock=True,
            input_truncated=report.truncated,
            budget=report,
        )

    # -- task handlers ----------------------------------------------------
    def _route(self, context: dict[str, Any]) -> str:
        question: str = context.get("question", "")
        registry: SemanticRegistry = context["registry"]
        language = context.get("language", "zh")

        metric = registry.find_metric(question)
        if metric is None and ("链上" in question or "onchain" in question.lower()):
            metric = registry.metrics.get("onchain_volume")
        time_key = detect_time_range(question, registry)
        action = "attribution" if is_attribution(question) else "query"

        clarification = ""
        if metric is None and time_key is None:
            clarification = tr(
                language,
                "请问您想分析哪个指标（如交易量、链上交易量、活跃用户）"
                "以及哪个时间范围（如过去7天、上周）？",
                "Which metric (e.g. trading volume, on-chain volume, active users) "
                "and which time range (e.g. last 7 days, last week) should I analyze?",
            )
        elif metric is None:
            clarification = tr(
                language,
                "请问您想分析哪个指标？目前支持交易量、链上交易量、"
                "活跃用户、手续费收入等。",
                "Which metric should I analyze? Available options include trading "
                "volume, on-chain volume, active users, and fee revenue.",
            )
        elif time_key is None:
            clarification = tr(
                language,
                "请问您想看哪个时间范围的数据？（如过去7天、上周、过去30天）",
                "Which time range should I use? "
                "(e.g. last 7 days, last week, last 30 days)",
            )

        payload = {
            "metric": metric.key if metric else None,
            "time_range": time_key,
            "action": action,
            "clarification": clarification,
        }
        return json.dumps(payload, ensure_ascii=False)

    def _insight(self, context: dict[str, Any]) -> str:
        label = context.get("metric_label", "指标")
        window = context.get("window_label", "统计窗口")
        current = context.get("current_value", "N/A")
        previous = context.get("previous_value", "N/A")
        change = context.get("change_pct", "N/A")
        top_driver = context.get("top_driver") or {}
        positive = context.get("positive_driver") or {}
        peak = context.get("peak") or {}
        language = context.get("language", "zh")

        if language == "en":
            lines = ["**Conclusion**", ""]
            lines.append(
                f"{window} {label}: **{current}**, versus {previous} "
                f"in the previous period, a change of **{change}**."
            )
            if top_driver:
                lines.append(
                    f"The largest drag came from **{top_driver.get('name', 'unknown')}** "
                    f"(change {top_driver.get('change', 'N/A')}), "
                    "the main factor to explain."
                )
            else:
                lines.append(
                    "The move was spread across dimensions; no single extreme "
                    "drag factor was identified."
                )

            lines += ["", "**Evidence**", ""]
            lines.append(
                f"- Current window: {window}; metric: {label}; aggregate {current}."
            )
            lines.append(f"- Change: {previous} → {current}, magnitude {change}.")
            if top_driver:
                lines.append(
                    f"- Largest negative contributor: {top_driver.get('name')}, "
                    f"change {top_driver.get('change')} "
                    f"(share {top_driver.get('share', 'N/A')})."
                )
            if positive:
                lines.append(
                    f"- Positive support: {positive.get('name')}, "
                    f"change {positive.get('change')}."
                )
            if peak:
                lines.append(
                    f"- Peak in the window: {peak.get('name', 'N/A')}, "
                    f"value {peak.get('value', 'N/A')}."
                )
            lines.append(
                "- Source: aggregated results from the local DuckDB read-only sandbox."
            )

            lines += ["", "**Recommendations**", ""]
            if top_driver:
                lines.append(
                    f"1. Investigate {top_driver.get('name')}: review market depth, "
                    "gas/bridge congestion, and campaign scheduling."
                )
            else:
                lines.append(
                    "1. Set up dimensional monitoring for this metric, starting "
                    "with asset and chain type."
                )
            lines.append(
                "2. Add a 7-day moving-average alert and trigger attribution when "
                "the change falls below -10%."
            )
            lines.append(
                "3. Record this attribution in the metric dashboard notes for "
                "operations and market-making review."
            )
            lines.append("")
            lines.append(
                "> Generated locally by MockLLM; no external API call, zero cost."
            )
            return "\n".join(lines)

        lines = ["**结论**", ""]
        lines.append(
            f"{window} {label}为 **{current}**，上一周期为 {previous}，"
            f"环比 **{change}**。"
        )
        if top_driver:
            lines.append(
                f"主要拖累来自 **{top_driver.get('name', '未知维度')}**"
                f"（变化 {top_driver.get('change', 'N/A')}），"
                f"是本次波动最需要解释的因子。"
            )
        else:
            lines.append("整体波动由多维度共同贡献，未发现单一极端拖累项。")

        lines += ["", "**数据支撑**", ""]
        lines.append(f"- 当前窗口：{window}，指标：{label}，聚合值 {current}。")
        lines.append(f"- 环比变化：{previous} → {current}，幅度 {change}。")
        if top_driver:
            lines.append(
                f"- 最大负向贡献维度：{top_driver.get('name')}，"
                f"变化 {top_driver.get('change')}"
                f"（占比 {top_driver.get('share', 'N/A')}）。"
            )
        if positive:
            lines.append(
                f"- 正向支撑维度：{positive.get('name')}，"
                f"变化 {positive.get('change')}。"
            )
        if peak:
            lines.append(
                f"- 区间峰值出现在 {peak.get('name', 'N/A')}，"
                f"数值 {peak.get('value', 'N/A')}。"
            )
        lines.append("- 数据来源：本地 DuckDB 只读沙箱聚合结果，未上传原始明细。")

        lines += ["", "**业务建议**", ""]
        if top_driver:
            lines.append(
                f"1. 针对 {top_driver.get('name')} 做专项排查：核对做市深度、"
                "链上 Gas/跨链桥拥堵与活动排期。"
            )
        else:
            lines.append("1. 建立该指标的分维度日常监控，优先覆盖币种与链类型。")
        lines.append("2. 对下滑维度设置 7 日移动均值告警，环比跌破 -10% 时自动触发归因。")
        lines.append("3. 将本次归因结论沉淀为指标看板注释，供运营与做市团队复盘。")
        lines.append("")
        lines.append("> 本段由 MockLLM 本地生成，未调用真实 API，不产生费用。")
        return "\n".join(lines)

    def _codegen(self, context: dict[str, Any]) -> str:
        sql = context.get("fallback_sql")
        if not sql:
            raise ValueError("MockLLM 缺少 fallback_sql，无法生成 SQL")
        return str(sql)
