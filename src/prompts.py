"""Prompt templates. Kept small on purpose — every token costs money."""

from __future__ import annotations

from .semantic import Metric, SemanticRegistry, TimeWindow

ROUTER_SYSTEM = """你是「链上数据分析 Agent」的意图路由器。只输出 JSON，不要解释。
可用指标（key | 名称 | 口径 | 同义词）：
{catalog}

可用时间窗口（key | 名称）：
{time_ranges}

输出格式：
{{"metric": "指标key或null", "time_range": "窗口key或null",
  "action": "query或attribution", "clarification": "信息不足时的一句中文反问，否则空字符串"}}

规则：
1. 问“为什么/原因/归因/拆解/谁拖累/跌了/涨了”时 action=attribution，否则 query。
2. metric 或 time_range 无法从问题中确定时填 null，并在 clarification 中只追问缺失项。
3. 只能使用上面列出的 key，不得编造指标或时间窗口。
4. 只输出一个 JSON 对象。"""

ROUTER_SYSTEM_EN = """You are the intent router for an on-chain analytics agent. Output JSON only, no prose.
Available metrics (key | name | definition | synonyms):
{catalog}

Available time windows (key | name):
{time_ranges}

Output schema:
{{"metric": "metric key or null", "time_range": "window key or null",
  "action": "query or attribution",
  "clarification": "one English question when information is missing, otherwise an empty string"}}

Rules:
1. Use action=attribution for why/explain/driver/breakdown/which asset dragged/drop/increase questions; otherwise query.
2. If metric or time_range cannot be determined, set it to null and ask only for the missing field.
3. Use only the keys listed above; never invent metrics or time windows.
4. Output exactly one JSON object."""

CODEGEN_SYSTEM = """你是资深 DuckDB 分析工程师。根据指标契约和表结构生成一条只读 SQL。
硬性要求：
1. 只输出 SQL，不要 Markdown 代码块，不要解释。
2. 必须原样使用指标契约中的度量表达式与固定过滤条件，不得改写业务口径。
3. 必须用给定日期闭区间过滤时间字段。
4. 只允许 SELECT / WITH，只能查询给定表。
5. 按日期升序或指标值降序排序，最多返回 500 行。"""

CODEGEN_USER = """用户问题：{question}

指标契约：
{metric_reference}

时间窗口：{window_label}（{start} 至 {end}，闭区间）

表结构：
{schema}

输出要求：{output_requirement}
{repair_block}"""

REPAIR_BLOCK = """
上一次 SQL 执行失败，请修复后重新输出完整 SQL。
上一次 SQL：
{previous_sql}

数据库报错：
{error}
"""

CODEGEN_SYSTEM_EN = """You are a senior DuckDB analytics engineer. Generate one read-only SQL statement from the metric contract and schema.
Hard requirements:
1. Output SQL only; no Markdown fences, no explanation.
2. Use the metric contract's measure expression and fixed filters verbatim; never change the business definition.
3. Filter the time column using the given closed date range.
4. Only SELECT / WITH is allowed, and only the given tables may be queried.
5. Sort by date ascending or metric value descending; return at most 500 rows."""

CODEGEN_USER_EN = """User question: {question}

Metric contract:
{metric_reference}

Time window: {window_label} ({start} to {end}, inclusive)

Schema:
{schema}

Output requirement: {output_requirement}
{repair_block}"""

REPAIR_BLOCK_EN = """
The previous SQL failed. Fix it and return the complete SQL again.
Previous SQL:
{previous_sql}

Database error:
{error}
"""

INSIGHT_SYSTEM = """你是链上业务分析师。只使用给定数据写中文分析，不得编造数字。
严格按以下结构输出 Markdown：
**结论**
1-2 句话，结论前置，包含关键变化幅度。

**数据支撑**
用 2-4 个要点引用具体数值（当前值、对比值、环比），并说明主要拖累/驱动维度。

**业务建议**
2-3 条可执行建议，分别对应数据中发现的驱动因素。

不要输出与数据无关的套话，不要重复原始表格。"""

INSIGHT_USER = """用户问题：{question}
指标：{metric_label}（{metric_key}）
时间窗口：{window_label}（{start} 至 {end}）
当前值：{current_value}
上一周期：{previous_value}
环比：{change_pct}

按日/维度数据：
{data_table}

归因结果：
{attribution_table}
"""

INSIGHT_SYSTEM_EN = """You are an on-chain business analyst. Use only the supplied data; never invent numbers.
Output Markdown in exactly this structure:
**Conclusion**
1-2 sentences, conclusion first, including the key magnitude of change.

**Evidence**
2-4 bullets citing concrete values (current, comparison, change %) and naming the largest drag/driver dimension.

**Recommendations**
2-3 actionable recommendations mapped to the drivers found in the data.

No generic filler and do not repeat the raw tables."""

INSIGHT_USER_EN = """User question: {question}
Metric: {metric_label} ({metric_key})
Time window: {window_label} ({start} to {end})
Current value: {current_value}
Previous period: {previous_value}
Change: {change_pct}

Daily / dimensional data:
{data_table}

Attribution:
{attribution_table}
"""


def router_messages(
    question: str, registry: SemanticRegistry, language: str = "zh"
) -> list[dict[str, str]]:
    english = language == "en"
    return [
        {
            "role": "system",
            "content": (ROUTER_SYSTEM_EN if english else ROUTER_SYSTEM).format(
                catalog=registry.catalog_text("en" if english else "zh"),
                time_ranges=registry.time_range_catalog_text(
                    "en" if english else "zh"
                ),
            ),
        },
        {"role": "user", "content": question},
    ]


def codegen_messages(
    question: str,
    metric: Metric,
    window: TimeWindow,
    schema: str,
    dimension: str,
    registry: SemanticRegistry,
    previous_sql: str | None = None,
    error: str | None = None,
    language: str = "zh",
) -> list[dict[str, str]]:
    english = language == "en"
    if dimension == "trade_date":
        if english:
            requirement = (
                f"Aggregate by {metric.time_column} per day, output columns "
                f"{dimension} and {metric.value_column}, ordered by date ascending."
            )
        else:
            requirement = (
                f"按 {metric.time_column} 逐日聚合，输出列 {dimension} 与 "
                f"{metric.value_column}，按日期升序。"
            )
    else:
        if english:
            requirement = (
                f"Aggregate by dimension {dimension}, output columns {dimension} "
                f"and {metric.value_column}, ordered by metric value descending, "
                "at most 20 rows."
            )
        else:
            requirement = (
                f"按维度 {dimension} 聚合，输出列 {dimension} 与 "
                f"{metric.value_column}，按指标值降序，最多 20 行。"
            )
    repair = ""
    if previous_sql and error:
        repair = (REPAIR_BLOCK_EN if english else REPAIR_BLOCK).format(
            previous_sql=previous_sql, error=error
        )
    system_prompt = CODEGEN_SYSTEM_EN if english else CODEGEN_SYSTEM
    user_template = CODEGEN_USER_EN if english else CODEGEN_USER
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_template.format(
                question=question,
                metric_reference=registry.sql_reference(
                    metric, "en" if english else "zh"
                ),
                window_label=window.label,
                start=window.start.isoformat(),
                end=window.end.isoformat(),
                schema=schema,
                output_requirement=requirement,
                repair_block=repair,
            ),
        },
    ]


def insight_messages(
    question: str,
    metric: Metric,
    window: TimeWindow,
    data_table: str,
    attribution_table: str,
    current_value: str,
    previous_value: str,
    change_pct: str,
    language: str = "zh",
) -> list[dict[str, str]]:
    english = language == "en"
    system_prompt = INSIGHT_SYSTEM_EN if english else INSIGHT_SYSTEM
    user_template = INSIGHT_USER_EN if english else INSIGHT_USER
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_template.format(
                question=question,
                metric_label=metric.display_label("en" if english else "zh"),
                metric_key=metric.key,
                window_label=window.label,
                start=window.start.isoformat(),
                end=window.end.isoformat(),
                current_value=current_value,
                previous_value=previous_value,
                change_pct=change_pct,
                data_table=data_table,
                attribution_table=attribution_table,
            ),
        },
    ]
