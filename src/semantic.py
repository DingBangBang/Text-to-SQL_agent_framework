"""Semantic layer: metrics.yaml -> governed SQL fragments.

Every natural-language metric must resolve to a definition in ``metrics.yaml``.
SQL is assembled from vetted expressions and a controlled time window, so the
LLM can never invent a table, a column, or a denominator on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Dimension:
    key: str
    label: str
    expr: str
    type: str = "category"


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    description: str
    table: str
    alias: str
    time_column: str
    expression: str
    value_column: str
    filters: tuple[str, ...] = ()
    unit: str = ""
    aggregation: str = "sum"
    synonyms: tuple[str, ...] = ()
    dimensions: dict[str, Dimension] = field(default_factory=dict)

    @property
    def time_expr(self) -> str:
        return f"{self.alias}.{self.time_column}"


@dataclass(frozen=True)
class TimeWindow:
    key: str
    label: str
    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def sql_filter(self, metric: Metric) -> str:
        return (
            f"{metric.time_expr} BETWEEN DATE '{self.start.isoformat()}' "
            f"AND DATE '{self.end.isoformat()}'"
        )


TIME_ALIASES: dict[str, str] = {
    "今天": "today",
    "今日": "today",
    "today": "today",
    "昨天": "yesterday",
    "昨日": "yesterday",
    "yesterday": "yesterday",
    "过去7天": "last_7_days",
    "最近7天": "last_7_days",
    "近7天": "last_7_days",
    "本周": "last_7_days",
    "last 7 days": "last_7_days",
    "last_7_days": "last_7_days",
    "过去14天": "last_14_days",
    "最近14天": "last_14_days",
    "近14天": "last_14_days",
    "last 14 days": "last_14_days",
    "last_14_days": "last_14_days",
    "过去30天": "last_30_days",
    "最近30天": "last_30_days",
    "近30天": "last_30_days",
    "过去一个月": "last_30_days",
    "近一个月": "last_30_days",
    "last 30 days": "last_30_days",
    "last_30_days": "last_30_days",
    "上周": "last_week",
    "上一周": "last_week",
    "last week": "last_week",
    "last_week": "last_week",
    "本月": "this_month",
    "这个月": "this_month",
    "this month": "this_month",
    "this_month": "this_month",
    "上月": "last_month",
    "上个月": "last_month",
    "last month": "last_month",
    "last_month": "last_month",
}



class SemanticRegistry:
    """Loaded metrics, dimensions and time-window logic."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.raw = config
        self.tables: dict[str, Any] = config.get("tables", {}) or {}
        self.time_range_specs: dict[str, Any] = config.get("time_ranges", {}) or {}
        self.default_time_range: str = config.get("default_time_range", "last_7_days")
        self.metrics: dict[str, Metric] = {}
        for key, item in (config.get("metrics") or {}).items():
            dims = {
                dkey: Dimension(
                    key=dkey,
                    label=dval.get("label", dkey),
                    expr=dval.get("expr", f"t.{dkey}"),
                    type=dval.get("type", "category"),
                )
                for dkey, dval in (item.get("dimensions") or {}).items()
            }
            self.metrics[key] = Metric(
                key=key,
                label=item.get("label", key),
                description=item.get("description", ""),
                table=item.get("table", "trades"),
                alias=item.get("alias", "t"),
                time_column=item.get("time_column", "trade_date"),
                expression=item["expression"],
                value_column=item.get("value_column", key),
                filters=tuple(item.get("filters") or ()),
                unit=item.get("unit", ""),
                aggregation=item.get("aggregation", "sum"),
                synonyms=tuple(item.get("synonyms") or ()),
                dimensions=dims,
            )

    @classmethod
    def load(cls, path: Path | str) -> "SemanticRegistry":
        with Path(path).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if not config.get("metrics"):
            raise ValueError(f"{path} 中未定义任何指标")
        return cls(config)

    # -- discovery --------------------------------------------------------
    def get(self, key: str) -> Metric:
        try:
            return self.metrics[key]
        except KeyError as exc:
            raise MetricNotFoundError(f"未知指标：{key}") from exc

    def find_metric(self, text: str) -> Metric | None:
        """Longest-synonym substring match, deterministic and explainable."""
        if not text:
            return None
        haystack = text.lower().replace(" ", "")
        best: tuple[int, Metric] | None = None
        for metric in self.metrics.values():
            candidates = {metric.key.lower(), metric.label.lower(), *metric.synonyms}
            for candidate in candidates:
                needle = candidate.lower().replace(" ", "")
                if needle and needle in haystack:
                    if best is None or len(needle) > best[0]:
                        best = (len(needle), metric)
        return best[1] if best else None

    def catalog_text(self) -> str:
        lines = []
        for metric in self.metrics.values():
            lines.append(
                f"- {metric.key} | {metric.label} | {metric.description} "
                f"| 单位: {metric.unit or '无'} | 同义词: {', '.join(metric.synonyms)}"
            )
        return "\n".join(lines)

    def time_range_catalog_text(self) -> str:
        return "\n".join(
            f"- {key} | {spec.get('label', key)}"
            for key, spec in self.time_range_specs.items()
        )

    # -- time windows -----------------------------------------------------
    def resolve_time_range(
        self, value: str | None, today: date | None = None
    ) -> TimeWindow | None:
        if not value:
            return None
        today = today or date.today()
        normalized = str(value).strip().lower()
        key = TIME_ALIASES.get(normalized, normalized.replace(" ", "_"))
        spec = self.time_range_specs.get(key)
        if not spec:
            return None
        mode = spec.get("mode", "trailing")
        label = spec.get("label", key)

        if mode == "trailing":
            days = int(spec.get("days", 7))
            return TimeWindow(key, label, today - timedelta(days=days - 1), today)
        if mode == "yesterday":
            day = today - timedelta(days=1)
            return TimeWindow(key, label, day, day)
        if mode == "previous_week":
            end = today - timedelta(days=today.weekday() + 1)
            return TimeWindow(key, label, end - timedelta(days=6), end)
        if mode == "current_month":
            return TimeWindow(key, label, today.replace(day=1), today)
        if mode == "previous_month":
            first_this = today.replace(day=1)
            last_prev = first_this - timedelta(days=1)
            return TimeWindow(key, label, last_prev.replace(day=1), last_prev)
        raise TimeRangeError(f"不支持的时间窗口模式：{mode}")

    @staticmethod
    def previous_window(window: TimeWindow) -> TimeWindow:
        prev_end = window.start - timedelta(days=1)
        prev_start = prev_end - timedelta(days=window.days - 1)
        return TimeWindow(
            key=f"{window.key}_prev",
            label=f"上一周期（{prev_start} ~ {prev_end}）",
            start=prev_start,
            end=prev_end,
        )

    # -- governed SQL builders -------------------------------------------
    def _where(self, metric: Metric, window: TimeWindow) -> str:
        clauses = [window.sql_filter(metric), *metric.filters]
        return " AND ".join(clauses)

    def build_metric_sql(
        self,
        metric: Metric,
        window: TimeWindow,
        dimension: str | None = "trade_date",
        limit: int = 500,
    ) -> str:
        """Aggregate a metric over a window, optionally split by one dimension.

        ``dimension=None`` returns a single-row scalar result.  Grouping by
        ``trade_date`` returns a daily series, which is what the dashboard
        charts and what attribution compares against.
        """
        if dimension is None:
            return (
                f"SELECT {metric.expression} AS {metric.value_column}\n"
                f"FROM {metric.table} {metric.alias}\n"
                f"WHERE {self._where(metric, window)}"
            )
        dim = metric.dimensions.get(dimension)
        if dim is None:
            raise MetricNotFoundError(
                f"指标 {metric.key} 不支持维度 {dimension}"
            )
        if dim.type == "time":
            ordering = "ORDER BY 1"
        else:
            ordering = f"ORDER BY 2 DESC"
        return (
            f"SELECT {dim.expr} AS {dim.key},\n"
            f"       {metric.expression} AS {metric.value_column}\n"
            f"FROM {metric.table} {metric.alias}\n"
            f"WHERE {self._where(metric, window)}\n"
            f"GROUP BY 1\n{ordering}\nLIMIT {int(limit)}"
        )

    def build_total_sql(self, metric: Metric, window: TimeWindow) -> str:
        return self.build_metric_sql(metric, window, dimension=None)

    def build_attribution_sql(
        self,
        metric: Metric,
        window: TimeWindow,
        dimension: str,
        limit: int = 20,
    ) -> str:
        """Compare current vs previous equal-length window along one dimension."""
        dim = metric.dimensions.get(dimension)
        if dim is None:
            raise MetricNotFoundError(f"指标 {metric.key} 不支持维度 {dimension}")
        previous = self.previous_window(window)
        current_where = self._where(metric, window)
        previous_where = self._where(metric, previous)
        return f"""WITH current_period AS (
    SELECT {dim.expr} AS dimension_value,
           {metric.expression} AS current_value
    FROM {metric.table} {metric.alias}
    WHERE {current_where}
    GROUP BY 1
),
previous_period AS (
    SELECT {dim.expr} AS dimension_value,
           {metric.expression} AS previous_value
    FROM {metric.table} {metric.alias}
    WHERE {previous_where}
    GROUP BY 1
)
SELECT COALESCE(c.dimension_value, p.dimension_value) AS dimension_value,
       COALESCE(c.current_value, 0) AS current_value,
       COALESCE(p.previous_value, 0) AS previous_value,
       COALESCE(c.current_value, 0) - COALESCE(p.previous_value, 0) AS change_value,
       CASE WHEN COALESCE(p.previous_value, 0) = 0 THEN NULL
            ELSE ROUND((COALESCE(c.current_value, 0) / p.previous_value - 1) * 100, 2)
       END AS change_pct
FROM current_period c
FULL OUTER JOIN previous_period p
       ON c.dimension_value IS NOT DISTINCT FROM p.dimension_value
ORDER BY change_value ASC NULLS LAST
LIMIT {int(limit)}"""

    def attribution_dimensions(self, metric: Metric) -> list[str]:
        """Category dimensions that are not already pinned by metric filters.

        E.g. ``onchain_volume`` fixes ``chain_type = 'onchain'``, so breaking it
        down by ``chain_type`` would produce a single, uninformative row.
        """
        result: list[str] = []
        for key, dim in metric.dimensions.items():
            if dim.type == "time":
                continue
            pinned = any(
                re.search(rf"{re.escape(key)}\b\s*=\s*", clause, flags=re.IGNORECASE)
                for clause in metric.filters
            )
            if not pinned:
                result.append(key)
        return result

    def sql_reference(self, metric: Metric) -> str:
        """Human-readable contract handed to the SQL generator."""
        filters = " AND ".join(metric.filters) if metric.filters else "无"
        dims = ", ".join(
            f"{dim.key} ({dim.expr})" for dim in metric.dimensions.values()
        )
        return (
            f"指标键: {metric.key}\n"
            f"业务口径: {metric.description}\n"
            f"度量表达式: {metric.expression}\n"
            f"来源表: {metric.table} {metric.alias}\n"
            f"时间字段: {metric.time_expr}\n"
            f"固定过滤: {filters}\n"
            f"可用维度: {dims}\n"
            f"输出值列名: {metric.value_column}（单位 {metric.unit or '无'}）"
        )



class MetricNotFoundError(KeyError):
    pass


class TimeRangeError(ValueError):
    pass
