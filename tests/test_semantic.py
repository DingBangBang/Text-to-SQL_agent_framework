"""Semantic layer and governed SQL tests."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.semantic import SemanticRegistry, TimeRangeError

TODAY = date(2026, 10, 20)


def test_metric_synonyms_resolve_to_governed_keys(registry: SemanticRegistry):
    assert registry.find_metric("过去7天链上交易量是多少？").key == "onchain_volume"
    assert registry.find_metric("上周成交量怎么样").key == "trade_volume"
    assert registry.find_metric("活跃地址有多少").key == "active_users"
    assert registry.find_metric("手续费收入").key == "fee_revenue"
    assert registry.find_metric("今天天气如何") is None


def test_previous_week_window_is_monday_to_sunday(registry: SemanticRegistry):
    window = registry.resolve_time_range("上周", TODAY)
    assert window is not None
    assert window.start == date(2026, 10, 12)
    assert window.end == date(2026, 10, 18)
    assert window.days == 7


def test_trailing_window_includes_today(registry: SemanticRegistry):
    window = registry.resolve_time_range("过去7天", TODAY)
    assert window.start == date(2026, 10, 14)
    assert window.end == TODAY


def test_unknown_time_range_returns_none(registry: SemanticRegistry):
    assert registry.resolve_time_range("石器时代", TODAY) is None


def test_metric_sql_keeps_filters_and_expression(registry: SemanticRegistry):
    metric = registry.get("onchain_volume")
    window = registry.resolve_time_range("last_7_days", TODAY)
    sql = registry.build_metric_sql(metric, window, dimension="symbol")

    assert "SUM(t.trade_volume_usd) AS onchain_volume_usd" in sql
    assert "t.chain_type = 'onchain'" in sql
    assert "DATE '2026-10-14'" in sql
    assert "GROUP BY 1" in sql


def test_attribution_sql_compares_equal_windows(registry: SemanticRegistry):
    metric = registry.get("trade_volume")
    window = registry.resolve_time_range("last_7_days", TODAY)
    sql = registry.build_attribution_sql(metric, window, "symbol")

    assert "current_period" in sql and "previous_period" in sql
    assert "DATE '2026-10-07'" in sql  # previous window end
    assert "change_pct" in sql
    assert "FULL OUTER JOIN" in sql


def test_previous_window_has_same_length(registry: SemanticRegistry):
    window = registry.resolve_time_range("last_30_days", TODAY)
    previous = registry.previous_window(window)
    assert previous.days == window.days
    assert previous.end == window.start - timedelta(days=1)


def test_attribution_dimensions_skip_pinned_filters(registry: SemanticRegistry):
    assert registry.attribution_dimensions(registry.get("onchain_volume")) == ["symbol"]
    assert registry.attribution_dimensions(registry.get("trade_volume")) == [
        "symbol",
        "chain_type",
    ]


def test_bad_mode_in_yaml_raises():
    broken = SemanticRegistry(
        {
            "metrics": {"x": {"expression": "COUNT(*)"}},
            "time_ranges": {"weird": {"mode": "lunar"}},
        }
    )
    with pytest.raises(TimeRangeError):
        broken.resolve_time_range("weird", TODAY)
