"""End-to-end graph tests with MockLLM — no network, no API cost."""

from __future__ import annotations

from src.graph import run_agent
from src.mock_llm import MockLLM

from conftest import TEST_TODAY


def _run(question: str, settings, scripted=None, language="zh"):
    llm = MockLLM(settings, scripted=scripted)
    return run_agent(
        question,
        settings=settings,
        llm=llm,
        today=TEST_TODAY,
        language=language,
    )


def test_query_flow_router_to_insight(settings):
    state, meta = _run("过去7天链上交易量是多少？", settings)

    assert state["status"] == "completed"
    assert state["route"]["metric"] == "onchain_volume"
    assert state["route"]["action"] == "query"
    assert state["retry_count"] == 0
    assert "chain_type = 'onchain'" in state["sql"]
    assert state["row_count"] > 0
    assert meta["cost_cny"] == 0.0
    assert meta["mock"] is True

    assert "**结论**" in state["insight"]
    assert "**数据支撑**" in state["insight"]
    assert "**业务建议**" in state["insight"]

    nodes = [step["node"] for step in state["trace"]]
    assert nodes == ["Router", "Semantic", "CodeGen", "Execution", "Insight"]


def test_attribution_flow_decomposes_dimensions(settings):
    state, _ = _run("为什么上周链上交易量跌了？", settings)

    assert state["status"] == "completed"
    assert state["action"] == "attribution"
    assert state["route"]["time_range"] == "last_week"
    assert state["sql_attempts"] == 1

    attribution = state["attribution"]
    # chain_type is pinned to 'onchain' by the metric definition, so only the
    # informative symbol split is produced.
    assert set(attribution["breakdowns"]) == {"symbol"}
    assert attribution["current_total"] > 0
    assert attribution["previous_total"] > 0
    assert attribution["change_pct"] is not None

    for frame in attribution["breakdowns"].values():
        assert list(frame.columns) == [
            "dimension_value",
            "current_value",
            "previous_value",
            "change_value",
            "change_pct",
        ]
        assert not frame.empty

    nodes = [step["node"] for step in state["trace"]]
    assert nodes == [
        "Router",
        "Semantic",
        "CodeGen",
        "Execution",
        "Attribution",
        "Insight",
    ]


def test_missing_metric_triggers_clarification(settings):
    state, meta = _run("帮我看一下", settings)

    assert state["needs_clarification"] is True
    assert state["status"] == "needs_clarification"
    assert state["route"]["metric"] is None
    assert "指标" in state["clarification"]
    assert "insight" not in state
    assert meta["input_tokens"] > 0


def test_missing_time_range_triggers_clarification(settings):
    state, _ = _run("交易量是多少", settings)

    assert state["needs_clarification"] is True
    assert state["route"]["metric"] == "trade_volume"
    assert "时间" in state["clarification"]


def test_sql_error_is_repaired_once(settings):
    state, _ = _run(
        "过去7天链上交易量是多少？",
        settings,
        scripted={"codegen": ["SELECT * FROM table_that_does_not_exist"]},
    )

    assert state["status"] == "completed"
    assert state["sql_attempts"] == 2
    assert state["retry_count"] == 1
    assert state["sql_error"] is None
    assert state["row_count"] > 0


def test_retry_budget_stops_after_three_self_corrections(settings):
    state, _ = _run(
        "过去7天链上交易量是多少？",
        settings,
        scripted={
            "codegen": [
                "SELECT * FROM missing_table_1",
                "SELECT * FROM missing_table_2",
                "SELECT * FROM missing_table_3",
                "SELECT * FROM missing_table_4",
            ]
        },
    )

    assert state["status"] == "failed"
    assert state["sql_attempts"] == settings.max_sql_retries + 1
    assert state["retry_count"] == settings.max_sql_retries
    assert "执行失败，请人工介入" in state["insight"]
    nodes = [step["node"] for step in state["trace"]]
    assert nodes[-1] == "Failure"


def test_bad_router_json_falls_back_without_extra_api_call(settings):
    llm = MockLLM(settings, scripted={"router": ["这不是 JSON"]})
    state, meta = run_agent(
        "过去7天交易量是多少？",
        settings=settings,
        llm=llm,
        today=TEST_TODAY,
    )

    assert state["status"] == "completed"
    assert state["route"]["source"] == "fallback-rules"
    assert state["route"]["metric"] == "trade_volume"
    tasks = [call["task"] for call in meta["calls"]]
    assert tasks == ["router", "codegen", "insight"]  # no router retry


def test_english_attribution_pipeline_is_fully_english(settings):
    state, meta = _run(
        "Why did on-chain volume drop last week?",
        settings,
        language="en",
    )

    assert state["status"] == "completed"
    assert state["route"]["metric"] == "onchain_volume"
    assert state["route"]["action"] == "attribution"
    assert state["route"]["time_range"] == "last_week"

    assert "**Conclusion**" in state["insight"]
    assert "**Evidence**" in state["insight"]
    assert "**Recommendations**" in state["insight"]
    assert "结论" not in state["insight"]
    assert "asset=" in state["insight"]

    trace_text = " ".join(step["detail"] for step in state["trace"])
    assert "SQL attempt 1/4" in trace_text
    assert "Executed successfully" in trace_text
    assert "breakdown" in trace_text
    assert "指标" not in trace_text
    assert meta["mock"] is True


def test_english_clarification_is_in_english(settings):
    state, _ = _run("Give me a summary", settings, language="en")

    assert state["needs_clarification"] is True
    assert "Which metric" in state["clarification"]
    assert "请问" not in state["clarification"]


def test_english_query_pipeline(settings):
    state, _ = _run(
        "How many active users were there last week?",
        settings,
        language="en",
    )

    assert state["status"] == "completed"
    assert state["route"]["metric"] == "active_users"
    assert state["route"]["action"] == "query"
    assert state["semantic"]["unit"] == "users"
    assert "**Conclusion**" in state["insight"]
