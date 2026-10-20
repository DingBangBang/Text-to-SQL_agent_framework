"""Streamlit dashboard: chat UI with step-by-step agent transparency."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.db import schema_snapshot  # noqa: E402
from src.graph import build_graph, run_agent  # noqa: E402
from src.reporting import (  # noqa: E402
    build_insight_context,
    format_pct,
)
from src.semantic import SemanticRegistry  # noqa: E402

st.set_page_config(
    page_title="Text-to-SQL 业务分析 Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1180px; }
    .step-tag { display:inline-block; background:#dbeafe; color:#1d4ed8;
                border-radius:7px; padding:1px 8px; font-size:12px;
                font-weight:600; margin-right:7px; }
    .muted { color:#6b7280; font-size:12.5px; }
    div[data-testid="stMetricValue"] { font-size: 22px; }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = Settings.from_env()
registry = SemanticRegistry.load(settings.metrics_path)


@st.cache_resource(show_spinner="加载 Agent 状态机…")
def load_graph(mode: str, _db_mtime: float):
    runtime = replace(settings, llm_mode=mode)
    return build_graph(runtime)


@st.cache_data(show_spinner=False)
def load_schema(_db_mtime: float) -> dict:
    return schema_snapshot(settings.db_path)


def chart_for(df: pd.DataFrame | None, unit: str = "") -> None:
    if df is None or df.empty:
        st.info("本次没有可展示的数据。")
        return
    x_column = df.columns[0]
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        st.dataframe(df, use_container_width=True)
        return
    y_column = numeric[-1]
    if pd.api.types.is_datetime64_any_dtype(df[x_column]) or "date" in str(x_column):
        figure = px.line(
            df,
            x=x_column,
            y=y_column,
            markers=True,
            labels={y_column: f"{y_column} ({unit})" if unit else y_column},
        )
    else:
        figure = px.bar(
            df.head(15),
            x=x_column,
            y=y_column,
            labels={y_column: f"{y_column} ({unit})" if unit else y_column},
        )
    figure.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        font_family="-apple-system, PingFang SC, sans-serif",
    )
    st.plotly_chart(figure, use_container_width=True)



def render_run(question: str, state: dict, meta: dict) -> None:
    st.markdown(f"**问题：** {question}")
    if state.get("needs_clarification"):
        st.warning(state.get("clarification", "信息不足，请补充指标或时间范围。"))
        return
    if state.get("status") == "failed":
        st.error(state.get("insight", "执行失败，请人工介入。"))
        return

    attribution = state.get("attribution") or {}
    semantic = state.get("semantic") or {}
    unit = semantic.get("unit", "")
    payload = build_insight_context(state, unit)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("当前值", payload["current_value"])
    col2.metric("上一周期", payload["previous_value"])
    col3.metric("环比", format_pct(attribution.get("change_pct")))
    col4.metric(
        "Token / 成本",
        f"{meta['input_tokens'] + meta['output_tokens']:,}",
        f"¥{meta['cost_cny']:.4f}",
    )

    with st.expander("① 思考过程 · 意图路由与执行轨迹", expanded=True):
        route = state.get("route") or {}
        st.markdown(
            f"<span class='step-tag'>Router</span> 指标 "
            f"<b>{route.get('metric')}</b> ｜ 时间窗口 "
            f"<b>{route.get('time_range')}</b> ｜ 动作 "
            f"<b>{route.get('action')}</b> ｜ 来源 "
            f"<b>{route.get('source')}</b>",
            unsafe_allow_html=True,
        )
        for step in state.get("trace", []):
            st.markdown(
                f"<span class='step-tag'>{step.get('node')}</span>"
                f"<span class='muted'>{step.get('detail')}</span>",
                unsafe_allow_html=True,
            )

    with st.expander("② 生成的 DuckDB SQL", expanded=True):
        st.code(state.get("sql", ""), language="sql")
        st.caption(
            f"SQL 来源：{state.get('sql_source')} ｜ "
            f"尝试次数：{state.get('sql_attempts')}"
        )

    with st.expander("③ 执行数据（本地只读沙箱）", expanded=True):
        df = state.get("result_df")
        chart_for(df, unit)
        if df is not None:
            st.dataframe(df, use_container_width=True, height=240)
        st.caption(
            f"执行 {state.get('exec_ms', 0)} ms ｜ "
            f"返回 {state.get('row_count', 0)} 行 ｜ 原始明细未离开本机"
        )

    if attribution.get("breakdowns"):
        with st.expander("④ 多维归因拆解", expanded=False):
            for dimension, frame in attribution["breakdowns"].items():
                st.markdown(f"**按 {dimension} 拆解**")
                st.dataframe(frame, use_container_width=True, height=220)

    with st.expander("⑤ 智能洞察报告", expanded=True):
        st.markdown(state.get("insight", ""))
        if meta.get("mock"):
            st.caption("MockLLM 本地生成，未调用外部 API，成本为 0。")


# ---------------------------------------------------------------- sidebar
db_ready = settings.db_path.exists()
with st.sidebar:
    st.title("📊 Analytics Agent")
    st.caption("本地 DuckDB · LangGraph · 成本受控")

    mode = st.radio(
        "LLM 模式",
        options=["mock", "deepseek"],
        index=0 if settings.llm_mode != "deepseek" else 1,
        help="mock：本地规则，零成本；deepseek：真实 API，受 Token 预算保护。",
    )
    if mode == "deepseek" and not settings.api_key:
        st.error("未配置 DEEPSEEK_API_KEY，已自动使用 MockLLM。")
        mode = "mock"

    st.divider()
    st.subheader("数据库 Schema")
    if not db_ready:
        st.error("未找到 data/analytics.duckdb\n请运行 `python simulate_data.py`")
    else:
        schema = load_schema(settings.db_path.stat().st_mtime)
        for table, meta in schema.items():
            with st.expander(f"{table} · {meta['rows']:,} 行", expanded=True):
                st.dataframe(
                    pd.DataFrame(meta["columns"]), hide_index=True, height=170
                )

    st.divider()
    st.subheader("指标口径")
    st.markdown(
        "\n".join(
            f"- **{metric.label}**：{metric.description}"
            for metric in registry.metrics.values()
        )
    )

    st.divider()
    st.subheader("成本护栏")
    st.markdown(
        f"- 单次输入上限：**{settings.max_input_tokens:,} tokens**\n"
        f"- SQL 自纠错：**{settings.max_sql_retries} 次**\n"
        f"- 单次成本上限：**¥{settings.max_run_cost_cny:.2f}**\n"
        f"- 数据存储：**本地 DuckDB**"
    )

    st.divider()
    st.subheader("示例问题")
    examples = [
        "过去7天链上交易量是多少？",
        "为什么上周链上交易量跌了？",
        "上周活跃用户是多少？",
        "过去30天手续费收入是多少？",
    ]
    for index, example in enumerate(examples):
        if st.button(example, key=f"example_{index}", use_container_width=True):
            st.session_state["pending_question"] = example

# ---------------------------------------------------------------- main
st.title("Text-to-SQL 业务自主分析 Agent")
st.caption(
    "Router → Semantic → CodeGen → Execution → Attribution → Insight ｜ "
    "所有聚合在本地完成"
)

if not db_ready:
    st.stop()

graph = load_graph(mode, settings.db_path.stat().st_mtime)
runtime_settings = replace(settings, llm_mode=mode)

st.session_state.setdefault("runs", [])

for item in st.session_state["runs"]:
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant"):
        render_run(item["question"], item["state"], item["meta"])


def handle(question: str) -> None:
    question = question.strip()
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Agent 正在本地分析与执行…"):
            state, meta = run_agent(
                question, graph=graph, settings=runtime_settings
            )
        render_run(question, state, meta)
    st.session_state["runs"].append(
        {"question": question, "state": state, "meta": meta}
    )


pending = st.session_state.pop("pending_question", None)
typed = st.chat_input("例如：为什么上周链上交易量跌了？")
question = pending or typed
if question:
    handle(question)

