"""Bilingual Streamlit dashboard with step-by-step agent transparency.

The language switch is not cosmetic: it also selects the agent language, so
routing clarifications, trace messages, Mock/DeepSeek prompts, and the final
insight are produced in the selected language.
"""

from __future__ import annotations

import html
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "app") not in sys.path:
    sys.path.insert(0, str(ROOT / "app"))

from src.config import Settings  # noqa: E402
from src.db import schema_snapshot  # noqa: E402
from src.graph import build_graph, run_agent  # noqa: E402
from src.reporting import build_insight_context, format_pct  # noqa: E402
from src.semantic import SemanticRegistry  # noqa: E402
from ui_text import EXAMPLES, UI_TEXT, dimension_label  # noqa: E402

st.set_page_config(
    page_title="Text-to-SQL Analytics Agent",
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


def cost_text(cost_cny: float, language: str) -> str:
    if language == "en":
        return f"${settings.cost_usd(cost_cny):.6f}"
    return f"¥{cost_cny:.4f}"


@st.cache_resource(show_spinner="Loading agent / 加载 Agent 状态机…")
def load_graph(mode: str, _db_mtime: float, language: str):
    runtime = replace(settings, llm_mode=mode)
    return build_graph(runtime, language=language)


@st.cache_data(show_spinner=False)
def load_schema(_db_mtime: float) -> dict:
    return schema_snapshot(settings.db_path)


def chart_for(df: pd.DataFrame | None, unit: str, copy: dict) -> None:
    if df is None or df.empty:
        st.info(copy["no_data"])
        return
    x_column = df.columns[0]
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        st.dataframe(df, use_container_width=True)
        return
    y_column = numeric[-1]
    labels = {y_column: f"{y_column} ({unit})" if unit else y_column}
    if pd.api.types.is_datetime64_any_dtype(df[x_column]) or "date" in str(x_column):
        figure = px.line(df, x=x_column, y=y_column, markers=True, labels=labels)
    else:
        figure = px.bar(df.head(15), x=x_column, y=y_column, labels=labels)
    figure.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        font_family="-apple-system, PingFang SC, sans-serif",
    )
    st.plotly_chart(figure, use_container_width=True)


def render_run(
    question: str,
    state: dict,
    meta: dict,
    language: str,
    copy: dict,
) -> None:
    st.markdown(f"**{copy['question']}:** {question}")
    if state.get("needs_clarification"):
        st.warning(state.get("clarification", copy["clarify_default"]))
        return
    if state.get("status") == "failed":
        st.error(state.get("insight", copy["failed_default"]))
        return

    attribution = state.get("attribution") or {}
    semantic = state.get("semantic") or {}
    unit = semantic.get("unit", "")
    payload = build_insight_context(state, unit, language)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(copy["kpi_current"], payload["current_value"])
    col2.metric(copy["kpi_previous"], payload["previous_value"])
    col3.metric(copy["kpi_change"], format_pct(attribution.get("change_pct")))
    col4.metric(
        copy["kpi_cost"],
        f"{meta['input_tokens'] + meta['output_tokens']:,}",
        cost_text(float(meta.get("cost_cny", 0.0)), language),
    )

    with st.expander(copy["thinking_title"], expanded=True):
        route = state.get("route") or {}
        separator = copy["separator"]
        action_raw = str(route.get("action"))
        action_label = copy["action_labels"].get(action_raw, action_raw)
        st.markdown(
            f"<span class='step-tag'>Router</span> {copy['route_metric']} "
            f"<b>{html.escape(str(route.get('metric')))}</b>{separator}"
            f"{copy['route_window']} "
            f"<b>{html.escape(str(route.get('time_range')))}</b>{separator}"
            f"{copy['route_action']} <b>{html.escape(action_label)}</b>{separator}"
            f"{copy['route_source']} "
            f"<b>{html.escape(str(route.get('source')))}</b>",
            unsafe_allow_html=True,
        )
        for step in state.get("trace", []):
            st.markdown(
                f"<span class='step-tag'>{html.escape(str(step.get('node')))}</span>"
                f"<span class='muted'>{html.escape(str(step.get('detail')))}</span>",
                unsafe_allow_html=True,
            )

    with st.expander(copy["sql_title"], expanded=True):
        st.code(state.get("sql", ""), language="sql")
        st.caption(
            f"{copy['sql_source']}: {state.get('sql_source')}  |  "
            f"{copy['sql_attempts']}: {state.get('sql_attempts')}"
        )

    with st.expander(copy["data_title"], expanded=True):
        df = state.get("result_df")
        chart_for(df, unit, copy)
        if df is not None:
            st.dataframe(df, use_container_width=True, height=240)
        st.caption(
            copy["exec_caption"].format(
                ms=state.get("exec_ms", 0), rows=state.get("row_count", 0)
            )
        )

    if attribution.get("breakdowns"):
        with st.expander(copy["attr_title"], expanded=False):
            for dimension, frame in attribution["breakdowns"].items():
                st.markdown(
                    copy["breakdown_by"].format(
                        dim=dimension_label(dimension, language)
                    )
                )
                st.dataframe(frame, use_container_width=True, height=220)

    with st.expander(copy["insight_title"], expanded=True):
        st.markdown(state.get("insight", ""))
        if meta.get("mock"):
            st.caption(copy["mock_caption"])



# ---------------------------------------------------------------- sidebar
st.session_state.setdefault("runs", [])
st.session_state.setdefault("ui_language", "中文")
db_ready = settings.db_path.exists()

with st.sidebar:
    language_choice = st.radio(
        "Language / 语言",
        options=["中文", "English"],
        key="ui_language",
        horizontal=True,
        help="切换界面与 Agent 输出语言 / Switch UI and agent output language",
    )
    language = "en" if language_choice == "English" else "zh"
    copy = UI_TEXT[language]

    previous_language = st.session_state.get("active_language")
    if previous_language is not None and previous_language != language:
        st.session_state["runs"] = []
        st.session_state.pop("pending_question", None)
        st.session_state["language_notice"] = copy["lang_switched"]
    st.session_state["active_language"] = language

    st.title(copy["sidebar_title"])
    st.caption(copy["sidebar_caption"])

    mode = st.radio(
        copy["mode_label"],
        options=["mock", "deepseek"],
        index=0 if settings.llm_mode != "deepseek" else 1,
        key="llm_mode",
        format_func=lambda value: (
            copy["mode_mock"] if value == "mock" else copy["mode_deepseek"]
        ),
        help=copy["mode_help"],
    )
    if mode == "deepseek" and not settings.api_key:
        st.error(copy["missing_key"])
        mode = "mock"

    st.divider()
    st.subheader(copy["schema_title"])
    if not db_ready:
        st.error(copy["schema_missing"])
    else:
        schema = load_schema(settings.db_path.stat().st_mtime)
        for table, meta in schema.items():
            with st.expander(
                f"{table} · {meta['rows']:,} {copy['rows_suffix']}", expanded=True
            ):
                st.dataframe(
                    pd.DataFrame(meta["columns"]), hide_index=True, height=170
                )

    st.divider()
    st.subheader(copy["metrics_title"])
    st.markdown(
        "\n".join(
            f"- **{metric.display_label(language)}**: "
            f"{metric.display_description(language)}"
            for metric in registry.metrics.values()
        )
    )
    st.caption(copy["metrics_note"])

    st.divider()
    st.subheader(copy["guardrails_title"])
    cap = (
        f"${settings.cost_usd(settings.max_run_cost_cny):.2f}"
        if language == "en"
        else f"¥{settings.max_run_cost_cny:.2f}"
    )
    st.markdown(
        f"- {copy['guard_input']}: **{settings.max_input_tokens:,} tokens**\n"
        f"- {copy['guard_retries']}: **{settings.max_sql_retries}**\n"
        f"- {copy['guard_cost']}: **{cap}**\n"
        f"- {copy['guard_storage']}: **{copy['guard_storage_value']}**"
    )

    st.divider()
    st.subheader(copy["examples_title"])
    for index, example in enumerate(EXAMPLES[language]):
        if st.button(
            example,
            key=f"example_{language}_{index}",
            use_container_width=True,
        ):
            st.session_state["pending_question"] = example


# ---------------------------------------------------------------- main
notice = st.session_state.pop("language_notice", None)
if notice:
    st.info(notice)

st.title(copy["app_title"])
st.caption(copy["app_caption"])

if not db_ready:
    st.stop()

graph = load_graph(mode, settings.db_path.stat().st_mtime, language)
runtime_settings = replace(settings, llm_mode=mode)

for item in st.session_state["runs"]:
    item_language = item.get("language", language)
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant"):
        render_run(
            item["question"],
            item["state"],
            item["meta"],
            item_language,
            UI_TEXT[item_language],
        )


def handle(question: str) -> None:
    question = question.strip()
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner(copy["spinner"]):
            state, meta = run_agent(
                question,
                graph=graph,
                settings=runtime_settings,
                language=language,
            )
        render_run(question, state, meta, language, copy)
    st.session_state["runs"].append(
        {
            "question": question,
            "state": state,
            "meta": meta,
            "language": language,
        }
    )


pending = st.session_state.pop("pending_question", None)
typed = st.chat_input(copy["chat_placeholder"])
question = pending or typed
if question:
    handle(question)

