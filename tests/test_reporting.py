"""Reporting tests: markdown/HTML/JSON artifacts must be complete and escaped."""

from __future__ import annotations

import json
from pathlib import Path

from src.graph import run_agent
from src.mock_llm import MockLLM
from src.reporting import (
    df_to_markdown,
    render_e2e_html,
    save_state_json,
    write_streamlit_placeholder,
)

from conftest import TEST_TODAY


def _state(settings):
    return run_agent(
        "为什么上周链上交易量跌了？",
        settings=settings,
        llm=MockLLM(settings),
        today=TEST_TODAY,
    )


def test_html_report_contains_all_sections(settings, tmp_path: Path):
    state, meta = _state(settings)
    output = tmp_path / "e2e.html"
    render_e2e_html(state, settings, output, meta=meta)

    document = output.read_text(encoding="utf-8")
    assert output.exists() and output.stat().st_size > 4000
    assert "端到端报告" in document
    assert "意图路由" in document
    assert "生成的 DuckDB SQL" in document
    assert "多维归因" in document
    assert "智能洞察" in document
    assert "MOCK · 本地零成本" in document
    assert "chain_type = &#x27;onchain&#x27;" in document  # escaped SQL
    assert "<script" in document  # plotly bundle


def test_json_export_is_serialisable(settings, tmp_path: Path):
    state, _ = _state(settings)
    path = save_state_json(state, tmp_path / "state.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["route"]["metric"] == "onchain_volume"
    assert isinstance(payload["result_df"], list)


def test_markdown_table_formats_percentages():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "dimension_value": ["BTC", "ETH"],
            "change_pct": [-12.3456, 3.2],
            "change_value": [-1_200_000.0, 300.0],
        }
    )
    markdown = df_to_markdown(frame, unit="USD")
    assert "| dimension_value | change_pct | change_value |" in markdown
    assert "-12.35%" in markdown
    assert "$1.20M" in markdown


def test_streamlit_placeholder_svg(tmp_path: Path):
    path = write_streamlit_placeholder(tmp_path / "dashboard.svg")
    svg = path.read_text(encoding="utf-8")
    assert svg.startswith("<svg")
    assert "Streamlit dashboard" in svg
