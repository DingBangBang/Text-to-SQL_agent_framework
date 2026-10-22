"""Streamlit smoke test via the official AppTest harness (no browser needed)."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_renders_without_exception(settings, monkeypatch):
    monkeypatch.setenv("DUCKDB_PATH", str(settings.db_path))
    monkeypatch.setenv("METRICS_PATH", str(settings.metrics_path))
    monkeypatch.setenv("LLM_MODE", "mock")

    app = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=90)
    app.run()

    assert not app.exception
    assert app.chat_input
    assert any("Text-to-SQL" in title.value for title in app.title)
    assert any("Analytics Agent" in title.value for title in app.sidebar.title)
    assert any("指标口径" in header.value for header in app.sidebar.subheader)


def test_dashboard_runs_example_question(settings, monkeypatch):
    monkeypatch.setenv("DUCKDB_PATH", str(settings.db_path))
    monkeypatch.setenv("METRICS_PATH", str(settings.metrics_path))
    monkeypatch.setenv("LLM_MODE", "mock")

    app = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=120)
    app.run()

    example = next(
        button for button in app.sidebar.button if "为什么上周链上交易量跌了" in button.label
    )
    example.click().run()

    assert not app.exception
    assert app.session_state["runs"]
    assert any("结论" in block.value for block in app.markdown)
    assert any("智能洞察报告" in expander.label for expander in app.expander)


def test_dashboard_switches_to_english_and_runs(settings, monkeypatch):
    monkeypatch.setenv("DUCKDB_PATH", str(settings.db_path))
    monkeypatch.setenv("METRICS_PATH", str(settings.metrics_path))
    monkeypatch.setenv("LLM_MODE", "mock")

    app = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=120)
    app.run()

    language_radio = next(
        radio for radio in app.sidebar.radio if radio.label == "Language / 语言"
    )
    language_radio.set_value("English").run()

    assert not app.exception
    assert any("Business Analytics Agent" in title.value for title in app.title)
    assert any(
        "Metric definitions" in header.value for header in app.sidebar.subheader
    )
    assert any("Cost guardrails" in header.value for header in app.sidebar.subheader)

    example = next(
        button
        for button in app.sidebar.button
        if "Why did on-chain volume drop last week?" in button.label
    )
    example.click().run()

    assert not app.exception
    assert app.session_state["runs"]
    assert any("**Conclusion**" in block.value for block in app.markdown)
    assert any("AI insight report" in expander.label for expander in app.expander)
    assert not any("结论" in block.value for block in app.markdown)

