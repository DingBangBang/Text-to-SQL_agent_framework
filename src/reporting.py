"""Formatting, markdown tables and the standalone HTML E2E report."""

from __future__ import annotations

import html
import json
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings
from .i18n import dimension_label

MAX_TABLE_ROWS = 25


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def format_value(value: Any, unit: str = "") -> str:
    """Human-readable business formatting (K/M/B for USD, thousands else)."""
    if is_missing(value):
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return "N/A"

    prefix = "$" if unit == "USD" else ""
    suffix = "" if unit in ("", "USD") else f" {unit}"
    absolute = abs(number)
    if unit == "USD" and number < 0:
        prefix = "-$"
    magnitude = absolute if unit == "USD" else number
    if unit in ("人", "users"):
        return f"{number:,.0f}{suffix}"
    if absolute >= 1_000_000_000:
        return f"{prefix}{magnitude / 1_000_000_000:,.2f}B{suffix}"
    if absolute >= 1_000_000:
        return f"{prefix}{magnitude / 1_000_000:,.2f}M{suffix}"
    if absolute >= 10_000:
        return f"{prefix}{magnitude / 1_000:,.1f}K{suffix}"
    if unit == "人" or (number.is_integer() and absolute >= 1000):
        return f"{number:,.0f}{suffix}"
    return f"{prefix}{magnitude:,.2f}{suffix}"


def format_pct(value: Any) -> str:
    if is_missing(value):
        return "N/A"
    try:
        return f"{float(value):+.2f}%"
    except (TypeError, ValueError):
        return str(value)


def df_to_markdown(df: pd.DataFrame, max_rows: int = MAX_TABLE_ROWS, unit: str = "") -> str:
    """Small dependency-free markdown table renderer."""
    if df is None or df.empty:
        return "_（无数据）_"
    view = df.head(max_rows).copy()
    columns = [str(c) for c in view.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        cells = []
        for column in view.columns:
            value = row[column]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if column in ("change_pct",):
                    cells.append(format_pct(value))
                else:
                    cells.append(format_value(value, unit))
            elif isinstance(value, (datetime, date)):
                cells.append(value.strftime("%Y-%m-%d"))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_仅展示前 {max_rows} 行，共 {len(df)} 行。_")
    return "\n".join(lines)


def df_to_html_table(
    df: pd.DataFrame, max_rows: int = MAX_TABLE_ROWS, unit: str = ""
) -> str:
    if df is None or df.empty:
        return '<p class="muted">无数据</p>'
    view = df.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in view.columns)
    rows = []
    for _, row in view.iterrows():
        cells = []
        for column in view.columns:
            value = row[column]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                text = (
                    format_pct(value)
                    if column == "change_pct"
                    else format_value(value, unit)
                )
            elif isinstance(value, (datetime, date)):
                text = value.strftime("%Y-%m-%d")
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    note = (
        f'<p class="muted">仅展示前 {max_rows} 行，共 {len(df)} 行</p>'
        if len(df) > max_rows
        else ""
    )
    return (
        '<div class="table-wrap"><table class="data-table"><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + note
    )


def simple_markdown_to_html(text: str) -> str:
    """Render the constrained markdown our insight prompt produces."""
    if not text:
        return ""
    lines = text.splitlines()
    out: list[str] = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                out.append("</ul>")
                in_list = False
            continue
        if stripped.startswith(">"):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f'<blockquote>{_inline(stripped[1:].strip())}</blockquote>')
            continue
        if stripped.startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(stripped[2:])}</li>")
            continue
        if re.match(r"^\d+\.\s", stripped):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<p>{_inline(stripped)}</p>")
            continue
        if stripped.startswith("**") and stripped.endswith("**"):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<h4>{_inline(stripped)}</h4>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        out.append(f"<p>{_inline(stripped)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _format_delta(value: Any, unit: str) -> str:
    if is_missing(value):
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    sign = "+" if number >= 0 else "-"
    return sign + format_value(abs(number), unit)


def build_insight_context(
    state: dict[str, Any], unit: str = "", language: str = "zh"
) -> dict[str, Any]:
    """Derive driver/peak facts from attribution output for prompt + MockLLM."""
    attribution = state.get("attribution") or {}
    daily: pd.DataFrame | None = state.get("result_df")
    breakdowns: dict[str, pd.DataFrame] = attribution.get("breakdowns") or {}

    current = attribution.get("current_total")
    previous = attribution.get("previous_total")
    change_value = attribution.get("change_value")
    change_pct = attribution.get("change_pct")

    candidates: list[dict[str, Any]] = []
    for dimension, frame in breakdowns.items():
        if frame is None or frame.empty:
            continue
        for _, row in frame.head(12).iterrows():
            delta = row.get("change_value")
            if is_missing(delta):
                continue
            share = None
            if not is_missing(change_value) and float(change_value) != 0:
                share = abs(float(delta)) / abs(float(change_value)) * 100
            candidates.append(
                {
                    "dimension": dimension,
                    "name": str(row.get("dimension_value", "")),
                    "change_value": float(delta),
                    "change_pct": row.get("change_pct"),
                    "share_value": share,
                }
            )

    def _enrich(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": (
                f"{dimension_label(item['dimension'], language)}={item['name']}"
            ),
            "dimension": item["dimension"],
            "change": _format_delta(item["change_value"], unit),
            "change_pct": format_pct(item.get("change_pct")),
            "share": (
                f"{item['share_value']:.1f}%"
                if item.get("share_value") is not None
                else "N/A"
            ),
        }

    negative = [c for c in candidates if c["change_value"] < 0]
    positive = [c for c in candidates if c["change_value"] > 0]
    top_driver = _enrich(min(negative, key=lambda c: c["change_value"])) if negative else None
    positive_driver = (
        _enrich(max(positive, key=lambda c: c["change_value"])) if positive else None
    )

    peak = None
    if daily is not None and not daily.empty:
        value_columns = [
            c
            for c in daily.columns
            if pd.api.types.is_numeric_dtype(daily[c])
        ]
        if value_columns and len(daily.columns) >= 2:
            value_col = value_columns[-1]
            try:
                idx = daily[value_col].idxmax()
                raw_name = daily.loc[idx, daily.columns[0]]
                peak_name = (
                    raw_name.strftime("%Y-%m-%d")
                    if hasattr(raw_name, "strftime")
                    else str(raw_name)
                )
                peak = {
                    "name": peak_name,
                    "value": format_value(daily.loc[idx, value_col], unit),
                }
            except Exception:
                peak = None

    window = attribution.get("window") or {}
    if window:
        start = window.get("start", "")
        end = window.get("end", "")
        label = window.get("label", state.get("time_range", ""))
        window_label = (
            f"{label} ({start} ~ {end})"
            if language == "en"
            else f"{label}（{start} ~ {end}）"
        )
    else:
        window_label = str(state.get("time_range", ""))
    metric = state.get("semantic") or {}

    return {
        "metric_label": metric.get(
            "label", state.get("metric", "metric" if language == "en" else "指标")
        ),
        "metric_key": metric.get("key", state.get("metric", "")),
        "window_label": window_label,
        "current_value": format_value(current, unit),
        "previous_value": format_value(previous, unit),
        "change_pct": format_pct(change_pct),
        "top_driver": top_driver,
        "positive_driver": positive_driver,
        "peak": peak,
    }


def attribution_tables_markdown(
    state: dict[str, Any],
    unit: str = "",
    max_rows: int = 8,
    language: str = "zh",
) -> str:
    attribution = state.get("attribution") or {}
    breakdowns: dict[str, pd.DataFrame] = attribution.get("breakdowns") or {}
    if not breakdowns:
        return (
            "_No attribution was run for this query._"
            if language == "en"
            else "_本次为查询型问题，未执行多维归因。_"
        )
    blocks = []
    for dimension, frame in breakdowns.items():
        title = (
            f"**Breakdown by {dimension}**"
            if language == "en"
            else f"**按 {dimension} 拆解**"
        )
        blocks.append(title + "\n")
        blocks.append(df_to_markdown(frame, max_rows=max_rows, unit=unit))
    return "\n\n".join(blocks)


def _inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped)
    return escaped


REPORT_CSS = """
:root { --bg:#f5f7fb; --card:#ffffff; --ink:#111827; --muted:#6b7280;
        --line:#e5e7eb; --accent:#2563eb; --accent-soft:#dbeafe;
        --green:#059669; --red:#dc2626; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
       "Hiragino Sans GB","Microsoft YaHei",sans-serif; line-height:1.65; }
.wrap { max-width:1180px; margin:0 auto; padding:32px 22px 64px; }
.hero { background:linear-gradient(135deg,#0f172a,#1d4ed8); color:#fff;
        border-radius:18px; padding:30px 34px; box-shadow:0 14px 40px rgba(15,23,42,.18); }
.hero h1 { margin:0 0 6px; font-size:27px; letter-spacing:.2px; }
.hero p { margin:0; color:#c7d2fe; font-size:14px; }
.badges { margin-top:16px; display:flex; gap:8px; flex-wrap:wrap; }
.badge { background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25);
         color:#fff; padding:4px 11px; border-radius:999px; font-size:12px; }
.badge.live { background:#059669; border-color:#059669; }
.badge.mock { background:#d97706; border-color:#d97706; }
.kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:22px 0; }
.kpi { background:var(--card); border:1px solid var(--line); border-radius:14px;
       padding:16px 18px; box-shadow:0 4px 14px rgba(15,23,42,.04); }
.kpi .label { color:var(--muted); font-size:12px; letter-spacing:.4px;
              text-transform:uppercase; }
.kpi .value { font-size:22px; font-weight:700; margin-top:6px; word-break:break-all; }
.kpi .value.pos { color:var(--green); } .kpi .value.neg { color:var(--red); }
.card { background:var(--card); border:1px solid var(--line); border-radius:16px;
        padding:22px 24px; margin:18px 0; box-shadow:0 6px 22px rgba(15,23,42,.05); }
.card h2 { font-size:17px; margin:0 0 14px; display:flex; align-items:center; gap:9px; }
.step { width:24px; height:24px; border-radius:8px; background:var(--accent-soft);
        color:var(--accent); font-size:13px; display:inline-flex; align-items:center;
        justify-content:center; font-weight:700; }
pre.sql { background:#0f172a; color:#e2e8f0; padding:18px; border-radius:12px;
          overflow:auto; font-size:13px; line-height:1.55;
          font-family:"SFMono-Regular",Consolas,Menlo,monospace; }
.table-wrap { overflow:auto; }
table.data-table { border-collapse:collapse; width:100%; font-size:13px; }
table.data-table th { background:#f8fafc; text-align:left; padding:9px 11px;
                      border-bottom:1px solid var(--line); color:#334155;
                      font-weight:600; white-space:nowrap; }
table.data-table td { padding:8px 11px; border-bottom:1px solid #f1f5f9;
                      white-space:nowrap; }
.muted { color:var(--muted); font-size:12px; }
.trace { list-style:none; padding:0; margin:0; }
.trace li { border-left:2px solid var(--accent-soft); padding:0 0 14px 18px;
            position:relative; font-size:14px; }
.trace li:last-child { padding-bottom:0; }
.trace li::before { content:""; width:9px; height:9px; border-radius:50%;
                    background:var(--accent); position:absolute; left:-5.5px; top:7px; }
.trace .node { font-weight:600; margin-right:8px; }
.trace .detail { color:var(--muted); font-size:13px; }
.insight h4 { margin:18px 0 6px; font-size:15px; color:#1e3a8a; }
.insight p { margin:6px 0; }
.insight ul { margin:6px 0 6px 20px; padding:0; }
.insight blockquote { margin:12px 0 0; padding:9px 13px; background:#f8fafc;
                      border-left:3px solid var(--accent); color:var(--muted);
                      font-size:12px; border-radius:0 8px 8px 0; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
.footer { color:var(--muted); font-size:12px; text-align:center; margin-top:28px; }
@media (max-width:860px) { .kpis { grid-template-columns:1fr 1fr; }
                           .grid2 { grid-template-columns:1fr; } }
"""


def _plot_html(daily: pd.DataFrame | None, value_column: str, title: str) -> str:
    if daily is None or daily.empty or len(daily.columns) < 2:
        return '<p class="muted">无可视化数据</p>'
    try:
        import plotly.express as px

        fig = px.line(
            daily,
            x=daily.columns[0],
            y=value_column if value_column in daily.columns else daily.columns[-1],
            markers=True,
            title=title,
        )
        fig.update_layout(
            template="plotly_white",
            height=380,
            margin=dict(l=20, r=20, t=46, b=20),
            title_font_size=15,
            font_family="-apple-system, PingFang SC, sans-serif",
        )
        return fig.to_html(full_html=False, include_plotlyjs=True)
    except Exception as exc:  # pragma: no cover - chart is best-effort
        return f'<p class="muted">图表渲染失败：{html.escape(str(exc))}</p>'




def render_e2e_html(
    state: dict[str, Any],
    settings: Settings,
    output_path: Path | str,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write a self-contained HTML report of one full agent run."""
    meta = meta or {}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    semantic = state.get("semantic") or {}
    unit = semantic.get("unit", "")
    route = state.get("route") or {}
    daily: pd.DataFrame | None = state.get("result_df")
    attribution = state.get("attribution") or {}
    breakdowns: dict[str, pd.DataFrame] = attribution.get("breakdowns") or {}
    ctx = build_insight_context(state, unit)

    change_value = attribution.get("change_pct")
    change_class = ""
    if not is_missing(change_value):
        change_class = "pos" if float(change_value) >= 0 else "neg"
    mock = bool(meta.get("mock", settings.llm_mode != "deepseek"))
    mode_badge = (
        '<span class="badge mock">MOCK · 本地零成本</span>'
        if mock
        else '<span class="badge live">LIVE · DeepSeek</span>'
    )
    generated_at = meta.get("generated_at") or datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    elapsed = float(meta.get("elapsed_s", 0) or 0)
    cost = float(meta.get("cost_cny", 0) or 0)
    tokens_in = int(meta.get("input_tokens", 0) or 0)
    tokens_out = int(meta.get("output_tokens", 0) or 0)

    trace_items = "".join(
        f'<li><span class="node">{html.escape(str(step.get("node", "")))}</span>'
        f'<span class="detail">{html.escape(str(step.get("detail", "")))}</span></li>'
        for step in state.get("trace", [])
    ) or '<li><span class="detail">无执行轨迹</span></li>'

    breakdown_html = "".join(
        "<h3 style='font-size:14px;margin:18px 0 8px'>按 "
        + html.escape(str(dim))
        + " 拆解</h3>"
        + df_to_html_table(frame, unit=unit)
        for dim, frame in breakdowns.items()
    ) or '<p class="muted">本次为查询型问题，未执行多维归因。</p>'

    chart = _plot_html(
        daily,
        semantic.get("value_column", ""),
        f"{semantic.get('label', state.get('metric', '指标'))} · "
        f"{attribution.get('window', {}).get('label', '')}",
    )

    if state.get("needs_clarification"):
        status_block = (
            '<div class="card"><h2><span class="step">?</span>需要补充信息</h2>'
            f'<p>{html.escape(str(state.get("clarification", "")))}</p></div>'
        )
    elif state.get("status") == "failed":
        status_block = (
            '<div class="card"><h2><span class="step">!</span>执行失败</h2>'
            f'<p>{html.escape(str(state.get("insight", "执行失败，请人工介入")))}</p>'
            f'<p class="muted">最后错误：{html.escape(str(state.get("sql_error") or ""))}</p>'
            "</div>"
        )
    else:
        status_block = ""

    header = f"""
  <section class="hero">
    <h1>Text-to-SQL 业务分析 Agent · 端到端报告</h1>
    <p>本地 DuckDB 沙箱执行 · LangGraph 多节点编排 · 单次成本受控</p>
    <div class="badges">
      {mode_badge}
      <span class="badge">模型：{html.escape(str(meta.get("model", settings.model)))}</span>
      <span class="badge">生成时间：{html.escape(str(generated_at))}</span>
      <span class="badge">耗时：{elapsed:.2f}s</span>
    </div>
  </section>

  <section class="kpis">
    <div class="kpi"><div class="label">当前值</div>
      <div class="value">{html.escape(str(ctx["current_value"]))}</div></div>
    <div class="kpi"><div class="label">上一周期</div>
      <div class="value">{html.escape(str(ctx["previous_value"]))}</div></div>
    <div class="kpi"><div class="label">环比</div>
      <div class="value {change_class}">{html.escape(str(ctx["change_pct"]))}</div></div>
    <div class="kpi"><div class="label">Token / 成本</div>
      <div class="value" style="font-size:17px">{tokens_in + tokens_out:,} tokens<br>
      ¥{cost:.4f}</div></div>
  </section>

  {status_block}

  <section class="card">
    <h2><span class="step">1</span>用户问题与意图路由</h2>
    <p><strong>问题：</strong>{html.escape(str(state.get("question", "")))}</p>
    <p><strong>路由结果：</strong>指标 = {html.escape(str(route.get("metric")))}
      ｜ 时间窗口 = {html.escape(str(route.get("time_range")))}
      ｜ 动作 = {html.escape(str(route.get("action")))}
      ｜ 来源 = {html.escape(str(route.get("source", "llm")))}
    </p>
    <p class="muted">语义口径：{html.escape(str(semantic.get("description", "")))}</p>
  </section>

  <section class="card">
    <h2><span class="step">2</span>生成的 DuckDB SQL</h2>
    <pre class="sql">{html.escape(str(state.get("sql", "（未生成）")))}</pre>
  </section>

  <section class="card">
    <h2><span class="step">3</span>执行结果（本地沙箱）</h2>
    {chart}
    {df_to_html_table(daily, unit=unit)}
    <p class="muted">执行耗时 {state.get("exec_ms", 0)} ms ·
      返回 {state.get("row_count", 0)} 行 · 只读连接，原始明细未出本机</p>
  </section>

  <section class="card">
    <h2><span class="step">4</span>多维归因（当前窗口 vs 上一等长窗口）</h2>
    {breakdown_html}
  </section>

  <section class="card insight">
    <h2><span class="step">5</span>智能洞察</h2>
    {simple_markdown_to_html(str(state.get("insight", "")))}
  </section>

  <section class="card">
    <h2><span class="step">i</span>Agent 执行轨迹</h2>
    <ul class="trace">{trace_items}</ul>
  </section>
"""
    footer = f"""
  <p class="footer">
    数据隐私：全部聚合在本地 DuckDB 完成，仅指标口径、聚合结果与问题文本进入 LLM；
    单次请求输入上限 {settings.max_input_tokens} tokens，SQL 自纠错上限
    {settings.max_sql_retries} 次，单次运行成本上限 ¥{settings.max_run_cost_cny:.2f}。
  </p>
"""
    document = (
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Text-to-SQL Agent · E2E Report</title>"
        f"<style>{REPORT_CSS}</style></head><body><div class='wrap'>{header}{footer}"
        "</div></body></html>"
    )
    output_path.write_text(document, encoding="utf-8")
    return output_path


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        for column in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(frame[column]):
                frame[column] = frame[column].astype(str)
        return frame.to_dict(orient="records")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    return value


def state_to_serializable(state: dict[str, Any]) -> dict[str, Any]:
    serialisable: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, dict):
            serialisable[key] = {k: _json_safe(v) for k, v in value.items()}
        elif isinstance(value, pd.DataFrame):
            serialisable[key] = _json_safe(value)
        elif isinstance(value, (str, int, float, bool, list, tuple)) or value is None:
            serialisable[key] = _json_safe(value)
        else:
            serialisable[key] = str(value)
    return serialisable


def save_state_json(state: dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state_to_serializable(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def write_streamlit_placeholder(path: Path | str) -> Path:
    """Generate an SVG mock of the dashboard (used when no browser screenshot)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="760"
viewBox="0 0 1280 760">
<rect width="1280" height="760" fill="#f5f7fb"/>
<rect x="0" y="0" width="290" height="760" fill="#0f172a"/>
<text x="24" y="46" fill="#e2e8f0" font-family="PingFang SC, sans-serif"
 font-size="17" font-weight="700">Schema &amp; Metrics</text>
<text x="24" y="86" fill="#93c5fd" font-family="monospace" font-size="12.5">users(user_id, register_date, tier_level)</text>
<text x="24" y="112" fill="#93c5fd" font-family="monospace" font-size="12.5">trades(trade_id, user_id, trade_date,</text>
<text x="24" y="132" fill="#93c5fd" font-family="monospace" font-size="12.5">  symbol, trade_volume_usd, fee_usd, chain_type)</text>
<text x="24" y="176" fill="#e2e8f0" font-family="PingFang SC, sans-serif"
 font-size="14" font-weight="600">指标口径</text>
<text x="24" y="204" fill="#cbd5e1" font-family="PingFang SC, sans-serif" font-size="12.5">活跃用户：窗口内 distinct user_id</text>
<text x="24" y="228" fill="#cbd5e1" font-family="PingFang SC, sans-serif" font-size="12.5">交易量：SUM(trade_volume_usd)</text>
<text x="24" y="252" fill="#cbd5e1" font-family="PingFang SC, sans-serif" font-size="12.5">链上交易：chain_type = 'onchain'</text>
<rect x="24" y="690" width="242" height="42" rx="10" fill="#1d4ed8"/>
<text x="52" y="716" fill="#ffffff" font-family="PingFang SC, sans-serif" font-size="13">Mock / DeepSeek 模式切换</text>
<text x="322" y="48" fill="#111827" font-family="PingFang SC, sans-serif"
 font-size="22" font-weight="700">Text-to-SQL 业务自主分析 Agent</text>
<text x="322" y="76" fill="#6b7280" font-family="PingFang SC, sans-serif" font-size="13">本地 DuckDB · LangGraph · 成本受控</text>
<rect x="322" y="100" width="900" height="86" rx="14" fill="#ffffff" stroke="#e5e7eb"/>
<text x="344" y="132" fill="#111827" font-family="PingFang SC, sans-serif" font-size="14" font-weight="600">为什么上周链上交易量跌了？</text>
<text x="344" y="160" fill="#6b7280" font-family="PingFang SC, sans-serif" font-size="12">Router → Semantic → CodeGen → Execution → Attribution → Insight</text>
<rect x="322" y="202" width="900" height="250" rx="14" fill="#ffffff" stroke="#e5e7eb"/>
<text x="344" y="232" fill="#1e3a8a" font-family="PingFang SC, sans-serif" font-size="14" font-weight="600">链上交易量 · 过去 7 天（每日）</text>
<polyline points="360,410 440,388 520,400 600,355 680,368 760,300 840,318 920,270 1000,286 1080,240 1160,258"
 fill="none" stroke="#2563eb" stroke-width="3"/>
<polyline points="360,430 440,420 520,425 600,408 680,412 760,395 840,400 920,388 1000,392 1080,380 1160,384"
 fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="6 5"/>
<line x1="360" y1="440" x2="1170" y2="440" stroke="#e5e7eb"/>
<rect x="322" y="470" width="438" height="230" rx="14" fill="#ffffff" stroke="#e5e7eb"/>
<text x="344" y="500" fill="#111827" font-family="PingFang SC, sans-serif" font-size="14" font-weight="600">归因：按币种</text>
<rect x="344" y="520" width="390" height="26" rx="5" fill="#dbeafe"/>
<rect x="344" y="556" width="300" height="26" rx="5" fill="#dbeafe"/>
<rect x="344" y="592" width="240" height="26" rx="5" fill="#dbeafe"/>
<rect x="344" y="628" width="370" height="26" rx="5" fill="#fee2e2"/>
<rect x="784" y="470" width="438" height="230" rx="14" fill="#ffffff" stroke="#e5e7eb"/>
<text x="806" y="500" fill="#111827" font-family="PingFang SC, sans-serif" font-size="14" font-weight="600">智能洞察</text>
<text x="806" y="530" fill="#059669" font-family="PingFang SC, sans-serif" font-size="13" font-weight="600">结论</text>
<text x="806" y="556" fill="#374151" font-family="PingFang SC, sans-serif" font-size="12">链上交易量环比下滑，主因 BTC 与 ETH 链上</text>
<text x="806" y="578" fill="#374151" font-family="PingFang SC, sans-serif" font-size="12">大额成交减少，建议排查跨链桥拥堵与做市深度。</text>
<text x="806" y="610" fill="#1e3a8a" font-family="PingFang SC, sans-serif" font-size="13" font-weight="600">数据支撑</text>
<text x="806" y="634" fill="#374151" font-family="PingFang SC, sans-serif" font-size="12">当前值 vs 上一周期，按日/币种/链类型拆解</text>
<text x="640" y="742" fill="#9ca3af" font-family="PingFang SC, sans-serif"
 font-size="11" text-anchor="middle">Streamlit dashboard · placeholder (replace with a real screenshot after `streamlit run app/dashboard.py`)</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return path

