"""Run exactly one end-to-end agent execution and render an HTML report.

Cost protection:
* the live report path is written at most once unless ``--force`` is passed;
* the run is aborted before any API call if the key is missing;
* the DeepSeek client enforces the 3,000-token input budget and a ¥1 run cap.

Usage:
    python scripts/run_e2e_test.py --mode mock        # offline preview
    python scripts/run_e2e_test.py --mode deepseek    # one real run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.graph import run_agent  # noqa: E402
from src.reporting import (  # noqa: E402
    render_e2e_html,
    save_state_json,
    write_streamlit_placeholder,
)
from simulate_data import build_database  # noqa: E402

DEFAULT_QUESTION = "为什么上周链上交易量跌了？"
DEFAULT_OUTPUT = ROOT / "outputs" / "e2e_result.html"
PREVIEW_OUTPUT = ROOT / "outputs" / "e2e_preview_mock.html"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--mode", choices=["mock", "deepseek"], default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="allow overwriting a live report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings.from_env()
    mode = args.mode or settings.llm_mode
    settings = replace(settings, llm_mode=mode)
    live = mode == "deepseek"
    output = args.output or (DEFAULT_OUTPUT if live else PREVIEW_OUTPUT)

    if live and not settings.api_key:
        print(
            "[abort] 未配置 DEEPSEEK_API_KEY。请先在 environment.env 中填入真实 Key。\n"
            "        本次未发起任何 API 调用，未产生费用。",
            file=sys.stderr,
        )
        return 3
    if live and output.exists() and not args.force:
        print(
            f"[abort] {output} 已存在。为防止重复计费，单次真实 E2E 默认只跑一次；"
            "如确需重跑请加 --force。",
            file=sys.stderr,
        )
        return 4

    if not settings.db_path.exists():
        print(f"[setup] 未找到 {settings.db_path}，正在生成本地数据…")
        build_database(settings.db_path)

    print(f"[run] mode={mode} question={args.question!r}")
    started = time.perf_counter()
    state, meta = run_agent(args.question, settings=settings)
    wall_clock = time.perf_counter() - started

    meta["elapsed_s"] = meta.get("elapsed_s", wall_clock)
    meta["mock"] = not live or meta.get("mock", False)
    meta["model"] = settings.model if live else "mock-local"
    render_e2e_html(state, settings, output, meta=meta)

    json_output = output.with_suffix(".json")
    save_state_json(state, json_output)
    meta_output = output.with_name(output.stem + "_meta.json")
    meta_output.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    placeholder = write_streamlit_placeholder(
        ROOT / "outputs" / "streamlit_dashboard_placeholder.svg"
    )

    print("-" * 64)
    print(f"status      : {state.get('status')}")
    print(f"route       : {state.get('route')}")
    print(f"sql attempts: {state.get('sql_attempts', 0)}")
    print(f"rows        : {state.get('row_count', 0)}")
    print(f"tokens      : in={meta.get('input_tokens', 0):,} "
          f"out={meta.get('output_tokens', 0):,}")
    print(f"cost (CNY)  : ¥{meta.get('cost_cny', 0.0):.4f}")
    print(f"llm calls   : {len(meta.get('calls', []))}")
    print(f"report      : {output}")
    print(f"report json : {json_output}")
    print(f"report meta : {meta_output}")
    print(f"screenshot  : {placeholder}")
    print("-" * 64)

    if meta.get("cost_cny", 0.0) > settings.max_run_cost_cny:
        print("[fail] 单次运行成本超过上限", file=sys.stderr)
        return 5
    if state.get("status") == "failed":
        print("[fail] Agent 执行失败，详见报告。", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
