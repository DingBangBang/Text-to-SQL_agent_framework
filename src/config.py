"""Central configuration, loaded from ``environment.env`` (never committed)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# environment.env wins over .env; both are git-ignored.
load_dotenv(ROOT / "environment.env", override=False)
load_dotenv(ROOT / ".env", override=False)

# DeepSeek list prices (CNY per 1M tokens). Configurable because pricing moves.
DEFAULT_PRICE_INPUT_CNY_PER_M = 2.0
DEFAULT_PRICE_CACHED_INPUT_CNY_PER_M = 0.2
DEFAULT_PRICE_OUTPUT_CNY_PER_M = 3.0


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path)


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for the agent."""

    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    llm_mode: str = "mock"
    db_path: Path = field(default_factory=lambda: ROOT / "data" / "analytics.duckdb")
    metrics_path: Path = field(default_factory=lambda: ROOT / "metrics.yaml")

    # Hard guardrails -------------------------------------------------------
    max_input_tokens: int = 3000
    max_sql_retries: int = 3
    max_run_cost_cny: float = 1.0
    temperature: float = 0.0
    max_result_rows: int = 2000
    request_timeout_s: int = 90

    # Pricing ---------------------------------------------------------------
    price_input_cny_per_m: float = DEFAULT_PRICE_INPUT_CNY_PER_M
    price_cached_input_cny_per_m: float = DEFAULT_PRICE_CACHED_INPUT_CNY_PER_M
    price_output_cny_per_m: float = DEFAULT_PRICE_OUTPUT_CNY_PER_M

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=_env("DEEPSEEK_API_KEY", ""),
            base_url=_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            model=_env("DEEPSEEK_MODEL", "deepseek-chat"),
            llm_mode=_env("LLM_MODE", "mock").lower(),
            db_path=_resolve_path(_env("DUCKDB_PATH", "data/analytics.duckdb")),
            metrics_path=_resolve_path(_env("METRICS_PATH", "metrics.yaml")),
            max_input_tokens=_env_int("MAX_INPUT_TOKENS", 3000),
            max_sql_retries=_env_int("MAX_SQL_RETRIES", 3),
            max_run_cost_cny=_env_float("MAX_RUN_COST_CNY", 1.0),
            temperature=_env_float("LLM_TEMPERATURE", 0.0),
            max_result_rows=_env_int("MAX_RESULT_ROWS", 2000),
            request_timeout_s=_env_int("LLM_TIMEOUT_SECONDS", 90),
            price_input_cny_per_m=_env_float(
                "PRICE_INPUT_CNY_PER_M", DEFAULT_PRICE_INPUT_CNY_PER_M
            ),
            price_cached_input_cny_per_m=_env_float(
                "PRICE_CACHED_INPUT_CNY_PER_M", DEFAULT_PRICE_CACHED_INPUT_CNY_PER_M
            ),
            price_output_cny_per_m=_env_float(
                "PRICE_OUTPUT_CNY_PER_M", DEFAULT_PRICE_OUTPUT_CNY_PER_M
            ),
        )

    @property
    def is_live(self) -> bool:
        return self.llm_mode == "deepseek"

    def cost_cny(
        self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0
    ) -> float:
        """Estimate the CNY cost of one call using configurable list prices."""
        miss_tokens = max(input_tokens - cached_input_tokens, 0)
        return (
            miss_tokens / 1_000_000 * self.price_input_cny_per_m
            + cached_input_tokens / 1_000_000 * self.price_cached_input_cny_per_m
            + output_tokens / 1_000_000 * self.price_output_cny_per_m
        )
