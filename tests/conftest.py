"""Shared pytest fixtures. Everything runs offline against a small DuckDB."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.semantic import SemanticRegistry  # noqa: E402
from simulate_data import build_database  # noqa: E402

TEST_TODAY = date(2026, 10, 20)


@pytest.fixture(scope="session")
def registry() -> SemanticRegistry:
    return SemanticRegistry.load(ROOT / "metrics.yaml")


@pytest.fixture(scope="session")
def test_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("warehouse") / "test_analytics.duckdb"
    build_database(
        db_path=path,
        n_users=800,
        n_trades=12_000,
        days=30,
        seed=7,
        today=TEST_TODAY,
    )
    return path


@pytest.fixture()
def settings(test_db: Path) -> Settings:
    return Settings(
        llm_mode="mock",
        db_path=test_db,
        metrics_path=ROOT / "metrics.yaml",
        max_input_tokens=3000,
        max_sql_retries=3,
        max_run_cost_cny=1.0,
        max_result_rows=2000,
    )
