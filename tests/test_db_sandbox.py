"""Read-only sandbox tests: generated SQL can never mutate local data."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.db import (
    DatabaseNotReadyError,
    UnsafeSQLError,
    execute_query,
    schema_snapshot,
    validate_sql,
)


def test_select_runs_against_local_duckdb(test_db: Path):
    result = execute_query(test_db, "SELECT COUNT(*) AS n FROM trades")
    assert result.row_count == 1
    assert int(result.df.iloc[0, 0]) == 12_000
    assert result.elapsed_ms >= 0


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE trades",
        "DELETE FROM trades",
        "UPDATE trades SET fee_usd = 0",
        "INSERT INTO trades VALUES (1,1,'2026-01-01','BTC',1,0,'cex')",
        "ATTACH 'evil.db'",
        "SELECT * FROM trades; DROP TABLE trades;",
        "COPY trades TO 'leak.csv'",
        "SELECT * FROM read_csv_auto('/etc/passwd')",
        "PRAGMA database_list",
    ],
)
def test_dangerous_sql_is_rejected(sql: str):
    with pytest.raises(UnsafeSQLError):
        validate_sql(sql)


def test_write_attempt_does_not_change_database(test_db: Path):
    before = schema_snapshot(test_db)["trades"]["rows"]
    with pytest.raises(UnsafeSQLError):
        execute_query(test_db, "DELETE FROM trades")
    after = schema_snapshot(test_db)["trades"]["rows"]
    assert before == after == 12_000


def test_results_are_capped(test_db: Path):
    result = execute_query(test_db, "SELECT * FROM trades", max_rows=10)
    assert result.row_count == 10
    assert result.truncated is True


def test_schema_snapshot_reports_both_tables(test_db: Path):
    schema = schema_snapshot(test_db)
    assert set(schema) == {"users", "trades"}
    trade_columns = [column["name"] for column in schema["trades"]["columns"]]
    assert trade_columns == [
        "trade_id",
        "user_id",
        "trade_date",
        "symbol",
        "trade_volume_usd",
        "fee_usd",
        "chain_type",
    ]


def test_missing_database_has_actionable_error(tmp_path: Path):
    with pytest.raises(DatabaseNotReadyError, match="simulate_data.py"):
        execute_query(tmp_path / "nope.duckdb", "SELECT 1")
