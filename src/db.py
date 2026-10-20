"""DuckDB access with a read-only, single-statement SQL sandbox.

Security model:
* the query connection is opened ``read_only=True`` — writes are impossible;
* exactly one statement is allowed, and it must be ``SELECT`` / ``WITH``;
* DDL, DML, file functions and DuckDB admin functions are rejected;
* external file access is disabled and results are hard-capped.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

FORBIDDEN_KEYWORDS = re.compile(
    r"\b("
    r"insert|update|delete|merge|drop|create|alter|attach|detach|copy|"
    r"install|load|pragma|call|export|import|truncate|grant|revoke|"
    r"vacuum|checkpoint|use|comment|analyze|begin|commit|rollback|"
    r"read_csv|read_csv_auto|read_parquet|read_json|read_json_auto|"
    r"parquet_scan|csv_scan|glob|sniff_csv|duckdb_\w+|sqlite_scan|"
    r"postgres_scan|mysql_scan|iceberg_scan|delta_scan"
    r")\b",
    flags=re.IGNORECASE,
)

SINGLE_STATEMENT_ALLOWED = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


class UnsafeSQLError(ValueError):
    """Raised when generated SQL violates the read-only sandbox contract."""


class DatabaseNotReadyError(FileNotFoundError):
    """Raised when the local warehouse has not been generated yet."""


@dataclass
class QueryResult:
    sql: str
    df: pd.DataFrame
    elapsed_ms: int
    row_count: int
    truncated: bool

    @property
    def columns(self) -> list[str]:
        return list(self.df.columns)


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql


def clean_sql(raw: str) -> str:
    """Normalise LLM output into a single executable statement."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:sql|duckdb)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    # Drop any prose before the actual statement.
    match = re.search(r"\b(select|with)\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start() :]
    return text.strip().rstrip(";").strip()


def validate_sql(raw: str) -> str:
    sql = clean_sql(raw)
    if not sql:
        raise UnsafeSQLError("SQL 为空")
    stripped = _strip_comments(sql)
    if not SINGLE_STATEMENT_ALLOWED.match(stripped):
        raise UnsafeSQLError("仅允许 SELECT / WITH 查询")
    if ";" in stripped.rstrip(";"):
        raise UnsafeSQLError("仅允许单条 SQL 语句")
    forbidden = FORBIDDEN_KEYWORDS.search(stripped)
    if forbidden:
        raise UnsafeSQLError(f"SQL 包含禁止的关键字或函数：{forbidden.group(0)}")
    return sql


def connect(db_path: Path | str, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    path = Path(db_path)
    if not path.exists():
        raise DatabaseNotReadyError(
            f"未找到本地数据库 {path}，请先运行 `python simulate_data.py`。"
        )
    con = duckdb.connect(str(path), read_only=read_only)
    if read_only:
        for statement in (
            "SET enable_external_access = false",
            "SET memory_limit = '512MB'",
            "SET threads = 2",
        ):
            try:
                con.execute(statement)
            except Exception:
                # Hardening is best-effort across DuckDB patch versions; the
                # read-only flag remains the primary guarantee.
                pass
    return con


def execute_query(
    db_path: Path | str,
    raw_sql: str,
    max_rows: int = 2000,
) -> QueryResult:
    """Execute one validated read-only statement and return a capped DataFrame."""
    sql = validate_sql(raw_sql)
    wrapped = f"SELECT * FROM (\n{sql}\n) AS _agent_result LIMIT {int(max_rows) + 1}"
    started = time.perf_counter()
    con = connect(db_path, read_only=True)
    try:
        df = con.execute(wrapped).fetchdf()
    finally:
        con.close()
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    truncated = len(df) > max_rows
    if truncated:
        df = df.head(max_rows).copy()
    return QueryResult(
        sql=sql,
        df=df,
        elapsed_ms=elapsed_ms,
        row_count=len(df),
        truncated=truncated,
    )


def schema_snapshot(db_path: Path | str) -> dict[str, Any]:
    """Tables, columns and row counts for the UI sidebar and prompts."""
    con = connect(db_path, read_only=True)
    try:
        rows = con.execute(
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'main'
            ORDER BY table_name, ordinal_position
            """
        ).fetchall()
        tables: dict[str, Any] = {}
        for table, column, dtype in rows:
            tables.setdefault(table, {"columns": [], "rows": 0})
            tables[table]["columns"].append({"name": column, "type": dtype})
        for table in tables:
            tables[table]["rows"] = con.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
    finally:
        con.close()
    return tables


def schema_ddl(db_path: Path | str) -> str:
    """Compact DDL used inside the Text-to-SQL prompt."""
    lines: list[str] = []
    for table, meta in schema_snapshot(db_path).items():
        cols = ",\n  ".join(f"{c['name']} {c['type']}" for c in meta["columns"])
        lines.append(
            f"CREATE TABLE {table} (\n  {cols}\n);  -- rows: {meta['rows']:,}"
        )
    return "\n\n".join(lines)


def preview_table(db_path: Path | str, table: str, limit: int = 5) -> pd.DataFrame:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise UnsafeSQLError("非法表名")
    return execute_query(
        db_path, f'SELECT * FROM "{table}" LIMIT {int(limit)}', max_rows=limit
    ).df
