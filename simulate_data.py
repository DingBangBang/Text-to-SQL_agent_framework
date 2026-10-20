"""Generate the local DuckDB warehouse used by the analytics agent.

The script is fully deterministic (fixed seed) and creates:

* ``users``  — 20,000 registered users
* ``trades`` — 100,000 trades spread over the last 30 days

A deliberate on-chain slowdown is injected into the last 7 days so that
attribution questions such as "为什么上周链上交易量跌了？" have a real,
explainable signal to discover instead of noise.

Usage:
    python simulate_data.py                 # create/refresh data/analytics.duckdb
    python simulate_data.py --force         # overwrite an existing database
    python simulate_data.py --trades 20000  # smaller dataset for quick demos
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "analytics.duckdb"

TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
TIER_WEIGHTS = [0.55, 0.28, 0.13, 0.04]
SYMBOLS = ["BTC", "ETH", "SOL", "ARB", "OP", "USDC", "USDT"]
SYMBOL_WEIGHTS = [0.30, 0.25, 0.15, 0.08, 0.06, 0.08, 0.08]
CHAIN_TYPES = ["onchain", "cex", "defi"]
CHAIN_WEIGHTS = [0.34, 0.51, 0.15]
FEE_RATES = {"onchain": 0.0008, "cex": 0.0010, "defi": 0.0012}


def generate_users(n_users: int, rng: np.random.Generator, today: date) -> pd.DataFrame:
    """Create the user dimension table."""
    register_offsets = rng.integers(0, 365, size=n_users)
    return pd.DataFrame(
        {
            "user_id": np.arange(1, n_users + 1, dtype=np.int64),
            "register_date": [
                today - timedelta(days=int(offset)) for offset in register_offsets
            ],
            "tier_level": rng.choice(TIERS, size=n_users, p=TIER_WEIGHTS),
        }
    )


def generate_trades(
    n_trades: int,
    n_users: int,
    days: int,
    rng: np.random.Generator,
    today: date,
) -> pd.DataFrame:
    """Create the trade fact table with a believable on-chain slowdown.

    Trade dates are drawn from the last ``days`` days.  For the most recent
    7 days the on-chain share and on-chain notional are both reduced, which
    gives the attribution node a genuine, measurable driver to report.
    """
    day_offsets = np.arange(days - 1, -1, -1)  # oldest -> newest
    # Slight weekly seasonality: weekends are a bit quieter.
    day_weights = np.array(
        [
            0.88 if (today - timedelta(days=int(o))).weekday() >= 5 else 1.0
            for o in day_offsets
        ]
    )
    day_weights = day_weights / day_weights.sum()
    chosen_offsets = rng.choice(day_offsets, size=n_trades, p=day_weights)

    weekday = np.array(
        [(today - timedelta(days=int(o))).weekday() for o in chosen_offsets]
    )
    recent_mask = chosen_offsets < 7  # last 7 days

    # Chain mix: on-chain gets a haircut in the most recent week.
    chain_types = rng.choice(CHAIN_TYPES, size=n_trades, p=CHAIN_WEIGHTS)
    downgrade = (
        recent_mask & (chain_types == "onchain") & (rng.random(n_trades) < 0.35)
    )
    chain_types[downgrade] = "cex"

    # Notional: log-normal with whale tail, larger on weekdays.
    volume = rng.lognormal(mean=6.4, sigma=1.15, size=n_trades)
    volume *= np.where(weekday >= 5, 0.72, 1.0)
    volume *= np.where(recent_mask & (chain_types == "onchain"), 0.55, 1.0)
    volume = np.round(volume, 2)

    fee_rate = np.array([FEE_RATES[c] for c in chain_types])
    fee = np.round(volume * fee_rate, 4)

    return pd.DataFrame(
        {
            "trade_id": np.arange(1, n_trades + 1, dtype=np.int64),
            "user_id": rng.integers(1, n_users + 1, size=n_trades, dtype=np.int64),
            "trade_date": [
                today - timedelta(days=int(offset)) for offset in chosen_offsets
            ],
            "symbol": rng.choice(SYMBOLS, size=n_trades, p=SYMBOL_WEIGHTS),
            "trade_volume_usd": volume,
            "fee_usd": fee,
            "chain_type": chain_types,
        }
    )


def write_to_duckdb(db_path: Path, users: pd.DataFrame, trades: pd.DataFrame) -> None:
    """Materialise both tables in DuckDB (overwriting any previous snapshot)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE OR REPLACE TABLE users (
                user_id BIGINT PRIMARY KEY,
                register_date DATE NOT NULL,
                tier_level VARCHAR NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE trades (
                trade_id BIGINT PRIMARY KEY,
                user_id BIGINT NOT NULL,
                trade_date DATE NOT NULL,
                symbol VARCHAR NOT NULL,
                trade_volume_usd DOUBLE NOT NULL,
                fee_usd DOUBLE NOT NULL,
                chain_type VARCHAR NOT NULL
            )
            """
        )
        con.register("users_df", users)
        con.register("trades_df", trades)
        con.execute("INSERT INTO users SELECT * FROM users_df")
        con.execute("INSERT INTO trades SELECT * FROM trades_df")
        con.unregister("users_df")
        con.unregister("trades_df")
        con.execute("CHECKPOINT")
    finally:
        con.close()


def summarise(db_path: Path, days: int) -> None:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        n_users = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        n_trades = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        start, end = con.execute(
            "SELECT MIN(trade_date), MAX(trade_date) FROM trades"
        ).fetchone()
        onchain = con.execute(
            """
            SELECT
                SUM(CASE WHEN trade_date >= CURRENT_DATE - INTERVAL '6 days'
                         AND chain_type = 'onchain' THEN trade_volume_usd ELSE 0 END),
                SUM(CASE WHEN trade_date BETWEEN CURRENT_DATE - INTERVAL '13 days'
                         AND CURRENT_DATE - INTERVAL '7 days'
                         AND chain_type = 'onchain' THEN trade_volume_usd ELSE 0 END)
            FROM trades
            """
        ).fetchone()
    finally:
        con.close()

    delta = (onchain[0] / onchain[1] - 1) * 100 if onchain[1] else 0.0
    print("=" * 62)
    print(" DuckDB warehouse ready:", db_path)
    print(f" users           : {n_users:,}")
    print(f" trades          : {n_trades:,}")
    print(f" date range      : {start} -> {end} ({days} days)")
    print(f" on-chain last7  : ${onchain[0]:,.2f}")
    print(f" on-chain prev7  : ${onchain[1]:,.2f}  ({delta:+.1f}%)")
    print("=" * 62)




def build_database(
    db_path: Path = DEFAULT_DB,
    n_users: int = 20_000,
    n_trades: int = 100_000,
    days: int = 30,
    seed: int = 42,
    today: date | None = None,
) -> Path:
    """Generate both tables and write them to ``db_path``."""
    if n_trades <= 0 or n_users <= 0 or days <= 0:
        raise ValueError("n_users, n_trades and days must all be positive")
    today = today or date.today()
    rng = np.random.default_rng(seed)
    users = generate_users(n_users, rng, today)
    trades = generate_trades(n_trades, n_users, days, rng, today)
    write_to_duckdb(db_path, users, trades)
    return db_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--users", type=int, default=20_000)
    parser.add_argument("--trades", type=int, default=100_000)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force", action="store_true", help="overwrite the database if it exists"
    )
    args = parser.parse_args(argv)

    if args.db.exists() and not args.force:
        print(
            f"[skip] {args.db} already exists. Use --force to regenerate.",
            file=sys.stderr,
        )
        summarise(args.db, args.days)
        return 0

    build_database(
        db_path=args.db,
        n_users=args.users,
        n_trades=args.trades,
        days=args.days,
        seed=args.seed,
    )
    summarise(args.db, args.days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
