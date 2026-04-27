#!/usr/bin/env python3
"""Download Binance FAPI market microstructure data for RL training.

Data sources:
  - Open Interest history      (/futures/data/openInterestHist)
  - Top-trader L/S ratio       (/futures/data/topLongShortPositionRatio)
  - Global (retail) L/S ratio  (/futures/data/globalLongShortAccountRatio)

Usage:
    python utils/download_binance_futures_data.py
    python utils/download_binance_futures_data.py --symbol BTCUSDT --days 1500 --period 4h

Files are saved to:
    <data_root>/binance/futures/BTC_USDT_USDT-4h-open_interest.parquet
    <data_root>/binance/futures/BTC_USDT_USDT-4h-top_ls_ratio.parquet
    <data_root>/binance/futures/BTC_USDT_USDT-4h-global_ls_ratio.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

BASE_URL = "https://fapi.binance.com"
DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data"
FUTURES_DIR = DATA_ROOT / "binance" / "futures"

PERIOD_MS = {
    "5m":  5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h":  60 * 60 * 1000,
    "2h":  2 * 60 * 60 * 1000,
    "4h":  4 * 60 * 60 * 1000,
    "6h":  6 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d":  24 * 60 * 60 * 1000,
}


def _fetch_simple(endpoint: str, symbol: str, period: str) -> List[dict]:
    """Fetch available data from a Binance FAPI stats endpoint.

    NOTE: Binance retains only ~28 days for OI/L-S endpoints.
    Passing startTime/endTime causes 400 errors; use limit=500 only.
    Run this script daily via cron to accumulate a rolling dataset.
    """
    try:
        resp = requests.get(
            f"{BASE_URL}{endpoint}",
            params={"symbol": symbol, "period": period, "limit": 500},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  ✗ Request failed: {e}")
        return []


def _merge_with_existing(new_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Merge new rows with an existing parquet file, dedup by index."""
    if not path.exists():
        return new_df
    try:
        old = pd.read_parquet(path)
        combined = pd.concat([old, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        return combined
    except Exception as e:
        print(f"  ⚠ Could not merge with existing file ({e}), overwriting")
        return new_df


def download_open_interest(symbol: str = "BTCUSDT", period: str = "4h", days: int = 30) -> pd.DataFrame:
    """Download open interest (Binance retains ~28 days; accumulate via cron)."""
    print(f"📥 Downloading Open Interest ({symbol}, {period})…")
    rows = _fetch_simple("/futures/data/openInterestHist", symbol, period)
    if not rows:
        print("  ✗ No data returned")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True).dt.tz_localize(None)
    df = df.set_index("date").sort_index()
    df["oi_usd"] = df["sumOpenInterestValue"].astype(float)
    df = df[["oi_usd"]].copy()
    print(f"  ✅ {len(df)} rows  [{df.index[0].date()} → {df.index[-1].date()}]")
    return df


def download_ls_ratio(
    symbol: str = "BTCUSDT",
    period: str = "4h",
    days: int = 30,
    endpoint: str = "/futures/data/topLongShortPositionRatio",
    label: str = "top",
) -> pd.DataFrame:
    """Download long/short ratio (Binance retains ~28 days; accumulate via cron)."""
    print(f"📥 Downloading L/S ratio [{label}] ({symbol}, {period})…")
    rows = _fetch_simple(endpoint, symbol, period)
    if not rows:
        print("  ✗ No data returned")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True).dt.tz_localize(None)
    df = df.set_index("date").sort_index()
    df[f"{label}_ls_ratio"] = df["longShortRatio"].astype(float)
    df[f"{label}_long_pct"]  = df["longAccount"].astype(float)
    df[f"{label}_short_pct"] = df["shortAccount"].astype(float)
    df = df[[f"{label}_ls_ratio", f"{label}_long_pct", f"{label}_short_pct"]].copy()
    print(f"  ✅ {len(df)} rows  [{df.index[0].date()} → {df.index[-1].date()}]")
    return df


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print(f"  💾 Saved → {path}")


def main(symbol: str = "BTCUSDT", period: str = "4h", days: int = 30) -> None:
    pair_slug = symbol.replace("USDT", "_USDT_USDT")  # BTCUSDT → BTC_USDT_USDT
    prefix = FUTURES_DIR / f"{pair_slug}-{period}"

    # Open Interest — merge with existing to accumulate history over time
    oi_path = prefix.with_name(f"{prefix.name}-open_interest.parquet")
    oi_df = download_open_interest(symbol, period, days)
    if not oi_df.empty:
        oi_df = _merge_with_existing(oi_df, oi_path)
        save_parquet(oi_df, oi_path)
        print(f"  Total accumulated OI rows: {len(oi_df)}")

    # Top-trader L/S ratio (position-based, institutional)
    top_path = prefix.with_name(f"{prefix.name}-top_ls_ratio.parquet")
    top_ls = download_ls_ratio(symbol, period, days, "/futures/data/topLongShortPositionRatio", "top")
    if not top_ls.empty:
        top_ls = _merge_with_existing(top_ls, top_path)
        save_parquet(top_ls, top_path)
        print(f"  Total accumulated Top L/S rows: {len(top_ls)}")

    # Global (retail) L/S ratio (account-based)
    global_path = prefix.with_name(f"{prefix.name}-global_ls_ratio.parquet")
    global_ls = download_ls_ratio(symbol, period, days, "/futures/data/globalLongShortAccountRatio", "global")
    if not global_ls.empty:
        global_ls = _merge_with_existing(global_ls, global_path)
        save_parquet(global_ls, global_path)
        print(f"  Total accumulated Global L/S rows: {len(global_ls)}")

    print("\n✅ All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance FAPI market microstructure data")
    parser.add_argument("--symbol", default="BTCUSDT", help="Futures symbol (default: BTCUSDT)")
    parser.add_argument("--period", default="4h", choices=list(PERIOD_MS.keys()), help="Candle period")
    parser.add_argument("--days",   default=1500, type=int, help="History length in days")
    args = parser.parse_args()
    main(args.symbol, args.period, args.days)
