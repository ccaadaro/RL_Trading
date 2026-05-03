#!/usr/bin/env python3
"""
Extend cache/dollar_bars_btc_50000000.feather from 2025-05-31 to 2026-03-18
by downloading Binance spot aggTrade daily ZIPs and building incremental bars.

Usage:
    python scripts/extend_dollar_bars.py [--start 2025-05-31] [--end 2026-03-18] [--dry-run]
"""

import argparse
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import requests

THETA = 50_000_000.0
AGGTRADES_DIR = Path("/home/nosferatu/freqtrade/user_data/data/binance/aggTrades")
BARS_PATH = Path("cache/dollar_bars_btc_50000000.feather")
BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT"
COLS = ["id", "price", "qty", "first_id", "last_id", "ts_us", "is_buyer_maker", "is_best_match"]
OUT_COLS = ["date", "open", "high", "low", "close", "vwap",
            "buy_vol", "sell_vol", "cvd", "aggressor_ratio",
            "trade_count", "buy_count", "sell_count", "total_cost"]


def date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def download_zip(day: date, out_dir: Path) -> Path:
    fname = f"BTCUSDT-aggTrades-{day}.zip"
    path = out_dir / fname
    if path.exists():
        return path
    url = f"{BASE_URL}/{fname}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                path.write_bytes(r.content)
                return path
            print(f"  HTTP {r.status_code} for {day}, attempt {attempt+1}")
        except Exception as e:
            print(f"  Error downloading {day}: {e}, attempt {attempt+1}")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {day} after 3 attempts")


def load_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f, names=COLS, header=None)
    df["price"] = df["price"].astype(np.float64)
    df["qty"] = df["qty"].astype(np.float64)
    df["cost"] = df["price"] * df["qty"]
    df["ts_dt"] = pd.to_datetime(df["ts_us"], unit="us", utc=True).dt.tz_localize(None)
    is_agg_buy = ~df["is_buyer_maker"].astype(bool)
    df["buy_vol"] = np.where(is_agg_buy, df["qty"], 0.0)
    df["sell_vol"] = np.where(is_agg_buy, 0.0, df["qty"])
    df["buy_cost"] = np.where(is_agg_buy, df["cost"], 0.0)
    df["is_buy_int"] = is_agg_buy.astype(np.int32)
    return df


def build_bars(df: pd.DataFrame, leftover: float) -> tuple[list[dict], float]:
    cost = df["cost"].to_numpy(dtype=np.float64)
    c_sum = leftover + cost.cumsum()
    bar_ids = (c_sum // THETA).astype(np.int64)
    df = df.copy()
    df["bar_id"] = bar_ids

    rows = []
    for bid, grp in df.groupby("bar_id", sort=True):
        total_vol = grp["qty"].sum()
        buy_vol = grp["buy_vol"].sum()
        sell_vol = grp["sell_vol"].sum()
        total_cost = grp["cost"].sum()
        buy_count = int(grp["is_buy_int"].sum())
        trade_count = len(grp)
        vwap = total_cost / total_vol if total_vol > 0 else np.nan
        rows.append({
            "bar_id": int(bid),
            "date": grp["ts_dt"].iloc[-1],
            "open": grp["price"].iloc[0],
            "high": grp["price"].max(),
            "low": grp["price"].min(),
            "close": grp["price"].iloc[-1],
            "vwap": vwap,
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
            "cvd": buy_vol - sell_vol,
            "aggressor_ratio": buy_vol / total_vol if total_vol > 0 else 0.5,
            "trade_count": trade_count,
            "buy_count": buy_count,
            "sell_count": trade_count - buy_count,
            "total_cost": total_cost,
        })

    new_leftover = float(c_sum[-1] % THETA)
    return rows, new_leftover


def merge_bars(rows_by_id: dict, new_rows: list[dict]) -> None:
    for r in new_rows:
        bid = r["bar_id"]
        if bid not in rows_by_id:
            rows_by_id[bid] = r
        else:
            existing = rows_by_id[bid]
            existing["high"] = max(existing["high"], r["high"])
            existing["low"] = min(existing["low"], r["low"])
            existing["close"] = r["close"]
            existing["date"] = r["date"]
            for col in ["buy_vol", "sell_vol", "cvd", "trade_count",
                        "buy_count", "sell_count", "total_cost"]:
                existing[col] += r[col]
            total_vol = existing["buy_vol"] + existing["sell_vol"]
            existing["aggressor_ratio"] = (existing["buy_vol"] / total_vol
                                           if total_vol > 0 else 0.5)
            existing["vwap"] = (existing["total_cost"] / total_vol
                                if total_vol > 0 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-05-31")
    ap.add_argument("--end", default="2026-03-18")
    ap.add_argument("--dry-run", action="store_true",
                    help="Download and parse files but don't write output")
    args = ap.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    AGGTRADES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading existing bars from {BARS_PATH}")
    existing = pd.read_feather(BARS_PATH)
    print(f"  Existing: {len(existing):,} bars  last={existing['date'].max()}")

    days = list(date_range(start_date, end_date))
    print(f"\nProcessing {len(days)} days: {start_date} → {end_date}")

    leftover: float = 0.0
    rows_by_id: dict = {}
    last_bar_id_offset = -1

    t0 = time.time()
    for i, day in enumerate(days):
        try:
            zip_path = download_zip(day, AGGTRADES_DIR)
        except RuntimeError as e:
            print(f"  SKIP {day}: {e}")
            continue

        df = load_zip(zip_path)
        new_rows, leftover = build_bars(df, leftover)
        merge_bars(rows_by_id, new_rows)

        if (i + 1) % 30 == 0 or i == len(days) - 1:
            elapsed = time.time() - t0
            pct = (i + 1) / len(days) * 100
            eta = elapsed / (i + 1) * (len(days) - i - 1)
            print(f"  [{pct:5.1f}%] {day}  bars_so_far={len(rows_by_id):,}  "
                  f"leftover=${leftover/1e6:.1f}M  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    if not rows_by_id:
        print("No new bars built. Exiting.")
        return

    # Drop the last (potentially incomplete) bar
    max_bid = max(rows_by_id.keys())
    if rows_by_id[max_bid]["total_cost"] < THETA * 0.99:
        print(f"  Dropping incomplete last bar (id={max_bid}, "
              f"cost=${rows_by_id[max_bid]['total_cost']/1e6:.1f}M)")
        del rows_by_id[max_bid]

    new_df = pd.DataFrame(sorted(rows_by_id.values(), key=lambda x: x["bar_id"]))[OUT_COLS]
    print(f"\nNew bars: {len(new_df):,}  "
          f"range={new_df['date'].min()} → {new_df['date'].max()}")

    combined = pd.concat([existing[OUT_COLS], new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    print(f"Combined: {len(combined):,} bars  "
          f"range={combined['date'].min()} → {combined['date'].max()}")

    if args.dry_run:
        print("DRY RUN — not writing output")
        return

    backup = BARS_PATH.with_suffix(".bak.feather")
    import shutil
    shutil.copy2(BARS_PATH, backup)
    print(f"Backup → {backup}")

    feather.write_feather(combined, str(BARS_PATH))
    print(f"Written → {BARS_PATH}")


if __name__ == "__main__":
    main()
