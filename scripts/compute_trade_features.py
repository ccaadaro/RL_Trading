#!/usr/bin/env python3
"""
scripts/compute_trade_features.py — Aggregate 27GB tick trades into
15m-aligned order-flow features.

Input
─────
`/home/nosferatu/freqtrade/user_data/data/binance/BTC_USDT-trades.feather`
Each row: timestamp(ms), id, type, side (buy/sell = aggressor), price,
amount, cost. Covers 2023-05-31 → 2025-05-30.

What "side" means here
──────────────────────
For Binance aggTrades, `side='buy'` means the MAKER is sell-side and the
TAKER bought — i.e. aggressive buy. So `side='buy'` → aggressor buy.
This is the sign-convention for CVD (cumulative volume delta).

Output columns (15m-bar aligned)
────────────────────────────────
  buy_volume          : taker-buy amount (coins)
  sell_volume         : taker-sell amount
  cvd                 : buy_volume - sell_volume
  aggressor_ratio     : buy_volume / (buy_volume + sell_volume)  [0..1]
  trade_count         : total trades in bar
  buy_count / sell_count
  vwap                : overall VWAP
  vwap_buy / vwap_sell: side-specific VWAPs (empty if 0 trades that side)
  max_trade_usd       : largest single trade notional
  n_large_trades      : count of trades > $100k notional (whales)
  n_whale_trades      : count of trades > $1M notional

Usage
─────
    python scripts/compute_trade_features.py \
        --bucket 15min \
        --out cache/trade_features_15m.feather
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather


TRADES_PATH = "/home/nosferatu/freqtrade/user_data/data/binance/BTC_USDT-trades.feather"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades",  type=str, default=TRADES_PATH)
    ap.add_argument("--bucket",  type=str, default="15min",
                    help="Pandas offset alias (15min, 5min, 1h, …).")
    ap.add_argument("--out",     type=str, default="cache/trade_features_15m.feather")
    ap.add_argument("--whale-usd", type=float, default=1_000_000.0)
    ap.add_argument("--large-usd", type=float, default=100_000.0)
    args = ap.parse_args()

    print(f"  Reading: {args.trades}  bucket={args.bucket}")
    reader = pa.ipc.RecordBatchFileReader(pa.memory_map(args.trades))
    n_batches = reader.num_record_batches
    print(f"  Batches: {n_batches:,}")

    # Per-bucket running aggregates — list of partial frames, concatenated later.
    # Using pandas dict accum to minimise DataFrame creation overhead.
    running = {}   # bucket_ts -> dict of summed stats

    t0 = time.time()
    last_report = t0
    total_rows = 0

    for batch_idx in range(n_batches):
        b = reader.get_batch(batch_idx).to_pandas()

        # Assign bucket timestamp per trade
        b["ts"]     = pd.to_datetime(b["timestamp"], unit="ms", utc=True)
        b["bucket"] = b["ts"].dt.floor(args.bucket)

        # Pre-split amounts by aggressor side — vectorised, no lambda.
        is_buy = (b["side"] == "buy").to_numpy()
        amt    = b["amount"].to_numpy()
        cost   = b["cost"].to_numpy()
        b["buy_vol"]  = np.where(is_buy, amt,  0.0)
        b["sell_vol"] = np.where(is_buy, 0.0,  amt)
        b["buy_cost"]  = np.where(is_buy, cost, 0.0)
        b["sell_cost"] = np.where(is_buy, 0.0, cost)
        b["is_buy"]   = is_buy.astype(np.uint32)
        b["is_large"] = (cost > args.large_usd).astype(np.uint32)
        b["is_whale"] = (cost > args.whale_usd).astype(np.uint32)

        # Groupby+sum — fast.
        agg = b.groupby("bucket", sort=False).agg(
            buy_vol      = ("buy_vol",  "sum"),
            sell_vol     = ("sell_vol", "sum"),
            buy_cost     = ("buy_cost", "sum"),
            sell_cost    = ("sell_cost","sum"),
            trade_count  = ("amount",   "size"),
            buy_count    = ("is_buy",   "sum"),
            total_cost   = ("cost",     "sum"),
            max_cost     = ("cost",     "max"),
            n_large      = ("is_large", "sum"),
            n_whale      = ("is_whale", "sum"),
        )

        # Merge into running dict (batches can straddle bucket boundaries)
        for idx, row in agg.iterrows():
            if idx not in running:
                running[idx] = {
                    "buy_vol": 0.0, "sell_vol": 0.0,
                    "buy_cost": 0.0, "sell_cost": 0.0,
                    "trade_count": 0, "buy_count": 0,
                    "total_cost": 0.0, "max_cost": 0.0,
                    "n_large": 0, "n_whale": 0,
                }
            r = running[idx]
            r["buy_vol"]     += row["buy_vol"]
            r["sell_vol"]    += row["sell_vol"]
            r["buy_cost"]    += row["buy_cost"]
            r["sell_cost"]   += row["sell_cost"]
            r["trade_count"] += int(row["trade_count"])
            r["buy_count"]   += int(row["buy_count"])
            r["total_cost"]  += row["total_cost"]
            r["max_cost"]    = max(r["max_cost"], row["max_cost"])
            r["n_large"]     += int(row["n_large"])
            r["n_whale"]     += int(row["n_whale"])

        total_rows += len(b)
        if time.time() - last_report > 20:
            elapsed = time.time() - t0
            pct = (batch_idx + 1) / n_batches * 100
            eta = elapsed * (n_batches / (batch_idx + 1) - 1)
            print(f"  [{pct:5.1f}%] batch {batch_idx+1:>6,}/{n_batches:,}  "
                  f"rows={total_rows/1e6:.1f}M  buckets={len(running):,}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
            last_report = time.time()

    print(f"\n  Total trades processed : {total_rows:,}")
    print(f"  Unique {args.bucket} buckets: {len(running):,}")
    print(f"  Total time             : {time.time()-t0:.1f}s")

    # ── Build DataFrame and derived features ──────────────────────────────
    df = pd.DataFrame.from_dict(running, orient="index").sort_index()
    df.index.name = "date"
    df["sell_count"]      = df["trade_count"] - df["buy_count"]
    df["cvd"]             = df["buy_vol"] - df["sell_vol"]
    total_vol             = df["buy_vol"] + df["sell_vol"]
    df["aggressor_ratio"] = (df["buy_vol"] / total_vol.replace(0, np.nan)).fillna(0.5)
    df["vwap"]            = df["total_cost"] / total_vol.replace(0, np.nan)
    df["vwap_buy"]        = (df["buy_cost"]  / df["buy_vol"].replace(0, np.nan))
    df["vwap_sell"]       = (df["sell_cost"] / df["sell_vol"].replace(0, np.nan))
    df["avg_trade_usd"]   = df["total_cost"] / df["trade_count"].replace(0, np.nan)

    # Rename output columns to the `*_trade_feature` convention so they
    # pick up automatically in the build_feature_matrix pipeline.
    rename = {
        "buy_vol":      "taker_buy_vol_trade_feature",
        "sell_vol":     "taker_sell_vol_trade_feature",
        "cvd":          "cvd_trade_feature",
        "aggressor_ratio":"aggressor_ratio_trade_feature",
        "trade_count":  "trade_count_trade_feature",
        "buy_count":    "buy_count_trade_feature",
        "sell_count":   "sell_count_trade_feature",
        "max_cost":     "max_trade_usd_trade_feature",
        "n_large":      "large_trades_trade_feature",
        "n_whale":      "whale_trades_trade_feature",
        "vwap":         "vwap_trade_feature",
        "vwap_buy":     "vwap_buy_trade_feature",
        "vwap_sell":    "vwap_sell_trade_feature",
        "avg_trade_usd":"avg_trade_usd_trade_feature",
    }
    df = df.rename(columns=rename)
    # Drop cost-scratch cols (not features themselves)
    df = df.drop(columns=["buy_cost", "sell_cost", "total_cost"], errors="ignore")

    # Ensure datetime-indexed (tz-aware → tz-naive to match parquet index)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Feather requires a RangeIndex → reset + explicit column
    df_out = df.reset_index()
    feather.write_feather(df_out, str(out_path))
    print(f"\n  Wrote {len(df):,} rows × {len(df.columns)} features → {out_path}")
    print(f"  Columns:")
    for c in df.columns:
        print(f"    {c:40s}  mean={df[c].mean():.3f}  std={df[c].std():.3f}")


if __name__ == "__main__":
    main()
