#!/usr/bin/env python3
"""
scripts/add_orderflow_to_4h.py — Aggregate 15m order-flow features to 4h
and merge into the main 4h parquet.

Strategy
────────
15m bars labeled at T cover T→T+15m.
4h bars labeled at T cover T→T+4h (16 × 15m bars).
Resample with label="left", closed="left" to stay in sync with the parquet.

Aggregation rules
─────────────────
  cvd                : sum   (additive: total net buy volume in 4h)
  aggressor_ratio    : mean  (average directional bias)
  trade_count        : sum
  whale/large trades : sum
  max_trade_usd      : max
  vwap_skew          : mean
  whale_intensity / large_intensity : mean (already z-scored)

Output columns (added to the 4h parquet)
─────────────────────────────────────────
  cvd_4h_sum_trade_feature
  aggressor_ratio_4h_mean_trade_feature
  whale_trades_4h_sum_trade_feature
  large_trades_4h_sum_trade_feature
  max_trade_usd_4h_max_trade_feature
  vwap_skew_4h_mean_trade_feature
  whale_intensity_4h_mean_trade_feature
  large_intensity_4h_mean_trade_feature
  cvd_4h_zscore_trade_feature           (z-scored rolling 90-bar window)
  aggressor_vs_baseline_trade_feature   (aggressor_ratio - rolling 90-bar mean)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

PARQUET_4H  = "cache/data_v1_430340a861af5f8f9bcbd7a4ca16ba95.parquet"
PARQUET_15M = "cache/dataset_btc_15m_v1.parquet"

ROLL_W = 90   # 4h bars for z-score (~15 trading days)


def _roll_z(s: pd.Series, w: int) -> pd.Series:
    m = s.rolling(w, min_periods=max(2, w // 4)).mean()
    sd = s.rolling(w, min_periods=max(2, w // 4)).std().replace(0, np.nan)
    return ((s - m) / sd).fillna(0).astype(np.float32)


def main():
    print("  Loading 15m dataset …")
    df15 = pd.read_parquet(PARQUET_15M)

    raw_cols = [
        "cvd_trade_feature",
        "aggressor_ratio_trade_feature",
        "trade_count_trade_feature",
        "whale_trades_trade_feature",
        "large_trades_trade_feature",
        "max_trade_usd_trade_feature",
        "vwap_skew_trade_feature",
        "whale_intensity_trade_feature",
        "large_intensity_trade_feature",
    ]
    missing = [c for c in raw_cols if c not in df15.columns]
    if missing:
        raise SystemExit(f"Missing columns in 15m parquet: {missing}")

    # Resample 15m → 4h
    print("  Resampling 15m → 4h …")
    agg = df15[raw_cols].resample("4h", label="left", closed="left").agg({
        "cvd_trade_feature":               "sum",
        "aggressor_ratio_trade_feature":   "mean",
        "trade_count_trade_feature":       "sum",
        "whale_trades_trade_feature":      "sum",
        "large_trades_trade_feature":      "sum",
        "max_trade_usd_trade_feature":     "max",
        "vwap_skew_trade_feature":         "mean",
        "whale_intensity_trade_feature":   "mean",
        "large_intensity_trade_feature":   "mean",
    })

    rename = {
        "cvd_trade_feature":               "cvd_4h_sum_trade_feature",
        "aggressor_ratio_trade_feature":   "aggressor_ratio_4h_mean_trade_feature",
        "trade_count_trade_feature":       "trade_count_4h_sum_trade_feature",
        "whale_trades_trade_feature":      "whale_trades_4h_sum_trade_feature",
        "large_trades_trade_feature":      "large_trades_4h_sum_trade_feature",
        "max_trade_usd_trade_feature":     "max_trade_usd_4h_max_trade_feature",
        "vwap_skew_trade_feature":         "vwap_skew_4h_mean_trade_feature",
        "whale_intensity_trade_feature":   "whale_intensity_4h_mean_trade_feature",
        "large_intensity_trade_feature":   "large_intensity_4h_mean_trade_feature",
    }
    agg = agg.rename(columns=rename)

    # Derived z-scores / deviation signals
    agg["cvd_4h_zscore_trade_feature"] = _roll_z(
        agg["cvd_4h_sum_trade_feature"].fillna(0), ROLL_W
    )
    roll_mean_agg = agg["aggressor_ratio_4h_mean_trade_feature"].rolling(
        ROLL_W, min_periods=max(2, ROLL_W // 4)
    ).mean()
    agg["aggressor_vs_baseline_trade_feature"] = (
        (agg["aggressor_ratio_4h_mean_trade_feature"] - roll_mean_agg)
        .fillna(0).astype(np.float32)
    )

    # Forward-fill a couple of bars (edge of resampled window may be NaN)
    agg = agg.ffill(limit=2).fillna(0)
    for c in agg.columns:
        agg[c] = agg[c].astype(np.float32)

    print(f"  4h resampled: {len(agg)} rows  {agg.index[0]} → {agg.index[-1]}")
    print(f"  New columns : {list(agg.columns)}")

    print("  Loading 4h parquet …")
    df4 = pd.read_parquet(PARQUET_4H)
    print(f"  4h rows     : {len(df4)}  features: {len([c for c in df4.columns if 'feature' in c])}")

    # Drop old trade cols if already present (idempotent re-run)
    old = [c for c in agg.columns if c in df4.columns]
    if old:
        print(f"  Dropping {len(old)} existing orderflow cols (re-run).")
        df4 = df4.drop(columns=old)

    # Left-join: keeps all 4h rows, NaN where 15m data is absent (pre-2023-06)
    df4 = df4.join(agg, how="left")
    overlap = df4[list(agg.columns)].notna().any(axis=1).sum()
    print(f"  Rows with order-flow coverage: {overlap}/{len(df4)}")

    # Fill pre-coverage bars with 0 (no signal = neutral)
    df4[list(agg.columns)] = df4[list(agg.columns)].fillna(0)

    feat_count = len([c for c in df4.columns if "feature" in c])
    print(f"  Total feature columns after merge: {feat_count}")

    df4.to_parquet(PARQUET_4H)
    print(f"  Saved → {PARQUET_4H}")
    print(f"  Size: {Path(PARQUET_4H).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
