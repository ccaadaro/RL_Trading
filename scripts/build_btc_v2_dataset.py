#!/usr/bin/env python3
"""
Build BTC dynamic-bar research dataset v2.

Inputs are dollar/volume bars built from trades. The output keeps the original
bar fields, adds orderflow/technical/cost features, and creates economic targets:
future long return net of estimated round-trip execution costs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.signal_features import build_feature_matrix, compute_microstructure_from_bars  # noqa: E402
from utils.slippage import add_slippage_feature  # noqa: E402


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        if "t_close" in df.columns:
            df["date"] = pd.to_datetime(df["t_close"], unit="ms", utc=True).dt.tz_localize(None)
        elif "end_ts" in df.columns:
            df["date"] = pd.to_datetime(df["end_ts"], utc=True).dt.tz_localize(None)
        else:
            raise ValueError("Bars need date, t_close, or end_ts.")

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    if "buy_volume" not in df.columns and "buy_vol" in df.columns:
        df["buy_volume"] = df["buy_vol"]
    if "sell_volume" not in df.columns and "sell_vol" in df.columns:
        df["sell_volume"] = df["sell_vol"]
    if "volume" not in df.columns:
        if {"buy_volume", "sell_volume"}.issubset(df.columns):
            df["volume"] = df["buy_volume"] + df["sell_volume"]
        else:
            raise ValueError("Bars need volume or buy/sell volume.")
    if "notional" not in df.columns:
        if "total_cost" in df.columns:
            df["notional"] = df["total_cost"]
        else:
            df["notional"] = df["close"] * df["volume"]
    if "cvd" not in df.columns and {"buy_volume", "sell_volume"}.issubset(df.columns):
        df["cvd"] = df["buy_volume"] - df["sell_volume"]
    if "aggressor_ratio" not in df.columns and {"buy_volume", "volume"}.issubset(df.columns):
        df["aggressor_ratio"] = df["buy_volume"] / df["volume"].replace(0, np.nan)
    df["aggressor_ratio"] = df["aggressor_ratio"].fillna(0.5).clip(0.0, 1.0)
    if "trade_count" not in df.columns:
        df["trade_count"] = 0
    return df


def add_dynamic_costs(
    df: pd.DataFrame,
    theta_usd: float,
    order_usd: float,
    fee_bps: float,
    spread_bps: float,
) -> pd.DataFrame:
    # Bars are dynamic, so estimate a 1-day ADV window from median bars/day.
    span_days = max((df["date"].iloc[-1] - df["date"].iloc[0]).total_seconds() / 86400.0, 1.0)
    bars_per_day = max(int(round(len(df) / span_days)), 4)
    df["bars_per_day_feature"] = float(bars_per_day)
    df["theta_usd_feature"] = float(theta_usd)

    add_slippage_feature(
        df,
        order_usd=order_usd,
        adv_window=bars_per_day,
        vol_window=bars_per_day,
        eta=0.1,
        spread_bps=spread_bps / 2.0,
        clip_max=50.0,
    )
    df["fee_bps_feature"] = float(fee_bps)
    df["estimated_oneway_cost_bps"] = df["fee_bps_feature"] + df["slippage_bps_feature"]
    df["estimated_roundtrip_cost_bps"] = 2.0 * df["estimated_oneway_cost_bps"]
    return df


def add_targets(
    df: pd.DataFrame,
    horizons: list[int],
    min_edge_bps: float,
) -> pd.DataFrame:
    close = df["close"].astype(float)
    log_ret = np.log(close).diff()
    bars_per_day = int(max(round(float(df["bars_per_day_feature"].iloc[0])), 4))
    daily_vol = log_ret.rolling(bars_per_day, min_periods=min(20, bars_per_day)).std() * np.sqrt(bars_per_day)
    df["daily_vol_feature"] = daily_vol.fillna(daily_vol.median()).fillna(0.02)

    for horizon in horizons:
        gross = close.shift(-horizon) / close - 1.0
        cost = df["estimated_roundtrip_cost_bps"] / 10_000.0
        net = gross - cost
        risk = df["daily_vol_feature"] * np.sqrt(max(horizon / bars_per_day, 1.0 / bars_per_day))
        df[f"future_return_{horizon}b"] = gross
        df[f"net_return_{horizon}b"] = net
        df[f"net_risk_adj_{horizon}b"] = net / risk.clip(lower=1e-6)
        df[f"target_tradeable_{horizon}b"] = (net > (min_edge_bps / 10_000.0)).astype("int8")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", required=True)
    parser.add_argument("--theta-usd", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--order-usd", type=float, default=5_000.0)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--spread-bps", type=float, default=1.0)
    parser.add_argument("--min-edge-bps", type=float, default=5.0)
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 6, 12, 24, 48])
    args = parser.parse_args()

    df = normalize_bars(feather.read_feather(args.bars))
    print(f"Loaded {len(df):,} bars from {args.bars}")
    print(f"Date range: {df['date'].iloc[0]} -> {df['date'].iloc[-1]}")

    # High-fidelity bar-derived orderflow features.
    compute_microstructure_from_bars(df, window=48)

    print("Computing signal feature matrix...")
    X = build_feature_matrix(df, eth_df=None, funding_series=None)
    for col in X.columns:
        df[col] = X[col].to_numpy()

    add_dynamic_costs(
        df,
        theta_usd=args.theta_usd,
        order_usd=args.order_usd,
        fee_bps=args.fee_bps,
        spread_bps=args.spread_bps,
    )
    add_targets(df, args.horizons, args.min_edge_bps)

    feature_cols = [c for c in df.columns if c.endswith("_feature")]
    target_cols = [c for c in df.columns if c.startswith(("target_tradeable_", "net_return_", "net_risk_adj_", "future_return_"))]
    warmup_subset = [c for c in feature_cols if c not in {"theta_usd_feature", "bars_per_day_feature"}]
    out = df.dropna(subset=warmup_subset + target_cols).reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(out, out_path)

    meta = {
        "bars": args.bars,
        "theta_usd": args.theta_usd,
        "output": str(out_path),
        "rows": len(out),
        "start": str(out["date"].iloc[0]) if len(out) else None,
        "end": str(out["date"].iloc[-1]) if len(out) else None,
        "feature_count": len(feature_cols),
        "target_columns": target_cols,
        "order_usd": args.order_usd,
        "fee_bps": args.fee_bps,
        "spread_bps": args.spread_bps,
        "min_edge_bps": args.min_edge_bps,
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {len(out):,} rows x {len(out.columns):,} cols -> {out_path}")
    for horizon in args.horizons:
        col = f"target_tradeable_{horizon}b"
        print(
            f"  {col}: positive={out[col].mean():.2%} "
            f"net_mean={out[f'net_return_{horizon}b'].mean()*10000:+.2f} bps"
        )


if __name__ == "__main__":
    main()
