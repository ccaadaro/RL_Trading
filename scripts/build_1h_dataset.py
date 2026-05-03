#!/usr/bin/env python3
"""
Build Phase 9 BTC 1h research dataset.

Architecture: 1h OHLCV as the decision clock, microstructure aggregated from
raw dollar bars as a risk/confirmation layer.

Three target families:
  1. Trend participation  — forward_ret_Nh > cost + buffer
  2. Triple barrier       — TP/SL/vertical hit detection
  3. Regime participation — directional + path-risk filtered

Usage:
    python scripts/build_1h_dataset.py \
        --candles /home/nosferatu/freqtrade/user_data/data/binance/BTC_USDT-1h.feather \
        --bars    cache/dollar_bars_btc_50000000.feather \
        --output  cache/btc_1h_phase9.feather \
        --start   2021-01-01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather


# ── Hull / RSI helpers ────────────────────────────────────────────────────────

def _wma(s: pd.Series, n: int) -> pd.Series:
    weights = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def _hma(s: pd.Series, n: int) -> pd.Series:
    half = max(n // 2, 1)
    sqrt_n = max(int(round(n**0.5)), 1)
    return _wma(2.0 * _wma(s, half) - _wma(s, n), sqrt_n)


def _rsi(s: pd.Series, n: int) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


# ── Trend features ────────────────────────────────────────────────────────────

def add_trend_features(df: pd.DataFrame) -> None:
    """Compute all trend + OHLCV-derived features in-place."""
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    vol   = df["volume"].astype(float)
    log_ret = np.log(close).diff()

    # Multi-horizon price returns
    for h in [1, 3, 6, 12, 24]:
        df[f"return_{h}h_feature"] = close.pct_change(h)

    # MA bias: close / SMA(n) − 1
    for n in [12, 24, 48, 96, 200]:
        ma = close.rolling(n, min_periods=n // 2).mean()
        df[f"ma_bias_{n}h_feature"] = close / ma.replace(0.0, np.nan) - 1.0

    # EMA bias and 3-bar slope
    for n in [12, 48, 96]:
        ema = close.ewm(span=n, adjust=False).mean()
        df[f"ema_bias_{n}h_feature"]  = close / ema.replace(0.0, np.nan) - 1.0
        df[f"ema_slope_{n}h_feature"] = ema.diff(3) / ema.shift(3).replace(0.0, np.nan)

    # Hull MA (48h) — position and slope
    hma = _hma(close, 48)
    df["hma_48h_pos_feature"]   = close / hma.replace(0.0, np.nan) - 1.0
    df["hma_48h_slope_feature"] = hma.diff(6) / hma.shift(6).replace(0.0, np.nan)

    # Realized volatility (annualized, 8760 h/year)
    for h in [24, 72, 168]:
        rv = log_ret.rolling(h, min_periods=h // 2).std() * np.sqrt(8760)
        df[f"realized_vol_{h}h_feature"] = rv

    # Volatility compression: short vol / long vol
    vol_24h  = log_ret.rolling(24,  min_periods=12).std()
    vol_168h = log_ret.rolling(168, min_periods=48).std().replace(0.0, np.nan)
    df["vol_compression_feature"] = vol_24h / vol_168h

    # RSI
    for n in [14, 42]:
        df[f"rsi_{n}h_feature"] = _rsi(close, n)

    # ATR%
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_pct_14h_feature"] = tr.rolling(14).mean() / close.replace(0.0, np.nan)

    # Distance to rolling high / low
    for h in [24, 72, 168]:
        df[f"dist_to_high_{h}h_feature"] = close / high.rolling(h).max().replace(0.0, np.nan) - 1.0
        df[f"dist_to_low_{h}h_feature"]  = close / low.rolling(h).min().replace(0.0, np.nan)  - 1.0

    # Drawdown from 168h running high
    df["drawdown_168h_feature"] = close / close.rolling(168).max().replace(0.0, np.nan) - 1.0

    # Volume z-score
    for h in [24, 168]:
        vmean = vol.rolling(h, min_periods=h // 2).mean()
        vstd  = vol.rolling(h, min_periods=h // 2).std().replace(0.0, np.nan)
        df[f"volume_zscore_{h}h_feature"] = (vol - vmean) / vstd

    # Bollinger band position and width
    for n in [24, 96]:
        ma  = close.rolling(n, min_periods=n // 2).mean()
        std = close.rolling(n, min_periods=n // 2).std().replace(0.0, np.nan)
        bb_upper = ma + 2.0 * std
        bb_lower = ma - 2.0 * std
        bb_width = bb_upper - bb_lower
        df[f"bb_pos_{n}h_feature"]   = (close - bb_lower) / bb_width.replace(0.0, np.nan)
        df[f"bb_width_{n}h_feature"] = bb_width / ma.replace(0.0, np.nan)


# ── Microstructure aggregation from dollar bars ───────────────────────────────

def aggregate_microstructure(bars_path: str, index_1h: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Resample raw $50M dollar bars to 1h microstructure features aligned to
    the 1h candle index (bar open convention: timestamp = start of hour).
    """
    bars = feather.read_feather(bars_path)
    bars["date"] = pd.to_datetime(bars["date"], utc=True).dt.tz_localize(None)
    bars = bars.sort_values("date").set_index("date")

    # Resample: bin [T, T+1h) → label T  (matches freqtrade 1h open convention)
    rs = bars.resample("1h", closed="left", label="left")

    # --- CVD ---
    cvd_1h = rs["cvd"].sum()
    out = pd.DataFrame(index=cvd_1h.index)
    out["cvd_1h_feature"]        = cvd_1h
    out["cvd_4h_feature"]        = cvd_1h.rolling(4,  min_periods=1).sum()
    out["cvd_8h_feature"]        = cvd_1h.rolling(8,  min_periods=1).sum()
    out["cvd_slope_4h_feature"]  = cvd_1h.rolling(4,  min_periods=2).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
    )
    cvd_mean = out["cvd_1h_feature"].rolling(24, min_periods=8).mean()
    cvd_std  = out["cvd_1h_feature"].rolling(24, min_periods=8).std().replace(0.0, np.nan)
    out["cvd_zscore_24h_feature"] = (cvd_1h - cvd_mean) / cvd_std

    # --- Aggressor ratio (notional-weighted) ---
    notional = rs["total_cost"].sum()
    agg_notional = (bars["aggressor_ratio"] * bars["total_cost"]).resample(
        "1h", closed="left", label="left"
    ).sum()
    agg_ratio_1h = agg_notional / notional.replace(0.0, np.nan)
    out["aggressor_ratio_1h_feature"] = agg_ratio_1h
    out["aggressor_ratio_4h_feature"] = agg_ratio_1h.rolling(4, min_periods=1).mean()

    # --- Trade counts ---
    trade_count = rs["trade_count"].sum()
    buy_count   = rs["buy_count"].sum()
    sell_count  = rs["sell_count"].sum()
    total_count = (buy_count + sell_count).replace(0.0, np.nan)
    out["trade_count_1h_feature"] = trade_count
    out["buy_ratio_1h_feature"]   = buy_count / total_count

    # --- Notional volume z-score ---
    out["notional_1h_feature"] = notional
    n_mean = notional.rolling(24, min_periods=8).mean()
    n_std  = notional.rolling(24, min_periods=8).std().replace(0.0, np.nan)
    out["notional_zscore_24h_feature"] = (notional - n_mean) / n_std

    # Align to the 1h candle index
    return out.reindex(index_1h)


# ── Target families ───────────────────────────────────────────────────────────

def add_trend_participation_targets(
    df: pd.DataFrame,
    horizons_h: list[int],
    cost_bps: float,
    buffers_bps: dict[int, float],
) -> None:
    """Label 1 if forward_return_Nh > roundtrip_cost + buffer."""
    close = df["close"].astype(float)
    threshold_per_h = {
        h: (cost_bps + buffers_bps.get(h, 75.0)) / 10_000.0
        for h in horizons_h
    }
    for h in horizons_h:
        fwd = close.shift(-h) / close - 1.0
        df[f"future_return_{h}h"] = fwd
        df[f"target_trend_{h}h"]  = (fwd > threshold_per_h[h]).astype("int8")
        pos = df[f"target_trend_{h}h"].mean()
        print(f"    target_trend_{h}h:  pos_rate={pos:.2%}  threshold={threshold_per_h[h]*10000:.0f} bps")


def add_triple_barrier_targets(
    df: pd.DataFrame,
    configs: list[dict],
) -> None:
    """
    Vectorized triple barrier labeling. For each config (tp_pct, sl_pct, horizon_h):
      label 1 = TP hit before SL within horizon
      label 0 = SL hit first or vertical barrier reached
      label NaN = last `horizon_h` bars (lookahead unavailable)

    The inner loop runs `horizon_h` times over vectorized pandas ops — O(n × h),
    much faster than an O(n × h) Python loop.
    """
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    for cfg in configs:
        tp = cfg["tp_pct"] / 100.0
        sl = cfg["sl_pct"] / 100.0
        h  = cfg["horizon_h"]
        not_hit = float(h + 1)

        tp_lag = pd.Series(not_hit, index=close.index, dtype=float)
        sl_lag = pd.Series(not_hit, index=close.index, dtype=float)

        for lag in range(1, h + 1):
            # Relative price at lag bars ahead
            tp_hit = (high.shift(-lag) / close - 1.0) >= tp
            sl_hit = (low.shift(-lag)  / close - 1.0) <= -sl
            # Record first hit: keep old value unless this is the first crossing
            tp_lag = tp_lag.where(~(tp_hit & (tp_lag >= not_hit)), other=float(lag))
            sl_lag = sl_lag.where(~(sl_hit & (sl_lag >= not_hit)), other=float(lag))

        labels = ((tp_lag < sl_lag) & (tp_lag <= h)).astype(float)
        labels.iloc[-h:] = np.nan

        col = f"target_barrier_{cfg['tp_pct']:.1f}tp_{cfg['sl_pct']:.1f}sl_{h}h"
        df[col] = labels.astype("float32")
        pos = labels.dropna().mean()
        print(f"    {col}: pos_rate={pos:.2%}")


def add_regime_targets(
    df: pd.DataFrame,
    horizons_h: list[int],
    min_fwd_ret_pct: float = 0.5,
    max_path_dd_pct: float = 2.0,
) -> None:
    """
    Regime participation: 1 if the next N hours have positive direction
    AND the path drawdown stays within tolerance.

    Vectorized using reverse rolling min/max — no Python loops over bars.
    """
    close = df["close"].astype(float)
    low   = df["low"].astype(float)

    for h in horizons_h:
        fwd_ret = close.shift(-h) / close - 1.0

        # Forward path minimum low over (i+1, i+h] bars:
        # Shift low by 1 (so index i contains low[i+1]), then reverse-roll min over h.
        fwd_path_low = low.shift(-1).iloc[::-1].rolling(h, min_periods=1).min().iloc[::-1]
        path_dd = fwd_path_low / close - 1.0

        good = (
            (fwd_ret   > min_fwd_ret_pct / 100.0) &
            (path_dd   > -max_path_dd_pct / 100.0)
        )
        labels = good.astype(float)
        labels[fwd_ret.isna()] = np.nan

        col = f"target_regime_{h}h"
        df[col] = labels.astype("float32")
        pos = labels.dropna().mean()
        print(f"    {col}: pos_rate={pos:.2%}  (fwd>{min_fwd_ret_pct}%, dd>{-max_path_dd_pct}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 9 BTC 1h research dataset")
    parser.add_argument("--candles", required=True,
                        help="Path to 1h OHLCV feather (freqtrade format)")
    parser.add_argument("--bars",
                        help="Path to raw dollar bar feather for microstructure aggregation")
    parser.add_argument("--output", required=True, help="Output feather path")
    parser.add_argument("--start",  default="2021-01-01", help="Start date (inclusive)")
    parser.add_argument("--end",    help="End date (inclusive)")
    parser.add_argument("--fee-bps", type=float, default=7.5,
                        help="Estimated round-trip cost in bps (default=7.5 → taker+taker)")
    args = parser.parse_args()

    # ── Load 1h candles ──────────────────────────────────────────────────────
    df = feather.read_feather(args.candles)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    print(f"Loaded {len(df):,} 1h candles  {df['date'].iloc[0]} → {df['date'].iloc[-1]}")

    if args.start:
        df = df[df["date"] >= pd.Timestamp(args.start)]
    if args.end:
        df = df[df["date"] <= pd.Timestamp(args.end)]
    df = df.reset_index(drop=True)
    print(f"Filtered to {len(df):,} rows  {df['date'].iloc[0]} → {df['date'].iloc[-1]}")

    # ── Trend features ───────────────────────────────────────────────────────
    print("Computing trend features...")
    add_trend_features(df)

    # ── Microstructure aggregation ───────────────────────────────────────────
    ms_start = None
    if args.bars:
        print(f"Aggregating microstructure from {args.bars}...")
        idx_1h = pd.DatetimeIndex(df["date"])
        ms = aggregate_microstructure(args.bars, idx_1h)
        ms_cols_added = 0
        for col in ms.columns:
            df[col] = ms[col].values
            ms_cols_added += 1
        # Record when microstructure data actually starts
        ms_start_mask = ms.notna().any(axis=1)
        if ms_start_mask.any():
            ms_start = str(ms.index[ms_start_mask][0])
        print(f"  Added {ms_cols_added} microstructure features  (data from {ms_start})")

    # ── Targets ──────────────────────────────────────────────────────────────
    print("Computing targets...")

    print("  Trend participation:")
    add_trend_participation_targets(
        df,
        horizons_h=[24, 48, 72],
        cost_bps=args.fee_bps,
        buffers_bps={24: 75, 48: 125, 72: 175},
    )

    print("  Triple barrier:")
    add_triple_barrier_targets(df, configs=[
        {"tp_pct": 1.5, "sl_pct": 0.8,  "horizon_h": 24},
        {"tp_pct": 2.5, "sl_pct": 1.2,  "horizon_h": 48},
        {"tp_pct": 3.5, "sl_pct": 1.5,  "horizon_h": 72},
    ])

    print("  Regime participation:")
    add_regime_targets(df, horizons_h=[48, 72], min_fwd_ret_pct=0.5, max_path_dd_pct=2.0)

    # ── Save ─────────────────────────────────────────────────────────────────
    feature_cols = sorted(c for c in df.columns if c.endswith("_feature"))
    target_cols  = sorted(
        c for c in df.columns
        if c.startswith(("target_", "future_return_"))
    )

    # Drop warmup rows where the slowest feature (200h MA) hasn't stabilized
    warmup_col = "ma_bias_200h_feature"
    if warmup_col in df.columns:
        df = df.dropna(subset=[warmup_col]).reset_index(drop=True)
        print(f"Dropped warmup rows; keeping {len(df):,} rows from {df['date'].iloc[0]}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(df, out_path)

    meta = {
        "candles":          args.candles,
        "bars":             args.bars,
        "start":            str(df["date"].iloc[0]),
        "end":              str(df["date"].iloc[-1]),
        "rows":             len(df),
        "feature_count":    len(feature_cols),
        "target_cols":      target_cols,
        "microstructure_start": ms_start,
        "fee_bps":          args.fee_bps,
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nWrote {len(df):,} rows × {len(df.columns)} cols → {out_path}")
    print(f"Features: {len(feature_cols)}   Targets: {len(target_cols)}")
    print("\nTarget summary:")
    for col in target_cols:
        if df[col].notna().any() and col.startswith("target_"):
            pos = df[col].dropna().mean()
            n   = df[col].notna().sum()
            print(f"  {col:<45s}  pos={pos:.2%}  n={n:,}")


if __name__ == "__main__":
    main()
