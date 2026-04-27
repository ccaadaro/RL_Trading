#!/usr/bin/env python3
"""
scripts/build_dataset_15m.py — Assemble the 15m BTC/USDT dataset with
OHLCV + order-flow features + existing futures data.

Inputs
──────
  - /home/nosferatu/freqtrade/user_data/data/binance/BTC_USDT-5m.feather
  - cache/trade_features_15m.feather   (from compute_trade_features.py)
  - /home/nosferatu/freqtrade/user_data/data/binance/futures/*.feather
  - (optional) ETH 5m for cross-asset features
  - (optional) 8h funding rate

Output
──────
  cache/dataset_btc_15m_v1.parquet
    OHLCV (resampled 5m→15m) + all `*_feature` columns the pipeline
    downstream expects, plus the new `*_trade_feature` columns.

Timeframe
─────────
Trade data bounds the dataset: 2023-05-31 → 2025-05-30 (~2 years, 70k
15m bars). Pre-2023-05 5m OHLCV gets dropped because we can't produce
order-flow features there.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


OHLCV_5M_PATH = "/home/nosferatu/freqtrade/user_data/data/binance/BTC_USDT-5m.feather"
ETH_5M_PATH   = None   # add if/when ETH 5m is downloaded
TRADE_FEATS_PATH = "cache/trade_features_15m.feather"
FUTURES_DIR   = "/home/nosferatu/freqtrade/user_data/data/binance/futures"
OUT_PARQUET   = "cache/dataset_btc_15m_v1.parquet"


def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def load_ohlcv_5m(path: str) -> pd.DataFrame:
    df = pd.read_feather(path)
    df = df.set_index("date").sort_index()
    df = _strip_tz(df)
    # Downcast to float32 to halve memory
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(np.float32)
    df["volume"] = df["volume"].astype(np.float32)
    return df


def resample_5m_to_15m(df5: pd.DataFrame) -> pd.DataFrame:
    """Standard OHLCV resampling. Uses 15min buckets closed/labelled on
    the left (matches the trade feature bucketing)."""
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    df = df5.resample("15min", label="left", closed="left").agg(agg)
    df = df.dropna(subset=["open", "close"])
    return df


def load_trade_feats(path: str) -> pd.DataFrame:
    df = pd.read_feather(path).set_index("date").sort_index()
    return _strip_tz(df)


def load_futures_series(pattern_4h: str, resample_to: str = "15min") -> pd.DataFrame:
    """Load a single 4h/8h futures feather, forward-fill to the target freq."""
    p = Path(pattern_4h)
    if not p.exists():
        return None
    if pattern_4h.endswith(".feather"):
        df = pd.read_feather(p)
    else:
        df = pd.read_parquet(p)
    if "date" in df.columns:
        df = df.set_index("date")
    df = df.sort_index()
    df = _strip_tz(df)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ROLLING DERIVED FEATURES (time-series only, no lookahead)
# ══════════════════════════════════════════════════════════════════════════════

def _roll_z(series: pd.Series, w: int) -> pd.Series:
    m = series.rolling(w, min_periods=max(2, w // 4)).mean()
    s = series.rolling(w, min_periods=max(2, w // 4)).std().replace(0, np.nan)
    return ((series - m) / s).fillna(0).astype(np.float32)


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    log_ret = np.log(close.clip(1e-10)).diff(1).fillna(0)
    df["log_return_feature"]      = log_ret.astype(np.float32)
    df["close_return_feature"]    = close.pct_change(1).fillna(0).astype(np.float32)

    # Windows expressed in 15m bars: 4 = 1h, 16 = 4h, 96 = 1d, 672 = 1w
    for w, name in [(4, "1h"), (16, "4h"), (96, "1d"), (672, "1w")]:
        df[f"trend_return_{name}_feature"] = (
            np.log(close.clip(1e-10)).diff(w).fillna(0).astype(np.float32)
        )
        sma = close.rolling(w, min_periods=1).mean()
        df[f"ma_bias_{name}_feature"] = (
            (close / sma.clip(1e-10) - 1).fillna(0).astype(np.float32)
        )
        df[f"volatility_{name}_feature"] = (
            log_ret.rolling(w, min_periods=2).std().fillna(0).astype(np.float32)
        )

    # EMAs
    for span, name in [(4, "1h"), (16, "4h"), (96, "1d")]:
        ema = close.ewm(span=span, adjust=False).mean()
        df[f"ema_ratio_{name}_feature"] = (
            (ema / close.clip(1e-10) - 1).fillna(0).astype(np.float32)
        )

    # Bollinger width proxy
    for w, name in [(16, "4h"), (96, "1d")]:
        sma = close.rolling(w, min_periods=1).mean()
        std = close.rolling(w, min_periods=2).std().fillna(0)
        mid_clip = sma.clip(1e-10)
        df[f"bb_width_{name}_feature"] = (2 * std / mid_clip).fillna(0).astype(np.float32)

    return df


def add_orderflow_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling/normalised versions of the raw trade features."""
    # z-scores of CVD at various horizons
    for w, name in [(4, "1h"), (16, "4h"), (96, "1d")]:
        df[f"cvd_z{name}_trade_feature"] = _roll_z(df["cvd_trade_feature"], w)
        df[f"aggressor_ratio_{name}_trade_feature"] = (
            df["aggressor_ratio_trade_feature"]
              .rolling(w, min_periods=2).mean().fillna(0.5).astype(np.float32)
        )
    # Rolling sum of CVD (cumulative flow at 1h / 1d horizon)
    for w, name in [(4, "1h"), (96, "1d")]:
        df[f"cvd_cum_{name}_trade_feature"] = (
            df["cvd_trade_feature"].rolling(w, min_periods=1).sum()
              .astype(np.float32)
        )
    # Whale / large-trade intensity z-scored
    df["whale_intensity_trade_feature"] = _roll_z(
        df["whale_trades_trade_feature"].astype(np.float32), 96
    )
    df["large_intensity_trade_feature"] = _roll_z(
        df["large_trades_trade_feature"].astype(np.float32), 96
    )
    # VWAP skew — buy VWAP vs sell VWAP as a pct (fatter = more directional flow)
    df["vwap_skew_trade_feature"] = (
        (df["vwap_buy_trade_feature"] - df["vwap_sell_trade_feature"])
        / df["vwap_trade_feature"].clip(1e-10)
    ).fillna(0).astype(np.float32)
    return df


def add_futures_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join whatever futures feathers are available, forward-filled."""
    # funding rate (8h)
    f = load_futures_series(f"{FUTURES_DIR}/BTC_USDT_USDT-8h-funding_rate.feather")
    if f is not None:
        rate = f["open"] if "open" in f.columns else f.iloc[:, 0]
        rate = rate.reindex(df.index, method="ffill")
        df["funding_rate_feature"]    = rate.fillna(0).astype(np.float32)
        df["funding_rate_ma_feature"] = (
            rate.rolling(96, min_periods=1).mean().fillna(0).astype(np.float32)
        )
        # z-score rolling over ~7d (672 bars)
        df["funding_rate_zscore_feature"] = _roll_z(rate.fillna(0), 672)

    # OI (4h) — present in parquet format; resample
    oi_path = f"{FUTURES_DIR}/BTC_USDT_USDT-4h-open_interest.parquet"
    if Path(oi_path).exists():
        oi = pd.read_parquet(oi_path)
        if "date" in oi.columns:
            oi = oi.set_index("date")
        oi = _strip_tz(oi.sort_index())
        col = "sumOpenInterest" if "sumOpenInterest" in oi.columns else oi.columns[0]
        oi_s = oi[col].reindex(df.index, method="ffill")
        df["open_interest_feature"]      = oi_s.fillna(0).astype(np.float32)
        df["open_interest_zscore_feature"] = _roll_z(oi_s.fillna(method="bfill").fillna(0), 96)

    # L/S ratios (top + global)
    for label, fname in [
        ("top_ls",    "BTC_USDT_USDT-4h-top_ls_ratio.parquet"),
        ("global_ls", "BTC_USDT_USDT-4h-global_ls_ratio.parquet"),
    ]:
        p = f"{FUTURES_DIR}/{fname}"
        if not Path(p).exists():
            continue
        ls = pd.read_parquet(p)
        if "date" in ls.columns:
            ls = ls.set_index("date")
        ls = _strip_tz(ls.sort_index())
        col = "longShortRatio" if "longShortRatio" in ls.columns else ls.columns[0]
        ls_s = ls[col].reindex(df.index, method="ffill")
        df[f"{label}_ratio_feature"] = ls_s.fillna(1.0).astype(np.float32)
    return df


def add_rsi_and_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the small RSI set the pipeline expects."""
    import pandas_ta as ta
    for w in (12, 48, 192):   # 3h, 12h, 2d  — roughly matches 4h bars' 3/12/48
        df[f"rsi_{w}_feature"] = (ta.rsi(df["close"], length=w) / 100.0).fillna(0.5).astype(np.float32)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=OUT_PARQUET)
    args = ap.parse_args()

    print("  Loading 5m OHLCV …")
    df5   = load_ohlcv_5m(OHLCV_5M_PATH)
    print(f"    rows: {len(df5):,}  range {df5.index[0]} → {df5.index[-1]}")

    print("  Resampling 5m → 15m …")
    df15  = resample_5m_to_15m(df5)
    print(f"    rows: {len(df15):,}")

    print("  Loading trade features …")
    tr    = load_trade_feats(TRADE_FEATS_PATH)
    print(f"    rows: {len(tr):,}  range {tr.index[0]} → {tr.index[-1]}")

    # Inner-join on index — limits the final dataset to the overlap
    df = df15.join(tr, how="inner")
    print(f"  After OHLCV+trade join: {len(df):,} rows  "
          f"({df.index[0]} → {df.index[-1]})")

    print("  Adding price features …")
    df = add_price_features(df)

    print("  Adding order-flow derived features …")
    df = add_orderflow_derived(df)

    print("  Adding futures features (ffill) …")
    df = add_futures_features(df)

    print("  Adding RSI …")
    df = add_rsi_and_basic(df)

    # Drop anything with all-zero feature cols (constant → useless)
    feat_cols = [c for c in df.columns if "feature" in c]
    print(f"\n  Feature columns: {len(feat_cols)}")
    for c in feat_cols:
        if df[c].nunique(dropna=False) <= 1:
            print(f"    [drop] constant: {c}")
            df = df.drop(columns=[c])
            feat_cols.remove(c)
    print(f"  Final features  : {len(feat_cols)}")

    # Persist
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"\n  Wrote {len(df):,} rows × {len(df.columns)} cols → {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
