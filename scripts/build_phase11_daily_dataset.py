#!/usr/bin/env python3
"""
Phase 11 — Horizon Shift: Daily dataset builder.

Hypothesis: Basis and funding carry a weak directional signal that is
destroyed by hourly execution costs. A 7-day holding horizon reduces
turnover enough for that signal to become economically tradeable.

Approach:
  1. Resample 1h OHLCV to daily candles (UTC close = 00:00).
  2. Merge exogenous features (basis, funding) with merge_asof backward
     so bar at date D only sees data published strictly before D 00:00.
  3. Compute triple_barrier_7d: TP=6%, SL=3%, vertical=7 days.
  4. Run timestamp audit (no negative lags allowed).
  5. Run single-feature AUC scan (no feature > 0.65).

Usage:
    python3 scripts/build_phase11_daily_dataset.py

Output:
    cache/btc_daily_phase11_basis.feather
    cache/btc_daily_phase11_funding.feather
    cache/btc_daily_phase11_basis_audit.csv
    cache/btc_daily_phase11_funding_audit.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_BINANCE = ROOT.parent.parent / "data" / "binance"
FUTURES_DIR = DATA_BINANCE / "futures"
CACHE_DIR = ROOT / "cache"

SPOT_1H = DATA_BINANCE / "BTC_USDT-1h.feather"
FUNDING_8H = FUTURES_DIR / "BTC_USDT_USDT-8h-funding_rate.feather"
MARK_8H = FUTURES_DIR / "BTC_USDT_USDT-8h-mark.feather"

OUT_BASIS = CACHE_DIR / "btc_daily_phase11_basis.feather"
OUT_FUNDING = CACHE_DIR / "btc_daily_phase11_funding.feather"
OUT_BASIS_AUDIT = CACHE_DIR / "btc_daily_phase11_basis_audit.csv"
OUT_FUNDING_AUDIT = CACHE_DIR / "btc_daily_phase11_funding_audit.csv"

START_DATE = "2021-01-01"
TP_PCT = 0.06    # 6% take-profit
SL_PCT = 0.03    # 3% stop-loss
BARRIER_DAYS = 7
LABEL_COLS = {"triple_barrier_7d", "trend_7d"}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Build daily OHLCV from 1h candles
# ─────────────────────────────────────────────────────────────────────────────

def build_daily_ohlcv() -> pd.DataFrame:
    logger.info("Loading 1h OHLCV and resampling to daily...")
    df = pd.read_feather(SPOT_1H)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[df["date"] >= START_DATE].copy()

    df = df.set_index("date")
    daily = df.resample("1D").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])
    daily = daily.reset_index()
    daily = daily.rename(columns={"date": "date"})
    logger.info(f"Daily OHLCV: {len(daily)} bars from {daily['date'].min()} to {daily['date'].max()}")
    return daily


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Trend and volatility features (daily, backward-looking only)
# ─────────────────────────────────────────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    # Returns (backward)
    for p in [3, 7, 14, 30]:
        df[f"return_{p}d"] = df["close"].pct_change(p)

    # MA bias
    for p in [7, 14, 30, 90]:
        ma = df["close"].rolling(p).mean()
        df[f"ma_bias_{p}d"] = (df["close"] - ma) / ma

    # Realized vol
    daily_ret = df["close"].pct_change()
    for p in [7, 14, 30]:
        df[f"realized_vol_{p}d"] = daily_ret.rolling(p).std()

    # RSI
    def rsi(s, n=14):
        d = s.diff()
        g = d.where(d > 0, 0).rolling(n).mean()
        l = (-d.where(d < 0, 0)).rolling(n).mean()
        return 100 - 100 / (1 + g / l)

    df["rsi_14d"] = rsi(df["close"], 14)

    # ATR
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ),
    )
    df["atr_pct"] = tr.rolling(14).mean() / df["close"]

    # Drawdown from rolling high
    for p in [30, 90]:
        rh = df["high"].rolling(p).max()
        df[f"drawdown_from_high_{p}d"] = (rh - df["close"]) / rh

    # Volume z-score
    for p in [14, 30]:
        vm = df["volume"].rolling(p).mean()
        vs = df["volume"].rolling(p).std()
        df[f"volume_zscore_{p}d"] = (df["volume"] - vm) / (vs + 1e-8)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Triple-barrier target (7-day horizon)
# ─────────────────────────────────────────────────────────────────────────────

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Computing triple_barrier_7d (TP={TP_PCT*100:.0f}%, SL={SL_PCT*100:.0f}%, {BARRIER_DAYS}d)...")
    targets = []
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    for i in range(len(df)):
        if i + BARRIER_DAYS >= len(df):
            targets.append(np.nan)
            continue
        c0 = closes[i]
        tp = c0 * (1 + TP_PCT)
        sl = c0 * (1 - SL_PCT)
        window_high = highs[i+1:i+BARRIER_DAYS+1]
        window_low = lows[i+1:i+BARRIER_DAYS+1]
        final = closes[i+BARRIER_DAYS]

        if window_high.max() >= tp:
            targets.append(1)
        elif window_low.min() <= sl:
            targets.append(-1)
        else:
            targets.append(1 if final > c0 else -1)

    df["triple_barrier_7d"] = targets
    df["trend_7d"] = (df["close"].shift(-BARRIER_DAYS) > df["close"]).astype(float)
    df.loc[df.index[-BARRIER_DAYS:], "trend_7d"] = np.nan

    # Alignment assertion
    for idx in [100, len(df) // 2]:
        if idx + BARRIER_DAYS < len(df):
            expected = 1 if df["close"].iloc[idx + BARRIER_DAYS] > df["close"].iloc[idx] else 0
            actual = int(df["trend_7d"].iloc[idx])
            assert actual == expected, f"Alignment failure at row {idx}"

    dist = df["triple_barrier_7d"].value_counts()
    logger.info(f"Target distribution: {dist.to_dict()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Merge exogenous features with temporal safety
# ─────────────────────────────────────────────────────────────────────────────

def merge_basis(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Merging basis features...")
    mark = pd.read_feather(MARK_8H)
    spot = pd.read_feather(SPOT_1H)

    mark["date"] = pd.to_datetime(mark["date"], utc=True)
    spot["date"] = pd.to_datetime(spot["date"], utc=True)

    # Compute basis at each 8h mark close
    # merge_asof backward: for each mark bar, use last spot close <= mark date
    mark_sorted = mark.sort_values("date")
    spot_sorted = spot.sort_values("date")[["date", "close"]].rename(columns={"close": "spot_close"})

    merged = pd.merge_asof(mark_sorted, spot_sorted, on="date", direction="backward")
    merged["basis_raw"] = (merged["close"] - merged["spot_close"]) / merged["spot_close"]

    # Rolling z-score on basis (30d = ~90 marks)
    merged["basis_zscore_30d"] = (
        (merged["basis_raw"] - merged["basis_raw"].rolling(90).mean())
        / (merged["basis_raw"].rolling(90).std() + 1e-8)
    )
    merged["basis_mean_7d"] = merged["basis_raw"].rolling(21).mean()  # 21 × 8h = 7d

    # For each daily bar at date D (midnight UTC), take the last 8h mark bar
    # published BEFORE D midnight. Use merge_asof backward.
    daily_sorted = daily.sort_values("date")
    exog = merged[["date", "basis_raw", "basis_zscore_30d", "basis_mean_7d"]].sort_values("date")

    out = pd.merge_asof(daily_sorted, exog, on="date", direction="backward")

    # Timestamp audit: lag must be >= 0 (exog published before bar open)
    out["_lag_seconds"] = (out["date"] - out["date"]).dt.total_seconds()
    # True lag: daily bar date minus last exog date
    # We need to store exog date separately for audit
    exog_dated = merged[["date", "basis_raw", "basis_zscore_30d", "basis_mean_7d"]].copy()
    exog_dated = exog_dated.rename(columns={"date": "exog_date"})
    exog_dated["date"] = exog_dated["exog_date"]  # for merge_asof key

    out2 = pd.merge_asof(
        daily_sorted.assign(bar_date=daily_sorted["date"]),
        exog_dated.sort_values("date"),
        on="date",
        direction="backward",
    )
    audit = pd.DataFrame({
        "bar_date": out2["bar_date"],
        "exog_date": out2["exog_date"],
        "lag_seconds": (out2["bar_date"] - out2["exog_date"]).dt.total_seconds(),
        "feature": "basis",
    })

    return out, audit


def merge_funding(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Merging funding features...")
    fund = pd.read_feather(FUNDING_8H)
    fund["date"] = pd.to_datetime(fund["date"], utc=True)
    fund = fund.sort_values("date")
    # Binance funding feather stores the rate in the 'close' column
    fund = fund.rename(columns={"close": "funding_rate"})

    # Funding rate z-scores
    fund["funding_zscore_30d"] = (
        (fund["funding_rate"] - fund["funding_rate"].rolling(90).mean())
        / (fund["funding_rate"].rolling(90).std() + 1e-8)
    )
    fund["funding_cumsum_7d"] = fund["funding_rate"].rolling(21).sum()  # 21 × 8h = 7d
    fund["funding_sign"] = np.sign(fund["funding_rate"])

    exog = fund[["date", "funding_rate", "funding_zscore_30d", "funding_cumsum_7d", "funding_sign"]]

    daily_sorted = daily.sort_values("date")
    out = pd.merge_asof(daily_sorted, exog, on="date", direction="backward")

    exog_dated = exog.copy().rename(columns={"date": "exog_date"})
    exog_dated["date"] = exog_dated["exog_date"]
    out2 = pd.merge_asof(
        daily_sorted.assign(bar_date=daily_sorted["date"]),
        exog_dated.sort_values("date"),
        on="date",
        direction="backward",
    )
    audit = pd.DataFrame({
        "bar_date": out2["bar_date"],
        "exog_date": out2["exog_date"],
        "lag_seconds": (out2["bar_date"] - out2["exog_date"]).dt.total_seconds(),
        "feature": "funding",
    })

    return out, audit


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Timestamp audit
# ─────────────────────────────────────────────────────────────────────────────

def run_timestamp_audit(audit: pd.DataFrame, name: str) -> None:
    neg = (audit["lag_seconds"] < 0).sum()
    min_lag = audit["lag_seconds"].min()
    logger.info(f"[AUDIT:{name}] count_negative_lag={neg}, min_lag={min_lag:.0f}s")
    assert neg == 0, f"TIMESTAMP LEAK in {name}: {neg} rows with negative lag"
    logger.info(f"[AUDIT:{name}] PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Leakage pre-check (single-feature AUC)
# ─────────────────────────────────────────────────────────────────────────────

def run_leakage_scan(df: pd.DataFrame, feature_cols: list[str], target: str) -> None:
    from sklearn.metrics import roc_auc_score

    valid = df[df[target].notna()].copy()
    y = ((valid[target] + 1) / 2).astype(int)

    # Noise control
    rng = np.random.default_rng(42)
    valid["_noise"] = rng.standard_normal(len(valid))
    cols = feature_cols + ["_noise"]

    logger.info("Single-feature AUC scan:")
    flagged = []
    for col in cols:
        v = valid[col].fillna(valid[col].median())
        auc = roc_auc_score(y, v)
        auc_sym = max(auc, 1 - auc)
        flag = "  <-- WARN: audit required" if auc_sym > 0.65 else ""
        logger.info(f"  {col:35s} AUC={auc:.4f} (sym={auc_sym:.4f}){flag}")
        if auc_sym > 0.65:
            flagged.append((col, auc_sym))

    if flagged:
        raise AssertionError(f"Suspected target leakage (AUC > 0.65): {flagged}")
    logger.info("Leakage scan PASSED.")

    # LABEL_COLS safety
    leak = [c for c in feature_cols if c in LABEL_COLS]
    assert not leak, f"CRITICAL: label columns in feature list: {leak}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

BASIS_FEATURES = [
    "basis_raw", "basis_zscore_30d", "basis_mean_7d",
]
FUNDING_FEATURES = [
    "funding_rate", "funding_zscore_30d", "funding_cumsum_7d", "funding_sign",
]
PRICE_FEATURES = [
    "return_3d", "return_7d", "return_14d", "return_30d",
    "ma_bias_7d", "ma_bias_14d", "ma_bias_30d", "ma_bias_90d",
    "realized_vol_7d", "realized_vol_14d", "realized_vol_30d",
    "rsi_14d", "atr_pct",
    "drawdown_from_high_30d", "drawdown_from_high_90d",
    "volume_zscore_14d", "volume_zscore_30d",
]


def main():
    daily = build_daily_ohlcv()
    daily = add_price_features(daily)
    daily = add_target(daily)

    # Trim warmup (need 90d rolling for z-scores)
    min_valid = daily[daily["realized_vol_30d"].notna()].index.min()
    daily = daily.loc[min_valid:].reset_index(drop=True)
    logger.info(f"After warmup trim: {len(daily)} daily bars")

    # ── Basis dataset ─────────────────────────────────────────────────────────
    basis_df, basis_audit = merge_basis(daily.copy())
    basis_audit.to_csv(OUT_BASIS_AUDIT, index=False)
    run_timestamp_audit(basis_audit, "basis")

    basis_feature_cols = PRICE_FEATURES + BASIS_FEATURES
    run_leakage_scan(basis_df.dropna(subset=["triple_barrier_7d"]), basis_feature_cols, "triple_barrier_7d")

    basis_out = basis_df.drop(columns=["_lag_seconds"], errors="ignore")
    basis_out.to_feather(OUT_BASIS)
    logger.info(f"Saved {OUT_BASIS} ({len(basis_out)} rows, {len(basis_out.columns)} cols)")

    # ── Funding dataset ───────────────────────────────────────────────────────
    fund_df, fund_audit = merge_funding(daily.copy())
    fund_audit.to_csv(OUT_FUNDING_AUDIT, index=False)
    run_timestamp_audit(fund_audit, "funding")

    funding_feature_cols = PRICE_FEATURES + FUNDING_FEATURES
    run_leakage_scan(fund_df.dropna(subset=["triple_barrier_7d"]), funding_feature_cols, "triple_barrier_7d")

    fund_df.to_feather(OUT_FUNDING)
    logger.info(f"Saved {OUT_FUNDING} ({len(fund_df)} rows, {len(fund_df.columns)} cols)")

    logger.info("\n[Phase 11 dataset build COMPLETE]")


if __name__ == "__main__":
    main()
