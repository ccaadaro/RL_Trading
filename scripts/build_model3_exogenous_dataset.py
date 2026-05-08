#!/usr/bin/env python3
"""
Phase 10 — Model 3: Exogenous Dataset Builder.

Builds btc_1h_phase10_{family}.feather by merging Phase 9 1h OHLCV
with exogenous feature families (funding, basis, OI) one at a time.

STRICT RULES (see MODEL3_PROTOCOLS.md):
 1. Timestamp safety: every exogenous value used for bar at time t
    must have been published STRICTLY BEFORE t (merge_asof backward).
 2. Timestamp audit MUST pass (count_negative_lag == 0) before any model trains.
 3. Do not hardcode signal direction — features are raw inputs.
 4. Only Family 1 (funding) is built unless protocol explicitly advances.

Usage:
    conda run -n freqtrade python3 scripts/build_model3_exogenous_dataset.py

Output:
    cache/btc_1h_phase10_funding.feather
    cache/btc_1h_phase10_funding_audit.csv   (timestamp audit table)
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

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_BINANCE = ROOT.parent.parent / "data" / "binance"
FUTURES_DIR = DATA_BINANCE / "futures"
CACHE_DIR = ROOT / "cache"

PHASE9_FEATHER = CACHE_DIR / "btc_1h_phase9.feather"
FUNDING_FEATHER = FUTURES_DIR / "BTC_USDT_USDT-8h-funding_rate.feather"
MARK_FEATHER = FUTURES_DIR / "BTC_USDT_USDT-8h-mark.feather"
SPOT_1H_FEATHER = DATA_BINANCE / "BTC_USDT-1h.feather"

OUTPUT_FUNDING = CACHE_DIR / "btc_1h_phase10_funding.feather"
OUTPUT_FUNDING_AUDIT = CACHE_DIR / "btc_1h_phase10_funding_audit.csv"
OUTPUT_BASIS = CACHE_DIR / "btc_1h_phase10_basis.feather"
OUTPUT_BASIS_AUDIT = CACHE_DIR / "btc_1h_phase10_basis_audit.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_utc_ns(series: pd.Series) -> pd.Series:
    """Normalize a datetime series to UTC, ns precision, tz-aware."""
    if series.dt.tz is None:
        series = series.dt.tz_localize("UTC")
    else:
        series = series.dt.tz_convert("UTC")
    return series


def _timestamp_audit(
    df: pd.DataFrame,
    feature_cols: list[str],
    source_ts_col: str = "_source_ts",
    bar_ts_col: str = "date",
) -> pd.DataFrame:
    """
    Build timestamp audit table per feature family.

    Returns a DataFrame with:
      feature_name, source_timestamp, bar_open_time,
      lag_seconds (bar - source), min_lag, p50_lag, p95_lag, count_negative_lag.
    """
    lag = (df[bar_ts_col] - df[source_ts_col]).dt.total_seconds()
    rows = []
    for feat in feature_cols:
        mask = df[feat].notna()
        lag_valid = lag[mask]
        rows.append({
            "feature_name": feat,
            "n_valid": int(mask.sum()),
            "min_lag_s": float(lag_valid.min()) if len(lag_valid) else np.nan,
            "p50_lag_s": float(lag_valid.median()) if len(lag_valid) else np.nan,
            "p95_lag_s": float(lag_valid.quantile(0.95)) if len(lag_valid) else np.nan,
            "count_negative_lag": int((lag_valid < 0).sum()),
        })
    return pd.DataFrame(rows)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Causal rolling z-score (no lookahead)."""
    mu = series.rolling(window, min_periods=window // 2).mean()
    sigma = series.rolling(window, min_periods=window // 2).std()
    return (series - mu) / sigma.replace(0, np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Family 1: Funding Rate
# ─────────────────────────────────────────────────────────────────────────────

def build_funding_features(df_base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge funding rate features into the base 1h OHLCV DataFrame.

    Timestamp safety: uses merge_asof(direction='backward') so that
    every bar at time t only sees funding published STRICTLY BEFORE t.

    The Binance funding_rate.feather stores the rate in the 'open' column
    at 8h intervals (00:00, 08:00, 16:00 UTC). The timestamp in 'date'
    is the settlement time. We treat 'date' as the publication time.

    Returns:
        df_merged: DataFrame with funding features added.
        audit_df: Timestamp audit table.
    """
    logger.info("=== Building Funding Rate Features ===")

    # --- Load and clean funding rate ---
    logger.info(f"Loading funding rate from {FUNDING_FEATHER}")
    fr = pd.read_feather(FUNDING_FEATHER)
    fr["date"] = _ensure_utc_ns(fr["date"])

    # The actual funding rate is stored in 'open' column (Freqtrade convention)
    fr = fr[["date", "open"]].rename(columns={"open": "funding_rate"}).copy()
    fr = fr.dropna(subset=["funding_rate"])
    fr = fr.sort_values("date").reset_index(drop=True)

    # Filter to Phase 9 era (2021+) — aligned with base dataset
    START_DATE = pd.Timestamp("2021-01-01", tz="UTC")
    fr = fr[fr["date"] >= START_DATE].reset_index(drop=True)

    logger.info(f"Funding rate: {len(fr)} rows from {fr['date'].min()} to {fr['date'].max()}")

    # --- Rolling statistics on the funding time series ---
    # 30d = 30 * 3 = 90 settlements; 90d = 90 * 3 = 270 settlements
    WINDOW_30D = 90    # 30 days × 3 settlements/day
    WINDOW_90D = 270   # 90 days × 3 settlements/day

    fr["funding_8h_zscore_30d"] = _rolling_zscore(fr["funding_rate"], WINDOW_30D)
    fr["funding_8h_zscore_90d"] = _rolling_zscore(fr["funding_rate"], WINDOW_90D)
    fr["funding_abs_zscore_30d"] = fr["funding_8h_zscore_30d"].abs()
    fr["funding_sign"] = np.sign(fr["funding_rate"])

    FUNDING_FEAT_COLS = [
        "funding_rate",          # renamed from funding_last in protocol (raw value)
        "funding_8h_zscore_30d",
        "funding_8h_zscore_90d",
        "funding_abs_zscore_30d",
        "funding_sign",
    ]

    # --- Prepare base DataFrame ---
    df = df_base.copy()
    df["date"] = _ensure_utc_ns(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # --- Timestamp-safe merge (backward) ---
    # merge_asof with direction='backward':
    #   for each bar at time t, picks the last funding row with date <= t.
    # We want STRICTLY BEFORE, so we shift the funding timestamps forward by 1ms
    # so that a funding row at exactly t does NOT match bar at t.
    fr_shifted = fr.copy()
    fr_shifted["_source_ts"] = fr_shifted["date"]  # preserve true publication time
    # Shift publication time by 1ms so merge_asof only picks up rows < bar_open
    fr_shifted["date"] = fr_shifted["date"] + pd.Timedelta(milliseconds=1)

    df = pd.merge_asof(
        df,
        fr_shifted[["date", "_source_ts"] + FUNDING_FEAT_COLS],
        on="date",
        direction="backward",
    )

    # --- Timestamp audit ---
    logger.info("Running timestamp audit...")
    audit_df = _timestamp_audit(df, FUNDING_FEAT_COLS, "_source_ts", "date")
    logger.info("\nTimestamp Audit Table:")
    logger.info("\n" + audit_df.to_string(index=False))

    neg_lag_total = int(audit_df["count_negative_lag"].sum())
    if neg_lag_total > 0:
        logger.error(
            f"BLOCKED: {neg_lag_total} rows with negative lag detected! "
            "This is temporal leakage. Fix timestamp alignment before proceeding."
        )
        sys.exit(1)
    else:
        logger.info("✅ Timestamp audit PASSED — zero negative lags detected.")

    # Drop internal column
    df = df.drop(columns=["_source_ts"], errors="ignore")

    # Rename 'funding_rate' to 'funding_last' to match protocol nomenclature
    df = df.rename(columns={"funding_rate": "funding_last"})
    FUNDING_FEAT_COLS = ["funding_last" if c == "funding_rate" else c for c in FUNDING_FEAT_COLS]

    logger.info(f"Funding features added: {FUNDING_FEAT_COLS}")
    logger.info(f"Non-null counts:\n{df[FUNDING_FEAT_COLS].notna().sum()}")

    return df, audit_df


# ─────────────────────────────────────────────────────────────────────────────
# Family 2: Perp-Spot Basis
# ─────────────────────────────────────────────────────────────────────────────

def build_basis_features(df_base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge perp-spot basis features.

    basis_now = (perp_mark_close - spot_close) / spot_close

    Timestamp safety: the mark price 8h bar closes at time t.
    We use merge_asof(direction='backward') so only mark bars
    with close_time <= bar_open are used.
    """
    logger.info("=== Building Perp-Spot Basis Features ===")

    mark = pd.read_feather(MARK_FEATHER)
    mark["date"] = _ensure_utc_ns(mark["date"])
    mark = mark[["date", "close"]].rename(columns={"close": "perp_mark"}).copy()
    mark = mark.dropna().sort_values("date").reset_index(drop=True)

    START_DATE = pd.Timestamp("2021-01-01", tz="UTC")
    mark = mark[mark["date"] >= START_DATE].reset_index(drop=True)
    logger.info(f"Mark price: {len(mark)} rows from {mark['date'].min()} to {mark['date'].max()}")

    # Shift mark timestamp forward by 1 bar (8h) so we only use
    # the mark price from the COMPLETED 8h bar before the decision bar.
    mark["_source_ts"] = mark["date"]
    mark["date"] = mark["date"] + pd.Timedelta(hours=8)  # bar completes at t+8h

    df = df_base.copy()
    df["date"] = _ensure_utc_ns(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = pd.merge_asof(
        df,
        mark[["date", "_source_ts", "perp_mark"]],
        on="date",
        direction="backward",
    )

    # Compute basis
    df["basis_now"] = (df["perp_mark"] - df["close"]) / df["close"]

    # Rolling stats
    WINDOW_30D_1H = 720   # 30d × 24h
    df["basis_zscore_30d"] = _rolling_zscore(df["basis_now"], WINDOW_30D_1H)
    df["basis_mean_24h"] = df["basis_now"].rolling(24, min_periods=12).mean()
    df["basis_change_24h"] = df["basis_now"] - df["basis_now"].shift(24)
    df["basis_compression"] = df["basis_now"] - df["basis_mean_24h"]

    BASIS_FEAT_COLS = [
        "basis_now",
        "basis_zscore_30d",
        "basis_mean_24h",
        "basis_change_24h",
        "basis_compression",
    ]

    # Timestamp audit
    audit_df = _timestamp_audit(df, BASIS_FEAT_COLS, "_source_ts", "date")
    logger.info("\nTimestamp Audit Table (Basis):")
    logger.info("\n" + audit_df.to_string(index=False))

    neg_lag_total = int(audit_df["count_negative_lag"].sum())
    if neg_lag_total > 0:
        logger.error(f"BLOCKED: {neg_lag_total} basis rows with negative lag. Fix before proceeding.")
        sys.exit(1)
    else:
        logger.info("✅ Basis timestamp audit PASSED.")

    df = df.drop(columns=["_source_ts", "perp_mark"], errors="ignore")
    return df, audit_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(family: str = "funding") -> None:
    """
    Build the exogenous dataset for the specified feature family.

    Args:
        family: One of 'funding', 'basis', 'all'.
                Use 'funding' for the initial Phase 10 experiment.
    """
    logger.info(f"Building Phase 10 dataset — family: {family}")
    logger.info(f"Loading Phase 9 base dataset from {PHASE9_FEATHER}")

    df_base = pd.read_feather(PHASE9_FEATHER)
    logger.info(f"Base dataset: {df_base.shape} rows×cols, "
                f"{df_base['date'].min()} → {df_base['date'].max()}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if family in ("funding", "all"):
        df_funding, funding_audit = build_funding_features(df_base)
        df_funding.to_feather(OUTPUT_FUNDING)
        funding_audit.to_csv(OUTPUT_FUNDING_AUDIT, index=False)
        logger.info(f"Saved: {OUTPUT_FUNDING} ({df_funding.shape})")
        logger.info(f"Saved audit: {OUTPUT_FUNDING_AUDIT}")

        logger.info("\n=== Dataset Summary (Funding) ===")
        funding_cols = [c for c in df_funding.columns if "funding" in c]
        logger.info(f"New funding columns: {funding_cols}")
        logger.info(f"Non-null counts:\n{df_funding[funding_cols].notna().sum()}")
        logger.info(f"\nFunding feature statistics:\n{df_funding[funding_cols].describe()}")

    if family in ("basis", "all"):
        # Load funding dataset as base if already built
        base_for_basis = (
            pd.read_feather(OUTPUT_FUNDING)
            if OUTPUT_FUNDING.exists() and family == "all"
            else df_base
        )
        df_basis, basis_audit = build_basis_features(base_for_basis)
        df_basis.to_feather(OUTPUT_BASIS)
        basis_audit.to_csv(OUTPUT_BASIS_AUDIT, index=False)
        logger.info(f"Saved: {OUTPUT_BASIS} ({df_basis.shape})")
        logger.info(f"Saved audit: {OUTPUT_BASIS_AUDIT}")

    logger.info("\n✅ Dataset build complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Phase 10 exogenous dataset")
    parser.add_argument(
        "--family",
        default="funding",
        choices=["funding", "basis", "all"],
        help="Feature family to build (default: funding)",
    )
    args = parser.parse_args()
    main(args.family)
