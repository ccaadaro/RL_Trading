"""
utils/signal_features.py
─────────────────────────
Shared feature engineering for the LightGBM signal model v2.

Used by BOTH the training script and the Freqtrade strategy to guarantee
that training-time features exactly match inference-time features.

Computable from:
  - Standard OHLCV candles (BTC/USDT 1h)
  - ETH/USDT 1h candles (for cross-asset features)
  - 8h funding rate feather file

Features included: 58 (vs 104 in v1 — drops all taker-volume,
cross-section-network, and external-API-only features).

Feature conventions (must match the original parquet to re-use v1 model):
  ema_ratio_N  = (EMA_N / close) - 1      ← EMA over close, NOT close over EMA
  rsi_N        = ta.rsi(close, N) / 100   ← scaled to [0, 1]
  eth_btc_ratio = (ratio / SMA_180(ratio)) - 1
  eth_btc_trend = log(ratio).diff(180)
  eth_btc_zscore = (ratio - SMA_540) / std_540
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Optional
from utils import tv_indicators as tv


# ─────────────────────────────────────────────────────────────────────────────
# Feature set definition
# ─────────────────────────────────────────────────────────────────────────────

# 1. Institutional Base (14 features)
FEATURE_SET_INSTITUTIONAL = sorted([
    "cvd_4h_sum_trade_feature", "aggressor_ratio_4h_mean_trade_feature",
    "whale_trades_4h_sum_trade_feature", "large_trades_4h_sum_trade_feature",
    "max_trade_usd_4h_max_trade_feature", "vwap_skew_4h_mean_trade_feature",
    "whale_intensity_4h_mean_trade_feature", "l2_imbalance_feature", 
    "liq_vola_feature", "cross_exchange_premium_feature",
    "tv_cvd_zscore_feature", "tv_cvd_slope_feature", 
    "tv_aggr_delta_feature", "tv_buy_sell_imbalance_feature"
])

# 2. Elite Set (Institutional + WVF + %R = 21 features)
SIGNAL_FEAT_COLS_V2 = sorted(FEATURE_SET_INSTITUTIONAL + [
    "tv_wvf_panic_feature", "tv_wvf_val_feature",
    "pr_exhaust_ob_feature", "pr_exhaust_os_feature",
    "pr_exhaust_ob_reversal_feature", "pr_exhaust_os_reversal_feature",
    "pr_spread_feature"
])


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all OHLCV-derivable features in-place (adds columns to df).

    Parameters
    ----------
    df : DataFrame with columns: open, high, low, close, volume
         Index must be DatetimeIndex.

    Returns
    -------
    Same DataFrame with new *_feature columns appended.
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]
    if "volume" not in df.columns:
        df["volume"] = df["buy_vol"] + df["sell_vol"]
    volume = df["volume"]
    log_close = np.log(close.clip(1e-10))
    log_ret   = log_close.diff(1).fillna(0)

    # ── Returns ──────────────────────────────────────────────────────────────
    df["close_return_feature"]      = close.pct_change(1).fillna(0)
    df["log_return_feature"]        = log_ret
    df["trend_return_180_feature"]  = log_close.diff(180).fillna(0)
    df["trend_return_540_feature"]  = log_close.diff(540).fillna(0)

    # ── MA bias (close/SMA - 1) ──
    for n in [3, 6, 12, 24, 42, 90, 180, 540]:
        df[f"ma_bias_{n}_feature"] = (close / ta.sma(close, n) - 1).fillna(0)

    # ── %R Trend Exhaustion (Pine Script Translation) ──
    # _pr(length) => 100 * (src - max) / (max - min)
    # TradingView logic uses high/low for the channel but 'src' (close) for the numerator.
    def get_pr(src, length):
        high_roll = df['high'].rolling(length).max()
        low_roll  = df['low'].rolling(length).min()
        return 100 * (src - high_roll) / (high_roll - low_roll).clip(lower=1e-9)

    pr_fast = get_pr(close, 21)
    pr_slow = get_pr(close, 112)
    
    thr = 20
    ob = (pr_fast >= -thr) & (pr_slow >= -thr)
    os = (pr_fast <= -100 + thr) & (pr_slow <= -100 + thr)
    
    df["pr_exhaust_ob_feature"] = ob.astype(float)
    df["pr_exhaust_os_feature"] = os.astype(float)
    df["pr_exhaust_ob_reversal_feature"] = ((~ob) & ob.shift(1)).astype(float).fillna(0)
    df["pr_exhaust_os_reversal_feature"] = ((~os) & os.shift(1)).astype(float).fillna(0)
    df["pr_spread_feature"] = (pr_fast - pr_slow) / 100.0

    # ── Refined TV Indicators ──
    # 1. Hull Suite
    hma55 = tv.tv_hma(close, 55)
    df["tv_hull_hma_55_feature"] = (close / hma55 - 1).fillna(0)
    df["tv_hull_hma_slope_feature"] = (hma55 / hma55.shift(1) - 1).fillna(0)
    df["tv_hull_hma_dist_feature"] = (close - hma55) / close.clip(1e-10)

    # 2. Williams Vix Fix
    wvf_df = tv.tv_williams_vix_fix(df)
    df["tv_wvf_panic_feature"] = wvf_df["tv_wvf_panic"]
    df["tv_wvf_val_feature"] = wvf_df["tv_wvf_val"]

    # 3. Laguerre Bundle
    lag_df = tv.tv_laguerre_bundle(close)
    df["tv_lag_fast_feature"] = lag_df["tv_lag_fast"] / close.clip(1e-10)
    df["tv_lag_mid_feature"]  = lag_df["tv_lag_mid"] / close.clip(1e-10)
    df["tv_lag_slow_feature"] = lag_df["tv_lag_slow"] / close.clip(1e-10)
    df["tv_lag_slope_feature"] = lag_df["tv_lag_slope"]
    df["tv_lag_dispersion_feature"] = lag_df["tv_lag_dispersion"]
    df["tv_price_dist_lag_feature"] = lag_df["tv_price_dist_lag"]

    # 4. Selective Koncorde / TSV
    konk_df = tv.tv_koncorde_selective(df)
    df["tv_pvi_nvi_spread_feature"] = konk_df["tv_pvi_nvi_spread"]
    df["tv_tsv_feature"] = konk_df["tv_tsv"]

    # 5. Refined Microstructure
    micro_df = tv.tv_microstructure_refined(df)
    df["tv_cvd_zscore_feature"] = micro_df["tv_cvd_zscore"]
    df["tv_cvd_slope_feature"] = micro_df["tv_cvd_slope"]
    df["tv_aggr_delta_feature"] = micro_df["tv_aggr_delta"]
    df["tv_buy_sell_imbalance_feature"] = micro_df["tv_buy_sell_imbalance"]

    # ── Institutional Data Placeholders (to be populated by daemon) ──
    # If not present in incoming df, we init to 0.0
    for inst_col in ["l2_imbalance_feature", "liq_vola_feature", "cross_exchange_premium_feature"]:
        if inst_col not in df.columns:
            df[inst_col] = 0.0

    # ── EMA ratio: (EMA_w / close) - 1  [EMA over close] ─────────────────
    for w in [3, 6, 12, 24, 42, 90, 180]:
        ema = close.ewm(span=w, adjust=False).mean()
        df[f"ema_ratio_{w}_feature"] = (ema / close.clip(1e-10) - 1).fillna(0)

    # ── RSI: normalized to [0, 1] ─────────────────────────────────────────
    for w in [3, 6, 12, 24, 42, 90, 180]:
        _rsi = ta.rsi(close, length=w)
        if _rsi is None:
            # pandas_ta can return None on integer-indexed Series; fall back
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(w, min_periods=1).mean()
            loss  = (-delta.clip(upper=0)).rolling(w, min_periods=1).mean().clip(1e-10)
            rs    = gain / loss
            _rsi  = (100 - 100 / (1 + rs)).fillna(50)
        df[f"rsi_{w}_feature"] = _rsi.fillna(50) / 100.0

    # ── Bollinger Bands ───────────────────────────────────────────────────
    for w in [12, 42]:
        bb = ta.bbands(close, length=w, std=2.0)
        if bb is not None and not bb.empty:
            lo_col  = next(c for c in bb.columns if c.startswith("BBL"))
            hi_col  = next(c for c in bb.columns if c.startswith("BBU"))
            mid_col = next(c for c in bb.columns if c.startswith("BBM"))
            bb_lo, bb_hi, bb_mid = bb[lo_col], bb[hi_col], bb[mid_col]
        else:
            sma = close.rolling(w, min_periods=1).mean()
            std = close.rolling(w, min_periods=1).std().fillna(0)
            bb_lo, bb_hi, bb_mid = sma - 2*std, sma + 2*std, sma
        bb_range = (bb_hi - bb_lo).clip(1e-9)
        df[f"bb_position_{w}_feature"] = (
            (close - bb_lo) / bb_range
        ).clip(0, 1).fillna(0.5)
        df[f"bb_width_{w}_feature"] = (
            bb_range / bb_mid.clip(1e-10)
        ).fillna(0)

    # ── Volatility ────────────────────────────────────────────────────────
    for w in [3, 6, 12, 24, 42, 90, 180]:
        df[f"volatility_{w}_feature"] = (
            log_ret.rolling(w, min_periods=2).std().fillna(0)
        )
    df["realized_vol_1h_feature"] = df["volatility_12_feature"]

    # ── Range / wick ──────────────────────────────────────────────────────
    candle_range = (high - low).clip(1e-9)
    body_hi      = np.maximum(open_, close)
    body_lo      = np.minimum(open_, close)
    df["wick_upper_feature"]    = ((high - body_hi) / candle_range).fillna(0)
    df["wick_lower_feature"]    = ((body_lo - low)  / candle_range).fillna(0)
    df["range_feature"]         = candle_range
    df["intraday_range_feature"] = (candle_range / close.clip(1e-10)).fillna(0)

    # ── Volume z-score ────────────────────────────────────────────────────
    for w in [3, 6, 12, 24, 42, 90, 180]:
        v_mean = volume.rolling(w, min_periods=2).mean()
        v_std  = volume.rolling(w, min_periods=2).std().clip(1e-9)
        df[f"volume_zscore_{w}_feature"] = (
            (volume - v_mean) / v_std
        ).fillna(0)

    # ── Fibonacci ─────────────────────────────────────────────────────────
    for w in [42, 180]:
        roll_hi   = high.rolling(w, min_periods=1).max()
        roll_lo   = low.rolling(w, min_periods=1).min()
        fib_range = (roll_hi - roll_lo).clip(1e-9)
        fib_618   = roll_lo + 0.618 * fib_range
        df[f"fib_position_{w}_feature"] = (
            (close - roll_lo) / fib_range
        ).clip(0, 1).fillna(0.5)
        df[f"fib_dist_618_{w}_feature"] = (
            (close - fib_618) / close.clip(1e-10)
        ).fillna(0)

    return df


def compute_eth_features(
    btc_df: pd.DataFrame,
    eth_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute ETH/BTC cross-asset features and add them to btc_df.

    Formulas reverse-engineered from original parquet (corr ≥ 0.98):
      eth_btc_ratio  = (raw_ratio / SMA_180(raw_ratio)) - 1
      eth_btc_trend  = log(raw_ratio).diff(180)
      eth_btc_zscore = (raw_ratio - SMA_540) / std_540

    Parameters
    ----------
    btc_df : BTC DataFrame (modified in-place, has DatetimeIndex).
    eth_df : ETH DataFrame — must have 'close' column and DatetimeIndex.
             Timezone is stripped if btc_df index is tz-naive.

    Returns
    -------
    btc_df with 3 new feature columns.
    """
    eth_close = eth_df["close"] if "close" in eth_df.columns else eth_df.iloc[:, 0]

    # Align timezones
    if eth_close.index.tz is not None and btc_df.index.tz is None:
        eth_close = eth_close.copy()
        eth_close.index = eth_close.index.tz_localize(None)

    eth_c    = eth_close.reindex(btc_df.index, method="ffill")
    btc_c    = btc_df["close"]
    raw_ratio = (eth_c / btc_c.clip(1e-10)).replace([np.inf, -np.inf], np.nan)
    log_ratio = np.log(raw_ratio.clip(1e-10))

    # (ratio / SMA_180) - 1
    sma_180 = raw_ratio.rolling(180, min_periods=48).mean().clip(1e-10)
    btc_df["eth_btc_ratio_feature"] = ((raw_ratio / sma_180) - 1).fillna(0)

    # log(ratio).diff(180)
    btc_df["eth_btc_trend_feature"] = log_ratio.diff(180).fillna(0)

    # (ratio - SMA_540) / std_540
    sma_540 = raw_ratio.rolling(540, min_periods=180).mean()
    std_540 = raw_ratio.rolling(540, min_periods=180).std().clip(1e-9)
    btc_df["eth_btc_zscore_feature"] = (
        (raw_ratio - sma_540) / std_540
    ).fillna(0)

    return btc_df


def compute_funding_features(
    df: pd.DataFrame,
    funding_series: pd.Series,
) -> pd.DataFrame:
    """
    Merge 8h funding rate (forward-filled to 1h) into df.

    Parameters
    ----------
    df             : BTC DataFrame (modified in-place).
    funding_series : 1h-resampled funding rate Series.
                     Timezone stripped if df index is tz-naive.

    Returns
    -------
    df with funding_rate_feature, funding_rate_ma_feature,
    funding_rate_zscore_feature added.
    """
    if funding_series is None or funding_series.empty:
        df["funding_rate_feature"]        = 0.0
        df["funding_rate_ma_feature"]     = 0.0
        df["funding_rate_zscore_feature"] = 0.0
        return df

    rate = funding_series.copy()
    if rate.index.tz is not None and df.index.tz is None:
        rate.index = rate.index.tz_localize(None)

    rate = rate.reindex(df.index, method="ffill").fillna(0.0)

    df["funding_rate_feature"]    = rate.values
    df["funding_rate_ma_feature"] = rate.rolling(24, min_periods=1).mean().values

    r_mean = rate.rolling(180, min_periods=24).mean()
    r_std  = rate.rolling(180, min_periods=24).std().clip(1e-9)
    df["funding_rate_zscore_feature"] = ((rate - r_mean) / r_std).fillna(0).values

    return df


def load_funding_series(funding_path: str) -> Optional[pd.Series]:
    """
    Load 8h funding rate feather file and resample to 1h.

    The feather file stores funding rate in the 'open' column.
    Returns None if the file is missing.
    """
    from pathlib import Path
    p = Path(funding_path)
    if not p.exists():
        return None
    fund = p.read_bytes()   # just check existence
    import pickle as _pickle  # noqa — avoid reimport confusion
    df_fund = pd.read_feather(str(p))
    if "date" in df_fund.columns:
        df_fund = df_fund.set_index("date")
    df_fund.index = pd.to_datetime(df_fund.index, utc=True)
    rate_8h = df_fund["open"].rename("funding_rate")
    return rate_8h.resample("1h").ffill()


def build_feature_matrix(
    btc_df: pd.DataFrame,
    eth_df: Optional[pd.DataFrame] = None,
    funding_series: Optional[pd.Series] = None,
    feat_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Full pipeline: compute all signal-v2 features and return feature matrix.

    Parameters
    ----------
    btc_df         : BTC OHLCV DataFrame (modified in-place).
    eth_df         : ETH OHLCV DataFrame (optional; ETH features set to 0 if None).
    funding_series : 1h-resampled funding rate (optional; set to 0 if None).
    feat_cols      : list of feature column names to include in output.
                     Defaults to SIGNAL_FEAT_COLS_V2.

    Returns
    -------
    DataFrame indexed like btc_df, columns = feat_cols.
    All NaN/inf values replaced with 0.
    """
    if feat_cols is None:
        feat_cols = SIGNAL_FEAT_COLS_V2

    compute_ohlcv_features(btc_df)

    if eth_df is not None and not eth_df.empty:
        compute_eth_features(btc_df, eth_df)
    else:
        for col in ["eth_btc_ratio_feature", "eth_btc_trend_feature",
                    "eth_btc_zscore_feature"]:
            btc_df[col] = 0.0

    compute_funding_features(btc_df, funding_series)

    # D-03 FIX: Compute microstructure features from bar-level data when
    # columns are available (live daemon). Falls back to 0 when absent.
    compute_microstructure_from_bars(btc_df)

    # Assemble output — fill missing columns with 0
    X = pd.DataFrame(index=btc_df.index)
    for col in feat_cols:
        X[col] = btc_df.get(col, 0.0)

    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_microstructure_from_bars(df: pd.DataFrame, window: int = 48) -> pd.DataFrame:
    """
    D-03 FIX: Approximates the 10 microstructure *_trade_feature columns
    from bar-level aggregates (buy_volume, aggressor_ratio, notional, trade_count).

    The daemon provides per-bar: buy_volume, volume, aggressor_ratio, notional,
    trade_count. A rolling window of `window` bars (~4h at 2M USD bar size)
    approximates the 4h aggregations computed offline from raw tick data.

    If microstructure columns already exist (e.g., pre-built feather), this
    function is a no-op to avoid overwriting higher-fidelity data.

    Parameters
    ----------
    df     : Bar DataFrame. Modified in-place.
    window : Rolling bars to approximate 4h. Default 48 (~4–8h at typical bar cadence).
    """
    # Skip if high-fidelity data already present
    if "cvd_4h_sum_trade_feature" in df.columns and df["cvd_4h_sum_trade_feature"].abs().sum() > 0:
        return df

    # Require at least buy_volume + volume or aggressor_ratio to proceed
    has_buy_vol = "buy_volume" in df.columns
    has_aggr    = "aggressor_ratio" in df.columns
    has_volume  = "volume" in df.columns
    has_notional = "notional" in df.columns
    has_trades  = "trade_count" in df.columns

    if not (has_buy_vol or has_aggr):
        # Nothing to compute from: set all to 0
        for col in [
            "cvd_4h_sum_trade_feature", "aggressor_ratio_4h_mean_trade_feature",
            "whale_trades_4h_sum_trade_feature", "large_trades_4h_sum_trade_feature",
            "max_trade_usd_4h_max_trade_feature", "vwap_skew_4h_mean_trade_feature",
            "whale_intensity_4h_mean_trade_feature", "large_intensity_4h_mean_trade_feature",
            "cvd_4h_zscore_trade_feature", "aggressor_vs_baseline_trade_feature",
        ]:
            df[col] = 0.0
        return df

    # Reconstruct volumes from available columns
    if has_buy_vol and has_volume:
        buy_vol  = df["buy_volume"].astype(float).clip(lower=0)
        sell_vol = (df["volume"].astype(float) - buy_vol).clip(lower=0)
        total_vol = df["volume"].astype(float).clip(lower=1e-9)
    elif has_aggr and has_volume:
        total_vol = df["volume"].astype(float).clip(lower=1e-9)
        buy_vol   = df["aggressor_ratio"].astype(float).clip(0, 1) * total_vol
        sell_vol  = total_vol - buy_vol
    else:
        df[["cvd_4h_sum_trade_feature", "aggressor_ratio_4h_mean_trade_feature"]] = 0.0
        return df

    cvd_per_bar  = buy_vol - sell_vol
    aggr_per_bar = (buy_vol / total_vol).fillna(0.5)

    # Average trade size in USD (proxy for whale/large activity)
    if has_notional and has_trades:
        notional    = df["notional"].astype(float).clip(lower=0)
        trade_count = df["trade_count"].astype(float).clip(lower=1)
        avg_trade_usd = notional / trade_count
    elif has_volume:
        close = df["close"].astype(float).clip(lower=1e-9)
        avg_trade_usd = total_vol * close / df.get("trade_count", pd.Series(100, index=df.index)).clip(lower=1)
    else:
        avg_trade_usd = pd.Series(0.0, index=df.index)

    # ── Rolling aggregations (bar-level approximation of 4h tick aggregations) ──
    roll = lambda s: s.rolling(window, min_periods=1)  # noqa: E731

    # CVD features
    cvd_roll = roll(cvd_per_bar)
    cvd_mean = cvd_roll.mean().fillna(0)
    cvd_std  = cvd_roll.std().fillna(1).clip(lower=1e-9)
    df["cvd_4h_sum_trade_feature"]    = cvd_roll.sum().fillna(0)
    df["cvd_4h_zscore_trade_feature"] = ((cvd_per_bar - cvd_mean) / cvd_std).fillna(0)

    # Aggressor features
    aggr_ma = roll(aggr_per_bar).mean().fillna(0.5)
    df["aggressor_ratio_4h_mean_trade_feature"] = aggr_ma
    df["aggressor_vs_baseline_trade_feature"]   = (aggr_per_bar - aggr_ma).fillna(0)

    # Whale / large trade proxies from average trade USD size
    # "Whale" proxy: bars where avg_trade_usd is above its 90th percentile
    whale_thr = avg_trade_usd.rolling(window * 4, min_periods=window).quantile(0.90).fillna(avg_trade_usd)
    large_thr = avg_trade_usd.rolling(window * 4, min_periods=window).quantile(0.75).fillna(avg_trade_usd)
    is_whale  = (avg_trade_usd >= whale_thr).astype(float)
    is_large  = (avg_trade_usd >= large_thr).astype(float)
    df["whale_trades_4h_sum_trade_feature"]    = roll(is_whale).sum().fillna(0)
    df["large_trades_4h_sum_trade_feature"]    = roll(is_large).sum().fillna(0)
    df["max_trade_usd_4h_max_trade_feature"]   = roll(avg_trade_usd).max().fillna(0)
    df["whale_intensity_4h_mean_trade_feature"] = (is_whale * avg_trade_usd / total_vol).rolling(window, min_periods=1).mean().fillna(0)
    df["large_intensity_4h_mean_trade_feature"] = (is_large * avg_trade_usd / total_vol).rolling(window, min_periods=1).mean().fillna(0)

    # VWAP skew: close vs rolling VWAP (approximated)
    if has_notional:
        vwap = df["notional"].rolling(window, min_periods=1).sum() / total_vol.rolling(window, min_periods=1).sum().clip(lower=1e-9)
        df["vwap_skew_4h_mean_trade_feature"] = ((df["close"].astype(float) - vwap) / vwap.clip(1e-9)).fillna(0)
    else:
        df["vwap_skew_4h_mean_trade_feature"] = 0.0

    return df
