"""
utils/cross_asset_features.py
──────────────────────────────
Cross-asset features that encode ETH/BTC relative dynamics.

Adding relative performance between BTC and ETH to the BTC model's observation
gives the agent information about:
  - Whether BTC is leading or lagging the broader crypto market
  - Periods of BTC dominance vs altcoin rotation
  - Cross-asset correlation regime (high correlation = risk-on, low = divergent)

All features are fully causal (no lookahead) and work with 1h candle data.

Usage
─────
    from utils.cross_asset_features import add_cross_asset_features

    btc_df = pd.read_feather("data/binance/BTC_USDT-1h.feather")
    eth_df = pd.read_feather("data/binance/ETH_USDT-1h.feather")
    btc_df = add_cross_asset_features(btc_df, eth_df)
    # New columns: eth_btc_ratio_feature, eth_btc_corr_feature,
    #              eth_rel_momentum_feature, eth_vol_ratio_feature,
    #              btc_dominance_proxy_feature
"""

import numpy as np
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_cross_asset_features(
    btc: pd.DataFrame,
    eth: pd.DataFrame,
    corr_window:      int = 48,    # 48h rolling correlation
    momentum_window:  int = 24,    # 24h relative momentum lookback
    vol_window:       int = 24,    # volume ratio smoothing
    min_periods:      int = 24,    # minimum bars before features are non-zero
) -> pd.DataFrame:
    """
    Compute cross-asset features using BTC and ETH 1h OHLCV frames.

    Parameters
    ----------
    btc : DataFrame indexed by DatetimeIndex with columns: open, high, low, close, volume
    eth : same format, same or overlapping index
    corr_window : bars for rolling Pearson correlation of log returns
    momentum_window : lookback in bars for relative momentum
    vol_window : smoothing window for relative volume

    Returns
    -------
    DataFrame indexed like *btc* with 5 new `_feature` columns.
    All NaN values are filled with 0.0 (neutral / no signal).
    """
    # Align on BTC's index
    eth_aligned = eth["close"].reindex(btc.index, method="ffill")
    eth_vol     = eth["volume"].reindex(btc.index, method="ffill")

    btc_log_ret = np.log(btc["close"] / btc["close"].shift(1).clip(1e-10))
    eth_log_ret = np.log(eth_aligned / eth_aligned.shift(1).clip(1e-10))

    out = pd.DataFrame(index=btc.index)

    # 1. ETH/BTC price ratio (normalised to rolling Z-score)
    raw_ratio = (eth_aligned / btc["close"].clip(1e-10)).replace([np.inf, -np.inf], np.nan)
    ratio_mean = raw_ratio.rolling(corr_window, min_periods=min_periods).mean()
    ratio_std  = raw_ratio.rolling(corr_window, min_periods=min_periods).std().clip(1e-9)
    out["eth_btc_ratio_feature"] = ((raw_ratio - ratio_mean) / ratio_std).fillna(0.0).clip(-3, 3)

    # 2. Rolling Pearson correlation of log returns (−1 to +1)
    out["eth_btc_corr_feature"] = (
        btc_log_ret.rolling(corr_window, min_periods=min_periods)
        .corr(eth_log_ret)
        .fillna(0.0)
        .clip(-1.0, 1.0)
    )

    # 3. Relative momentum: ETH − BTC cumulative return over lookback
    btc_ret = btc["close"].pct_change(momentum_window)
    eth_ret = eth_aligned.pct_change(momentum_window)
    out["eth_rel_momentum_feature"] = (eth_ret - btc_ret).fillna(0.0).clip(-1.0, 1.0)

    # 4. ETH/BTC relative volume (normalised)
    btc_vol_roll = btc["volume"].rolling(vol_window, min_periods=min_periods).mean().clip(1e-9)
    eth_vol_roll = eth_vol.rolling(vol_window, min_periods=min_periods).mean().clip(1e-9)
    raw_vol_ratio = (eth_vol / btc["volume"].clip(1e-9)).replace([np.inf, -np.inf], np.nan)
    vol_r_mean = raw_vol_ratio.rolling(corr_window, min_periods=min_periods).mean()
    vol_r_std  = raw_vol_ratio.rolling(corr_window, min_periods=min_periods).std().clip(1e-9)
    out["eth_vol_ratio_feature"] = ((raw_vol_ratio - vol_r_mean) / vol_r_std).fillna(0.0).clip(-3, 3)

    # 5. BTC dominance proxy: BTC is "dominant" when BTC return > ETH return
    #    Smooth with EMA to capture regime, not noise
    btc_winning = (btc_log_ret > eth_log_ret).astype(float)
    out["btc_dominance_proxy_feature"] = (
        btc_winning.ewm(span=corr_window, min_periods=min_periods).mean()
        .fillna(0.5)
        .clip(0.0, 1.0)
    )

    return out


def add_cross_asset_features(
    btc_df: pd.DataFrame,
    eth_df: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """
    Compute cross-asset features and join them into *btc_df* in-place.

    Parameters
    ----------
    btc_df : BTC DataFrame to enrich (modified in-place and returned).
    eth_df : ETH reference DataFrame (read-only).
    **kwargs : passed to compute_cross_asset_features().

    Returns
    -------
    btc_df with 5 new feature columns added.
    """
    cross = compute_cross_asset_features(btc_df, eth_df, **kwargs)

    for col in cross.columns:
        btc_df[col] = cross[col].values

    existing = [c for c in btc_df.columns if "feature" in c]
    print(
        f"   Cross-asset features added: {list(cross.columns)}\n"
        f"   Total feature cols now: {len(existing)}"
    )
    return btc_df


# ─────────────────────────────────────────────────────────────────────────────
# Convenience loader
# ─────────────────────────────────────────────────────────────────────────────

def load_eth_df(eth_path: str = "../../data/binance/ETH_USDT-1h.feather") -> pd.DataFrame:
    """
    Load ETH/USDT 1h candles from the default Freqtrade data path.

    Returns empty DataFrame (no cross-asset features) if file is missing.
    """
    from pathlib import Path
    p = Path(eth_path)
    if not p.exists():
        print(f"   [WARN] ETH data not found at {p} — cross-asset features skipped")
        return pd.DataFrame()

    df = pd.read_feather(str(p))
    if "date" in df.columns:
        df = df.set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    return df
