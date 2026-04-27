"""
utils/turbulence.py
───────────────────
Kritzman & Li (2010) Mahalanobis-distance turbulence index.

At each bar t the index measures how unusual the current multi-dimensional
return vector r_t is relative to its rolling historical distribution:

    T_t = (r_t - μ_{t-1})^T · Σ_{t-1}^{-1} · (r_t - μ_{t-1})

Values near 0 → normal; values >> 10 → turbulent / regime-change.

The index is used two ways in this project:
  1. As an observation feature so the agent knows the current market regime.
  2. As a position-sizing signal in MultiLevelRiskWrapper (risk-off when T > threshold).
"""

import numpy as np
import pandas as pd
from typing import List


# ──────────────────────────────────────────────────────────────────────────────
# default return columns (present in both train_rl.py and train2.py pipelines)
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULT_RETURN_COLS = [
    "log_return_1h_feature",
    "log_return_24h_feature",
    "volume_z_feature",
    "volatility_20_feature",
]


def compute_turbulence(
    df: pd.DataFrame,
    return_cols: List[str],
    window: int = 252,
    min_periods: int = 60,
) -> pd.Series:
    """
    Rolling Mahalanobis turbulence index.

    Parameters
    ----------
    df : DataFrame with the feature columns.
    return_cols : list of column names forming the return vector.
    window : look-back window for rolling mean/cov (bars).
    min_periods : minimum bars before producing a non-zero value.

    Returns
    -------
    pd.Series indexed like df, values in [0, ∞).
    """
    existing = [c for c in return_cols if c in df.columns]
    if not existing:
        return pd.Series(0.0, index=df.index, name="turbulence_feature")

    data = df[existing].fillna(0.0).values.astype(np.float64)
    n, k = data.shape
    turb = np.zeros(n, dtype=np.float64)

    for i in range(min_periods, n):
        start = max(0, i - window)
        hist = data[start:i]           # shape (w, k)
        curr = data[i]                 # shape (k,)

        if hist.shape[0] < 2:
            continue

        mu   = hist.mean(axis=0)
        diff = curr - mu               # shape (k,)

        try:
            if k == 1:
                var = float(np.var(hist[:, 0], ddof=1))
                turb[i] = diff[0] ** 2 / var if var > 1e-12 else 0.0
            else:
                cov     = np.cov(hist.T, ddof=1)          # (k, k)
                cov_inv = np.linalg.pinv(cov)             # pseudo-inverse
                t_val   = float(diff @ cov_inv @ diff)
                turb[i] = max(0.0, t_val)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            turb[i] = 0.0

    return pd.Series(turb, index=df.index, name="turbulence_feature")


def add_turbulence_feature(
    df: pd.DataFrame,
    return_cols: List[str] | None = None,
    window: int = 252,
    min_periods: int = 60,
    clip_pct: float = 99.0,
) -> pd.DataFrame:
    """
    Compute turbulence and append it as `turbulence_feature` to df in-place.

    The raw Mahalanobis distance is clipped at the `clip_pct` percentile of
    non-zero values and then divided by that same value, giving a normalised
    score where 1.0 ≈ historical 99th-percentile turbulence.

    Parameters
    ----------
    df : DataFrame (modified in-place and returned).
    return_cols : columns forming the return vector; defaults to standard set.
    window, min_periods : passed to compute_turbulence.
    clip_pct : percentile used for normalisation clip.
    """
    cols = return_cols or _DEFAULT_RETURN_COLS
    raw = compute_turbulence(df, cols, window=window, min_periods=min_periods)

    nonzero = raw[raw > 0]
    if len(nonzero) > 0:
        cap = float(np.percentile(nonzero, clip_pct))
        if cap > 1e-10:
            raw = (raw / cap).clip(0.0, 3.0)   # >3 = extreme; keep scale finite

    df["turbulence_feature"] = raw.fillna(0.0).values
    return df
