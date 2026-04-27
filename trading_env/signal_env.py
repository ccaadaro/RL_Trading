"""
trading_env/signal_env.py
─────────────────────────
Simplified RL environment for Layer 2 of the two-layer architecture:

    LightGBM → P(up) → [this env] → position size

The RL agent does NOT see the raw market features. Those are fully
encapsulated in the LightGBM signal columns. The agent only sees:

  Market context (5 static features from the parquet):
    1. signal_score_feature      — monotonic RL score derived from the raw
                                    probability by scripts/calibrate_signal_v2.py
    2. signal_prob_feature       — raw LightGBM P(up) for the next bar
    3. trend_return_180_feature  — bull/bear regime context (lagged, no leakage)
    4. volatility_180_feature    — long-term vol regime
    5. signal_confidence_feature — |score - 0.5| * 2  in [0,1]

  Portfolio state (4 dynamic features computed at runtime):
    6. last_position             — current exposure [-1, 1]
    7. drawdown                  — current drawdown from episode peak
    8. bars_in_trade             — steps since last position change (normalized)
    9. real_position             — current realized exposure

Total: 9 features. A 10× reduction from 104 — far easier to optimize.
"""

import numpy as np
import pandas as pd
from typing import List, Optional

from trading_env.trading_env import (
    TradingEnv,
    differential_sharpe_reward,
    dynamic_feature_last_position_taken,
    dynamic_feature_real_position,
    dynamic_feature_sharpe_A,
    dynamic_feature_sharpe_B,
    History,
)

# ══════════════════════════════════════════════════════════════════════════════
# Feature columns passed from parquet to the RL agent
# ══════════════════════════════════════════════════════════════════════════════

# Canonical columns seen by the RL env. `prepare_signal_df` maps the
# parquet's versioned raw-probability and stretched-score columns to these
# names so downstream code stays version-agnostic.
SIGNAL_SCORE_COL = "signal_score_feature"
SIGNAL_PROB_COL = "signal_prob_feature"

# Order of preference for the RL score column.
_SIGNAL_SCORE_CANDIDATES = [
    "signal_score_v2_feature",
    "signal_prob_v2_cal_feature",  # legacy alias; semantics are score, not probability
]

# Order of preference for the raw probability column.
_SIGNAL_PROB_CANDIDATES = [
    "signal_prob_v2_feature",
    "signal_prob_1h_feature",
]

SIGNAL_FEATURE_COLS = [
    SIGNAL_PROB_COL,              # primary signal (version-agnostic alias)
    "trend_return_180_feature",   # regime context
    "volatility_180_feature",     # vol regime context
]

# Fallback names to check if preferred columns are missing
_REGIME_FALLBACKS = ["trend_return_90_feature", "ma_bias_540_feature"]
_VOL_FALLBACKS    = ["volatility_90_feature", "volatility_42_feature", "volatility_24_feature"]


# ══════════════════════════════════════════════════════════════════════════════
# New dynamic feature functions
# ══════════════════════════════════════════════════════════════════════════════

def dynamic_feature_signal_confidence(history: History) -> float:
    """Confidence of the LightGBM signal: |P(up) - 0.5| * 2, in [0, 1].

    0.0 → completely uncertain (prob ≈ 0.5)
    1.0 → maximally confident (prob ≈ 0.0 or 1.0)
    """
    try:
        # Signal prob is the first static feature column (canonical alias).
        signal_prob = float(history[f"data_{SIGNAL_SCORE_COL}", -1])
    except (KeyError, IndexError):
        return 0.0
    return float(abs(signal_prob - 0.5) * 2.0)


def dynamic_feature_drawdown(history: History) -> float:
    """Current drawdown from the episode peak, in [0, 1].

    0.0 → at or above all-time episode high
    0.1 → 10% below peak
    """
    vals = history["portfolio_valuation"]   # all steps so far
    if len(vals) == 0:
        return 0.0
    peak    = float(np.max(vals))
    current = float(vals[-1])
    if peak <= 1e-10:
        return 0.0
    return float(np.clip((peak - current) / peak, 0.0, 1.0))


def dynamic_feature_bars_in_trade(history: History) -> float:
    """Number of consecutive bars in the current position, normalized by 100.

    Helps the agent reason about trade duration and mean-reversion timing.
    """
    if len(history) < 2:
        return 0.0
    pos_history = history["position_index"]   # all steps
    current_pos = int(pos_history[-1])
    count = 0
    for i in range(len(pos_history) - 1, -1, -1):
        if int(pos_history[i]) == current_pos:
            count += 1
        else:
            break
    return float(count) / 100.0   # ~1.0 after 100 bars in same position


# ══════════════════════════════════════════════════════════════════════════════
# DataFrame preparation
# ══════════════════════════════════════════════════════════════════════════════

def prepare_signal_df(df: pd.DataFrame) -> pd.DataFrame:
    """Strip all market features except the signal columns.

    Keeps: OHLCV, signal_score_feature, signal_prob_feature,
           trend_return_180_feature, volatility_180_feature
           (with fallbacks if missing).

    The canonical `signal_prob_feature` is sourced from the raw probability
    column (`signal_prob_v2_feature` preferred; `signal_prob_1h_feature`
    legacy). The canonical `signal_score_feature` is sourced from the
    stretched RL score (`signal_score_v2_feature` preferred; legacy alias
    `signal_prob_v2_cal_feature` accepted).

    Returns a DataFrame where the only `*feature*` columns are the 3
    signal context features — so TradingEnv's _set_df picks them up cleanly.
    """
    df = df.copy()

    keep_features = []

    # Locate raw probability column and rename to canonical name.
    prob_col = next((c for c in _SIGNAL_PROB_CANDIDATES if c in df.columns), None)
    if prob_col is None:
        raise ValueError(
            f"No raw probability column found. Expected one of {_SIGNAL_PROB_CANDIDATES}. "
            "Run scripts/retrain_signal_v2.py (preferred) or "
            "scripts/train_signal_model.py (legacy) first."
        )
    if prob_col != SIGNAL_PROB_COL:
        df[SIGNAL_PROB_COL] = df[prob_col]
    print(f"   Signal env: using raw prob {prob_col} → {SIGNAL_PROB_COL}")
    keep_features.append(SIGNAL_PROB_COL)

    # Locate RL score column; if missing, fall back to the raw probability.
    score_col = next((c for c in _SIGNAL_SCORE_CANDIDATES if c in df.columns), None)
    if score_col is None:
        score_col = prob_col
    if score_col != SIGNAL_SCORE_COL:
        df[SIGNAL_SCORE_COL] = df[score_col]
    print(f"   Signal env: using RL score {score_col} → {SIGNAL_SCORE_COL}")
    keep_features.append(SIGNAL_SCORE_COL)

    # Regime context — try preferred, then fallbacks
    regime_col = None
    for col in ["trend_return_180_feature"] + _REGIME_FALLBACKS:
        if col in df.columns:
            regime_col = col
            break
    if regime_col and regime_col != "trend_return_180_feature":
        df["trend_return_180_feature"] = df[regime_col]
    if regime_col:
        keep_features.append("trend_return_180_feature")

    # Volatility context — try preferred, then fallbacks
    vol_col = None
    for col in ["volatility_180_feature"] + _VOL_FALLBACKS:
        if col in df.columns:
            vol_col = col
            break
    if vol_col and vol_col != "volatility_180_feature":
        df["volatility_180_feature"] = df[vol_col]
    if vol_col:
        keep_features.append("volatility_180_feature")

    # Derive signal_confidence as a static feature (no need for dynamic function)
    df["signal_confidence_feature"] = (
        (df[SIGNAL_SCORE_COL] - 0.5).abs() * 2.0
    ).clip(0.0, 1.0)
    keep_features.append("signal_confidence_feature")

    # Drop all other feature columns — keep only OHLCV + kept features
    all_feature_cols = [c for c in df.columns if "feature" in c]
    drop_cols = [c for c in all_feature_cols if c not in keep_features]
    df = df.drop(columns=drop_cols)

    n_kept = len([c for c in df.columns if "feature" in c])
    print(f"   Signal env: kept {n_kept} feature cols → {keep_features}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def make_signal_env(
    df: pd.DataFrame,
    positions: list = [-1, 0, 1],
    trading_fees: float = 0,
    fee_rate: float = 5e-4,
    slippage_bps: float = 2,
    borrow_interest_rate: float = 0,
    portfolio_initial_value: float = 1000,
    initial_position: float = 0,
    max_episode_duration = "max",
    verbose: int = 1,
    name: str = "SignalEnv",
    log_frequency: int = 10,
    reward_function = None,
) -> TradingEnv:
    """
    Create a TradingEnv where the RL agent only sees:
      - 3 signal/context features from the parquet
      - 5 portfolio-state dynamic features

    Parameters mirror TradingEnv. reward_function defaults to
    differential_sharpe_reward (demean_market=True).
    """
    if reward_function is None:
        import functools
        reward_function = functools.partial(
            differential_sharpe_reward,
            eta=0.01,
            fee_penalty=2e-4,
            hold_penalty=0.0,
            demean_market=True,
        )

    filtered_df = prepare_signal_df(df)

    dynamic_features = [
        dynamic_feature_last_position_taken,
        dynamic_feature_real_position,
        dynamic_feature_drawdown,
        dynamic_feature_bars_in_trade,
    ]

    env = TradingEnv(
        df                    = filtered_df,
        positions             = positions,
        dynamic_feature_functions = dynamic_features,
        reward_function       = reward_function,
        windows               = None,
        trading_fees          = trading_fees,
        fee_rate              = fee_rate,
        slippage_bps          = slippage_bps,
        borrow_interest_rate  = borrow_interest_rate,
        portfolio_initial_value = portfolio_initial_value,
        initial_position      = initial_position,
        max_episode_duration  = max_episode_duration,
        verbose               = verbose,
        name                  = name,
        log_frequency         = log_frequency,
    )

    return env
