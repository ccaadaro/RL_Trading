"""
utils/slippage.py
─────────────────
Dynamic transaction cost estimation via the square-root market impact model.

References
──────────
  Almgren & Chriss (2001) "Optimal execution of portfolio transactions"
  Kyle (1985)             "Continuous auctions and insider trading"

Model
─────
One-way execution cost in basis points:

    cost_bps = spread_bps/2                    (half bid-ask spread, always paid)
             + η × σ_daily × √(Q / ADV) × 10k  (price impact, grows with order size)

where:
  η         — market impact coefficient (0.1 is empirically calibrated for BTC spot)
  σ_daily   — daily return standard deviation (rolling, annualised to 1-day)
  Q         — order notional in USD
  ADV       — rolling average daily USD volume

The fee_rate (exchange commission) is handled separately in TradingEnv and is NOT
included here; this module estimates slippage only.

Why this matters
────────────────
A fixed 2 bps slippage is wrong in two ways:
  1. During low-volume periods (weekends, off-hours) spreads widen to 5–15 bps.
  2. During high-volatility regimes, market makers pull liquidity — impact doubles.
Training with a dynamic estimate teaches the agent to prefer liquid, calm conditions.

Usage
─────
    from utils.slippage import add_slippage_feature

    df = add_slippage_feature(df, order_usd=5_000, adv_window=24)
    # New column: slippage_bps_feature

TradingEnv integration
──────────────────────
    In _trade(), replace the fixed self.slippage_bps with:
        if "slippage_bps_feature" in self.df.columns:
            slip = float(self.df.iloc[self._idx]["slippage_bps_feature"])
        else:
            slip = self.slippage_bps
        costs = notional * (self.fee_rate + slip / 1e4)
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Core formula
# ─────────────────────────────────────────────────────────────────────────────

def compute_market_impact_bps(
    order_usd:  float,
    adv_usd:    float,
    daily_vol:  float,
    eta:        float = 0.1,
    spread_bps: float = 1.0,
    clip_max:   float = 50.0,
) -> float:
    """
    Estimate one-way execution cost in basis points (Almgren-Chriss model).

    Parameters
    ----------
    order_usd  : notional value of the trade in USD.
    adv_usd    : rolling average daily USD volume.
    daily_vol  : rolling daily return std (not annualised — e.g. 0.02 = 2 %/day).
    eta        : market impact coefficient. 0.1 is typical for BTC/USDT on Binance.
    spread_bps : minimum half-spread in basis points (paid regardless of order size).
    clip_max   : hard cap in bps to avoid extreme values on very illiquid bars.

    Returns
    -------
    float : one-way execution cost in basis points.
    """
    if adv_usd <= 0.0:
        return clip_max
    participation = order_usd / adv_usd
    impact_bps = eta * daily_vol * np.sqrt(participation) * 10_000
    return float(np.clip(spread_bps + impact_bps, spread_bps, clip_max))


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame-level feature builder
# ─────────────────────────────────────────────────────────────────────────────

def add_slippage_feature(
    df:          pd.DataFrame,
    order_usd:   float = 1_000.0,
    adv_window:  int   = 24,       # 24 × 1 h bars = 1 trading day ADV
    vol_window:  int   = 24,       # rolling vol window (same as ADV for consistency)
    eta:         float = 0.1,
    spread_bps:  float = 1.0,
    clip_max:    float = 50.0,
) -> pd.DataFrame:
    """
    Compute bar-by-bar dynamic slippage and add as ``slippage_bps_feature``.

    All calculations are fully causal (no look-ahead): rolling windows only
    reference current and past bars.

    Parameters
    ----------
    df         : DataFrame with at least ``close`` and ``volume`` columns.
    order_usd  : assumed order size in USD for market-impact calculation.
                 Represents a «typical» position size for the backtested capital.
    adv_window : bars for rolling ADV (average daily volume in USD).
    vol_window : bars for rolling daily volatility.
    eta        : market impact coefficient (0.1 = calibrated for BTC/USDT spot).
    spread_bps : base half-spread in bps added on top of impact.
    clip_max   : maximum slippage cap in bps.

    Returns
    -------
    df with ``slippage_bps_feature`` column added in-place.
    """
    # ── Rolling ADV in USD ────────────────────────────────────────────────────
    # dollar_volume per 1h bar; ADV = 24h rolling mean
    dollar_vol = df["close"] * df["volume"]
    adv = dollar_vol.rolling(adv_window, min_periods=1).mean().clip(lower=1.0)

    # ── Rolling daily volatility ──────────────────────────────────────────────
    log_ret   = np.log(df["close"] / df["close"].shift(1).clip(lower=1e-10))
    daily_vol = log_ret.rolling(vol_window, min_periods=1).std().fillna(0.01)

    # ── Square-root impact ────────────────────────────────────────────────────
    participation = order_usd / adv                            # Q / ADV (dimensionless)
    impact_bps    = eta * daily_vol * np.sqrt(participation) * 10_000

    df["slippage_bps_feature"] = (
        (spread_bps + impact_bps)
        .clip(lower=spread_bps, upper=clip_max)
        .fillna(spread_bps)
    )

    # Diagnostics
    s = df["slippage_bps_feature"]
    pct_above_fixed = (s > 2.0).mean() * 100
    print(
        f"   Dynamic slippage (order_usd=${order_usd:,.0f}): "
        f"mean={s.mean():.2f} bps  p50={s.median():.2f} bps  "
        f"p99={s.quantile(0.99):.2f} bps  "
        f"{pct_above_fixed:.1f}% bars exceed fixed 2 bps"
    )
    return df
