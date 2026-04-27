#!/usr/bin/env python3
"""
backtest_threshold.py — Threshold strategy backtest on LightGBM signal
═══════════════════════════════════════════════════════════════════════

Tests the high-confidence filter:
  signal_prob > THRESH  → LONG  1.0
  signal_prob < 1-THRESH → SHORT (optional)
  otherwise              → FLAT

Reports realistic P&L including:
  - Trading fees (0.05% per side)
  - Slippage (2 bps per side)
  - Max drawdown
  - Sharpe ratio (annualized)
  - Win rate, trade count

Also tests sizing variants:
  - Binary: full in/out at threshold
  - Proportional: size = (|prob - 0.5| - threshold + 0.5) / 0.5  (scales 0→1 above threshold)
  - RL-gated: use RL model only on high-confidence bars
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH   = "cache/data_v1_430340a861af5f8f9bcbd7a4ca16ba95.parquet"
TRAIN_END   = "2024-06-30"
VAL_START   = "2024-07-01"
VAL_END     = "2025-06-30"
TEST_START  = "2025-07-01"

FEE_RATE    = 5e-4    # 0.05% per side (taker)
SLIPPAGE    = 2e-4    # 2 bps per side
TOTAL_COST  = FEE_RATE + SLIPPAGE   # one-way cost
# Funding rate column (8h funding resampled to 4h, forward-filled from the
# original feed). The raw rate is the per-8h charge — we expense position * rate every bar
# at which funding rate is observed. Keeping the sign convention of the
# Binance feed: positive rate ⇒ longs pay shorts.
FUNDING_COL          = "funding_rate_feature"
# Funding settles every 8h on Binance perps. At the parquet's 4h bar
# cadence, that's 2 bars per funding settlement.
FUNDING_BARS_PER_PAY = 2

# Signal column preference — v2 (58 computable features) > v1 legacy.
SIGNAL_CANDIDATES = ["signal_prob_v2_feature", "signal_prob_1h_feature"]

THRESHOLDS  = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62]
ALLOW_SHORT = False   # True to test long/short, False for long/flat only

BARS_PER_YEAR = 365 * 6


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def backtest(df: pd.DataFrame, threshold: float, allow_short: bool = False,
             sizing: str = "binary") -> dict:
    """
    Simulate the threshold strategy on a single DataFrame split.

    Parameters
    ----------
    df         : DataFrame with signal_prob_1h_feature and close columns
    threshold  : minimum P(up) to go long (e.g. 0.60)
    allow_short: if True, go short when P(up) < 1-threshold
    sizing     : 'binary'       → full position at threshold
                 'proportional' → scale position by confidence above threshold
    """
    signal_col = next((c for c in SIGNAL_CANDIDATES if c in df.columns), None)
    if signal_col is None:
        raise ValueError(
            f"No signal column in DataFrame. Expected one of {SIGNAL_CANDIDATES}."
        )
    signal  = df[signal_col].values
    close   = df["close"].values
    # Funding rate: per-bar charge applied to current position (long pays
    # when positive). Zero if feature is missing so the CI still runs on
    # old parquets — but the summary prints a warning in that case.
    if FUNDING_COL in df.columns:
        funding_per_bar = df[FUNDING_COL].values / FUNDING_BARS_PER_PAY
    else:
        funding_per_bar = np.zeros(len(df))
    n      = len(df)

    # Compute target positions
    if sizing == "binary":
        pos = np.where(signal > threshold, 1.0,
              np.where(allow_short & (signal < 1 - threshold), -1.0, 0.0))
    else:  # proportional
        long_conf  = np.clip((signal - threshold) / (1 - threshold), 0, 1)
        short_conf = np.clip(((1 - threshold) - signal) / (1 - threshold), 0, 1) if allow_short else 0
        pos = long_conf - short_conf

    # Simulate bar-by-bar
    portfolio = 1.0         # normalized to 1
    position  = 0.0         # current position
    peak_val  = 1.0

    equity    = np.zeros(n)
    positions = np.zeros(n)
    trades    = []

    for i in range(n):
        # signal[i] predicts close[i+1]/close[i].
        # So we enter at close[i] (end of bar i) to capture bar i+1's return.
        # In the loop: at bar i we use pos[i-1] as the active position,
        # because pos[i-1] was decided when bar i-1 closed (= entry at close[i-1]).
        target = pos[i - 1] if i > 0 else 0

        if target != position:
            exec_price = close[i - 1] if i > 0 else close[0]
            delta = abs(target - position)
            cost  = delta * TOTAL_COST * portfolio
            portfolio -= cost
            if len(trades) == 0 or trades[-1]["exit_i"] is not None:
                if target != 0:
                    trades.append({"entry_i": i, "entry_price": exec_price,
                                   "side": 1 if target > 0 else -1,
                                   "exit_i": None, "exit_price": None})
            else:
                trades[-1]["exit_i"] = i
                trades[-1]["exit_price"] = exec_price
            position = target

        # Mark to market at close[i] using the ACTIVE position during bar i
        if i > 0:
            log_ret = np.log(close[i] / close[i - 1]) if close[i - 1] > 0 else 0
            portfolio *= np.exp(position * log_ret)
            # Funding: longs pay when rate > 0, shorts receive (and vice versa).
            # funding_per_bar is the 4h funding carry derived from the 8h rate.
            portfolio -= position * funding_per_bar[i] * portfolio

        peak_val      = max(peak_val, portfolio)
        equity[i]     = portfolio
        positions[i]  = position

    # Final trade close
    if trades and trades[-1]["exit_i"] is None:
        trades[-1]["exit_i"] = n - 1
        trades[-1]["exit_price"] = close[-1]

    # Metrics
    total_return   = (equity[-1] - 1) * 100
    bh_return      = (close[-1] / close[0] - 1) * 100
    alpha          = total_return - bh_return
    max_dd         = max((peak_val - v) / peak_val for v in equity) * 100 if len(equity) > 0 else 0

    log_rets = np.diff(np.log(np.maximum(equity, 1e-10)))
    sharpe   = (log_rets.mean() / (log_rets.std() + 1e-12)) * np.sqrt(BARS_PER_YEAR) if len(log_rets) > 1 else 0

    n_trades    = len([t for t in trades if t["exit_i"] is not None])
    coverage    = np.mean(np.abs(positions) > 0.01) * 100

    # Direction accuracy
    in_market   = np.abs(positions) > 0.01
    mkt_returns = np.diff(np.log(np.maximum(close, 1e-10)))
    if in_market[1:].sum() > 0:
        signed_returns = np.sign(mkt_returns) * np.sign(positions[1:])
        accuracy = np.mean(signed_returns[in_market[1:]] > 0) * 100
    else:
        accuracy = 50.0

    # Win rate (per trade)
    win_trades = 0
    for t in trades:
        if t["exit_i"] is None:
            continue
        entry_p = t["entry_price"]
        exit_p  = t["exit_price"]
        if entry_p > 0:
            trade_ret = (exit_p / entry_p - 1) * t["side"]
            if trade_ret > 0:
                win_trades += 1

    win_rate = win_trades / max(n_trades, 1) * 100

    return {
        "total_return": total_return,
        "bh_return":    bh_return,
        "alpha":        alpha,
        "max_dd":       max_dd,
        "sharpe":       sharpe,
        "n_trades":     n_trades,
        "coverage":     coverage,
        "accuracy":     accuracy,
        "win_rate":     win_rate,
        "equity":       equity,
        "positions":    positions,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df = pd.read_parquet(DATA_PATH)

    signal_col = next((c for c in SIGNAL_CANDIDATES if c in df.columns), None)
    if signal_col is None:
        raise RuntimeError(
            f"No signal column found. Expected one of {SIGNAL_CANDIDATES}. "
            "Run scripts/retrain_signal_v2.py first."
        )
    print(f"  Signal column: {signal_col}")
    if FUNDING_COL not in df.columns:
        print(f"  [WARN] {FUNDING_COL} missing — funding costs set to 0 "
              f"(backtest optimistic on shorts/longs in funding regimes).")

    splits = {
        "TRAIN": df[:TRAIN_END],
        "VAL":   df[VAL_START:VAL_END],
        "TEST":  df[TEST_START:],
    }

    for sizing in ["binary", "proportional"]:
        print(f"\n{'═'*80}")
        print(f"  SIZING: {sizing.upper()}   |   allow_short={ALLOW_SHORT}")
        print(f"{'═'*80}")

        for split_name, df_split in splits.items():
            print(f"\n  ── {split_name} ({len(df_split)} bars) ──────────────────────────")
            print(f"  {'Thresh':>7}  {'Return':>8}  {'B&H':>8}  {'Alpha':>8}  "
                  f"{'MaxDD':>7}  {'Sharpe':>7}  {'Trades':>7}  {'Cover':>7}  {'Acc':>6}  {'WinR':>6}")
            print(f"  {'─'*85}")

            for thresh in THRESHOLDS:
                r = backtest(df_split, threshold=thresh,
                             allow_short=ALLOW_SHORT, sizing=sizing)
                print(f"  {thresh:>7.2f}  "
                      f"{r['total_return']:>+7.1f}%  "
                      f"{r['bh_return']:>+7.1f}%  "
                      f"{r['alpha']:>+7.1f}%  "
                      f"{r['max_dd']:>6.1f}%  "
                      f"{r['sharpe']:>+6.2f}  "
                      f"{r['n_trades']:>7d}  "
                      f"{r['coverage']:>6.1f}%  "
                      f"{r['accuracy']:>5.1f}%  "
                      f"{r['win_rate']:>5.1f}%")

    # ── Best threshold deep dive ──────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"  DEEP DIVE — Threshold 0.60, binary, long-only")
    print(f"{'═'*80}")
    for split_name, df_split in splits.items():
        r = backtest(df_split, threshold=0.60, allow_short=False, sizing="binary")
        print(f"\n  [{split_name}]")
        print(f"    Portfolio return : {r['total_return']:+.2f}%")
        print(f"    Buy & Hold       : {r['bh_return']:+.2f}%")
        print(f"    Alpha            : {r['alpha']:+.2f}%")
        print(f"    Max Drawdown     : {r['max_dd']:.1f}%")
        print(f"    Sharpe (ann.)    : {r['sharpe']:+.3f}")
        print(f"    # Trades         : {r['n_trades']}")
        print(f"    Coverage         : {r['coverage']:.1f}% of bars")
        print(f"    Direction acc.   : {r['accuracy']:.1f}%")
        print(f"    Win rate         : {r['win_rate']:.1f}%")


if __name__ == "__main__":
    main()
