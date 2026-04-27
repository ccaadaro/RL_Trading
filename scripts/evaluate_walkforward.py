#!/usr/bin/env python3
"""
scripts/evaluate_walkforward.py
────────────────────────────────
Evaluate a frozen model across multiple OOS sub-windows so a single
point-estimate doesn't mislead us about stability.

Why this exists
───────────────
Our training flow uses a single (train, val, test) split — all metrics
reported so far are point estimates. This script doesn't retrain the
model; it just runs it on N rolling windows inside the OOS region and
reports the *distribution* of alpha / Sharpe / max-DD across folds.

Supported model types
─────────────────────
- `ppo` / `sac`   : SB3 RecurrentPPO/PPO/SAC zip + (optional)
                    VecNormalize pickle. Runs through the Layer-2
                    `signal_env` exactly like the trainers.
- `signal`        : a column name in the parquet (e.g.
                    `signal_prob_v2_cal_feature`). Runs a binary
                    threshold-long strategy on that signal, computing
                    the same alpha metric.

Usage
─────
    python scripts/evaluate_walkforward.py --model ppo \
        --model-path models/signal_rl_bull_specialist_v1/best_model.zip \
        --vecnorm    models/signal_rl_bull_specialist_v1/vecnormalize.pkl \
        --start 2024-07-01 --end 2026-03-18 \
        --window-days 90 --step-days 30

    python scripts/evaluate_walkforward.py --model signal \
        --signal-col signal_prob_v2_cal_feature \
        --threshold 0.55 --start 2024-07-01 --end 2026-03-18

Output
──────
Prints per-fold table + aggregated stats, and writes
`reports/walkforward_<tag>_<timestamp>.json`.
"""

import argparse
import json
import sys
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from utils.walkforward import oos_subwindows, summarise_folds


# ══════════════════════════════════════════════════════════════════════════════
# RL EVALUATION (PPO / SAC)
# ══════════════════════════════════════════════════════════════════════════════

def _load_rl_model(kind: str, path: str, vecnorm_path: Optional[str]):
    if kind == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(path, device="cpu")
    elif kind == "sac":
        from stable_baselines3 import SAC
        model = SAC.load(path, device="cpu")
    else:
        raise ValueError(f"Unsupported RL kind: {kind}")
    vec = None
    if vecnorm_path and Path(vecnorm_path).exists():
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        vec = vecnorm_path   # lazy-load once we have an env
    return model, vec


def _build_signal_env_vec(df_slice: pd.DataFrame,
                          reward_clip: float,
                          trade_penalty: float,
                          risk: bool,
                          vecnorm_pickle: Optional[str]):
    """Build a VecEnv mirroring what the RL saw during training.
    If vecnorm_pickle is given, load it (training=False) so obs
    normalisation matches training exactly."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from trading_env.signal_env import make_signal_env
    from trading_env.risk_wrappers import MultiLevelRiskWrapper
    from train_signal_rl import SignReward

    reward_fn = SignReward(trade_penalty=trade_penalty, mode="magnitude",
                           clip=reward_clip)

    def _make():
        env = make_signal_env(
            df_slice,
            positions=[-1, 0, 1], fee_rate=5e-4, slippage_bps=2,
            portfolio_initial_value=1000.0, initial_position=0,
            max_episode_duration="max", verbose=0,
            name="WalkfwdEval", log_frequency=9999,
            reward_function=reward_fn,
        )
        if risk:
            env = MultiLevelRiskWrapper(env, dd_hard=0.15, cooldown_steps=24)
        env = Monitor(env)
        env.reset(seed=0)
        return env

    vec = DummyVecEnv([_make])
    if vecnorm_pickle:
        # VecNormalize.load requires the pickle path AND the inner env.
        vec = VecNormalize.load(vecnorm_pickle, vec)
        vec.training = False
        vec.norm_reward = False
    else:
        # Fresh VecNormalize with training=False still applies a running-mean-of-1
        # obs norm (identity) — fine if the RL model was trained WITHOUT vecnorm.
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False,
                           clip_obs=10.0, training=False)
    return vec


def eval_rl_on_slice(model, vecnorm_pickle: Optional[str],
                     slice_df: pd.DataFrame,
                     reward_clip: float = 0.15,
                     trade_penalty: float = 5e-5,
                     risk: bool = True) -> dict:
    vec = _build_signal_env_vec(slice_df, reward_clip, trade_penalty,
                                 risk, vecnorm_pickle)
    obs = vec.reset()
    done = np.array([False])
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, done, _info = vec.step(action)
    metrics = vec.env_method("get_metrics")[0]
    pret = float(metrics.get("Portfolio Return Value", 0.0))
    mret = float(metrics.get("Market Return Value", 0.0))
    vec.close()
    return _metrics_from_slice(slice_df, pret, mret)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL THRESHOLD EVALUATION (LGBM + any signal column)
# ══════════════════════════════════════════════════════════════════════════════

def eval_signal_on_slice(slice_df: pd.DataFrame,
                         signal_col: str,
                         threshold: float,
                         fee_rate: float = 5e-4,
                         slippage: float = 2e-4,
                         funding_col: str = "funding_rate_feature",
                         funding_bars_per_pay: int = None) -> dict:
    """Binary long-only threshold strategy with explicit fees+slippage+funding.

    funding_bars_per_pay: how many bars fit in an 8h funding cycle.
      4h bars → 2, 1h bars → 8, 15m bars → 32, 5m bars → 96.
      If None, auto-detects from the median index spacing.
    """
    if signal_col not in slice_df.columns:
        raise ValueError(f"Missing {signal_col}")

    signal = slice_df[signal_col].values
    close  = slice_df["close"].values
    n      = len(slice_df)

    # Auto-detect bars-per-funding-cycle if not given
    if funding_bars_per_pay is None and n >= 2:
        median_dt_sec = pd.Series(slice_df.index).diff().dt.total_seconds().median()
        if median_dt_sec > 0:
            funding_bars_per_pay = max(1, int(round(8 * 3600 / median_dt_sec)))
        else:
            funding_bars_per_pay = 2
    funding_bars_per_pay = funding_bars_per_pay or 2

    if funding_col in slice_df.columns:
        funding_per_bar = slice_df[funding_col].values / funding_bars_per_pay
    else:
        funding_per_bar = np.zeros(n)

    pos_target = np.where(signal > threshold, 1.0, 0.0)
    portfolio  = 1.0
    position   = 0.0
    peak       = 1.0
    equity     = np.zeros(n)

    total_cost = fee_rate + slippage
    for i in range(n):
        target = pos_target[i - 1] if i > 0 else 0.0
        if target != position:
            portfolio -= abs(target - position) * total_cost * portfolio
            position = target
        if i > 0 and close[i - 1] > 0:
            portfolio *= np.exp(position * np.log(close[i] / close[i - 1]))
            portfolio -= position * funding_per_bar[i] * portfolio
        peak = max(peak, portfolio)
        equity[i] = portfolio

    port_ret_pct = (equity[-1] - 1) * 100
    mkt_ret_pct  = (close[-1] / close[0] - 1) * 100
    return _metrics_from_slice(slice_df, port_ret_pct, mkt_ret_pct, equity=equity)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _metrics_from_slice(slice_df, port_ret_pct, mkt_ret_pct, equity=None) -> dict:
    alpha = port_ret_pct - mkt_ret_pct

    # If we don't have the equity curve (RL path), synthesise a minimal one
    # for Sharpe/DD from initial→final linear interpolation.
    if equity is None:
        equity = np.linspace(1.0, 1.0 + port_ret_pct / 100, num=len(slice_df))

    log_rets = np.diff(np.log(np.maximum(equity, 1e-10)))
    # Data is 4h bars (6/day); 365*6 = 2190 bars per year for annualisation.
    BARS_PER_YEAR = 365 * 6
    sharpe   = ((log_rets.mean() / (log_rets.std() + 1e-12)) *
                np.sqrt(BARS_PER_YEAR)) if len(log_rets) > 1 else 0.0

    peak  = np.maximum.accumulate(equity)
    dd    = (peak - equity) / np.where(peak > 0, peak, 1.0)
    max_dd = float(dd.max() * 100)

    return {
        "portfolio_return": float(port_ret_pct),
        "market_return":    float(mkt_ret_pct),
        "alpha":            float(alpha),
        "sharpe":           float(sharpe),
        "max_dd":           max_dd,
        "n_bars":           int(len(slice_df)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",       type=str, required=True,
                    choices=["ppo", "sac", "signal"],
                    help="Model type to evaluate.")
    ap.add_argument("--model-path",  type=str, default=None,
                    help="Path to .zip (for ppo/sac).")
    ap.add_argument("--vecnorm",     type=str, default=None,
                    help="Path to VecNormalize .pkl (ppo/sac).")
    ap.add_argument("--signal-col",  type=str, default=None,
                    help="Parquet column name (signal mode).")
    ap.add_argument("--threshold",   type=float, default=0.55,
                    help="Long threshold (signal mode).")
    ap.add_argument("--parquet",     type=str,
                    default="cache/data_v1_430340a861af5f8f9bcbd7a4ca16ba95.parquet")
    ap.add_argument("--start",       type=str, required=True)
    ap.add_argument("--end",         type=str, required=True)
    ap.add_argument("--window-days", type=int, default=90)
    ap.add_argument("--step-days",   type=int, default=30)
    ap.add_argument("--tag",         type=str, default="",
                    help="Suffix for the report file.")
    args = ap.parse_args()

    if args.model in ("ppo", "sac") and not args.model_path:
        raise SystemExit("--model-path required for ppo/sac")
    if args.model == "signal" and not args.signal_col:
        raise SystemExit("--signal-col required for signal mode")

    print("=" * 65)
    print(f"  Walk-forward OOS sub-window evaluation")
    print(f"  Model  : {args.model}")
    print(f"  Window : {args.window_days}d  step {args.step_days}d  "
          f"[{args.start} → {args.end}]")
    print("=" * 65)

    df = pd.read_parquet(args.parquet)

    folds = oos_subwindows(df, args.start, args.end,
                           window_days=args.window_days,
                           step_days=args.step_days)
    print(f"  Folds  : {len(folds)}")

    # ── Load frozen model (RL only) ───────────────────────────────────────
    rl_model = None
    if args.model in ("ppo", "sac"):
        rl_model, _ = _load_rl_model(args.model, args.model_path, args.vecnorm)

    # ── Per-fold evaluation ───────────────────────────────────────────────
    rows = []
    print(f"\n  {'Fold':>4}  {'Start':<10}  {'End':<10}  "
          f"{'Bars':>5}  {'Mkt':>7}  {'Port':>7}  {'Alpha':>7}  "
          f"{'Sharpe':>7}  {'MaxDD':>6}  {'Regime':<8}")
    print(f"  {'─'*90}")

    for slice_df, meta in folds:
        if args.model in ("ppo", "sac"):
            m = eval_rl_on_slice(rl_model, args.vecnorm, slice_df)
        else:
            m = eval_signal_on_slice(slice_df, args.signal_col, args.threshold)

        row = {
            "fold_idx":      meta.fold_idx,
            "test_start":    str(meta.test_start.date()),
            "test_end":      str(meta.test_end.date()),
            "regime_tag":    meta.regime_tag,
            **m,
        }
        rows.append(row)
        print(f"  {meta.fold_idx:>4}  "
              f"{row['test_start']:<10}  {row['test_end']:<10}  "
              f"{m['n_bars']:>5}  "
              f"{m['market_return']:>+6.1f}%  "
              f"{m['portfolio_return']:>+6.1f}%  "
              f"{m['alpha']:>+6.1f}%  "
              f"{m['sharpe']:>+6.2f}  "
              f"{m['max_dd']:>5.1f}%  "
              f"{meta.regime_tag:<8}")

    # ── Aggregate ─────────────────────────────────────────────────────────
    summary = summarise_folds(rows)
    print(f"\n{'═'*65}")
    print(f"  Summary across {summary.get('n_folds', 0)} folds")
    print(f"{'═'*65}")
    print(f"  Alpha  : mean={summary.get('alpha_mean', 0):+6.2f}%  "
          f"std={summary.get('alpha_std', 0):5.2f}  "
          f"min={summary.get('alpha_min', 0):+6.2f}%  "
          f"max={summary.get('alpha_max', 0):+6.2f}%")
    print(f"  Sharpe : mean={summary.get('sharpe_mean', 0):+6.2f}  "
          f"std={summary.get('sharpe_std', 0):5.2f}")
    print(f"  MaxDD  : mean={summary.get('max_dd_mean', 0):5.1f}%  "
          f"max={summary.get('max_dd_max', 0):5.1f}%")
    print(f"  Consistency (% folds with α>0): "
          f"{summary.get('alpha_consistency', 0):.0%}")
    for tag in ("bull", "bear", "sideways"):
        if f"alpha_{tag}_mean" in summary:
            print(f"    {tag:<9} α̂={summary[f'alpha_{tag}_mean']:+.2f}%  "
                  f"(n={summary[f'n_folds_{tag}']})")

    # ── Persist report ────────────────────────────────────────────────────
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out_path = reports_dir / f"walkforward_{args.model}{tag}_{ts}.json"
    out_path.write_text(json.dumps({
        "model":       args.model,
        "model_path":  args.model_path,
        "signal_col":  args.signal_col,
        "threshold":   args.threshold,
        "start":       args.start,
        "end":         args.end,
        "window_days": args.window_days,
        "step_days":   args.step_days,
        "folds":       rows,
        "summary":     summary,
    }, indent=2, default=str))
    print(f"\n  Report → {out_path}")


if __name__ == "__main__":
    main()
