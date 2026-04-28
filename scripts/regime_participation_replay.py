"""
Regime-participation prototype.

Sizing-based architecture (vs gate-based):
    base_exposure = f(HMM regime, trend filter)
    alpha_multiplier = g(alpha_prob)
    target = base_exposure * alpha_multiplier
    kill-switch: target=0 if turbulence_pct > 0.95 or regime == panic_selloff

Goal: capture part of bull-trend upside that gate-based architecture
strangled, with bounded drawdown.

Reuses walk-forward HMM, simulator, and summary helpers from
scripts/global_alpha_replay.py to guarantee identical accounting.
"""
import sys
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

_HERE = Path(__file__).resolve().parent.parent
sys.path.append(str(_HERE))
sys.path.append(str(_HERE / "scripts"))

from utils.signal_features import SIGNAL_FEAT_COLS_V2, FEATURE_SET_INSTITUTIONAL
from utils.risk_directors import MahalanobisTurbulence
from global_alpha_replay import (
    _walk_forward_hmm, _simulate, _summarize,
    TURB_WINDOW, TURB_PCTILE_WIN,
)


# ─── Regime → exposure mapping ──────────────────────────────────────────────
REGIME_BASE_EXPOSURE = {
    "bull_calm":         0.75,
    "bull_neutral":      0.40,
    "high_vol_rebound":  0.20,
    "bear_calm":         0.00,
    "bear_neutral":      0.00,
    "panic_selloff":     0.00,
    "unknown":           0.00,
}

# Alpha multiplier bounds per regime (clip after linear map of alpha_prob)
ALPHA_MULT_BOUNDS = {
    "bull_calm":         (0.7, 1.5),
    "bull_neutral":      (0.5, 1.3),
    "high_vol_rebound":  (0.0, 1.0),
    "bear_calm":         (0.0, 0.0),
    "bear_neutral":      (0.0, 0.0),
    "panic_selloff":     (0.0, 0.0),
    "unknown":           (0.0, 0.0),
}


def _semantic_regime(canonical: int, recent_ret: float) -> str:
    if canonical == 0:
        return "bull_calm" if recent_ret >= 0 else "bear_calm"
    if canonical == 1:
        return "bull_neutral" if recent_ret >= 0 else "bear_neutral"
    if canonical == 2:
        return "high_vol_rebound" if recent_ret >= 0 else "panic_selloff"
    return "unknown"


def _alpha_multiplier(alpha_prob: np.ndarray, regime_labels: np.ndarray,
                      lo_anchor: float = 0.45, hi_anchor: float = 0.65) -> np.ndarray:
    """Linear map alpha_prob ∈ [lo, hi] → 1.0 ± k, clipped per-regime bounds."""
    n = len(alpha_prob)
    # Center map at 1.0 when alpha = (lo+hi)/2
    span = hi_anchor - lo_anchor
    raw = 0.5 + (alpha_prob - lo_anchor) / span * 1.5  # alpha=lo→0.5, mid→1.25, hi→2.0
    out = np.zeros(n)
    for r, (lo_b, hi_b) in ALPHA_MULT_BOUNDS.items():
        mask = regime_labels == r
        out[mask] = np.clip(raw[mask], lo_b, hi_b)
    return out


def _capture_ratio(strat_pnl: np.ndarray, market_pnl: np.ndarray, mask: np.ndarray):
    if mask.sum() == 0:
        return 0.0
    s = (1 + strat_pnl[mask]).prod() - 1
    m = (1 + market_pnl[mask]).prod() - 1
    return s / m if m != 0 else 0.0


def run(model_path: str, data_path: str, kill_turb_pct: float = 0.95,
        max_target: float = 1.0, target_smooth_span: int = 100,
        rebal_threshold: float = 0.20):
    print(f"Loading data: {data_path}")
    df = pd.read_feather(data_path)
    print(f"Loading alpha: {model_path}")
    alpha_model = lgb.Booster(model_file=model_path)

    nfeat = len(alpha_model.feature_name())
    if nfeat == 21:
        features, model_name = SIGNAL_FEAT_COLS_V2, "Elite v2.1"
    elif nfeat == 14:
        features, model_name = FEATURE_SET_INSTITUTIONAL, "Institutional Base"
    else:
        features = [c for c in df.columns
                    if c.endswith("_feature") and c in alpha_model.feature_name()]
        model_name = "Unknown"

    df["alpha_prob"] = alpha_model.predict(df[features])

    print("Computing turbulence + walk-forward HMM...")
    turb_engine = MahalanobisTurbulence(window=TURB_WINDOW)
    risk_feats = ["log_return_feature", "volatility_24_feature", "intraday_range_feature"]
    df["turbulence_score"] = turb_engine.compute(df, [c for c in risk_feats if c in df.columns]) \
        .bfill().fillna(5.0)
    df["turbulence_pct"] = df["turbulence_score"].rolling(TURB_PCTILE_WIN).rank(pct=True)

    hmm_feats = ["log_return_feature", "volatility_24_feature", "aggressor_ratio",
                 "intraday_range_feature"]
    df["hmm_canonical"] = _walk_forward_hmm(df, [c for c in hmm_feats if c in df.columns])

    # Semantic regime label using rolling 200-bar mean log return (stable label)
    if "log_return_feature" in df.columns:
        rec_ret = df["log_return_feature"].rolling(200).mean().fillna(0).values
    else:
        rec_ret = df["close"].pct_change(200).fillna(0).values
    canonical = df["hmm_canonical"].values
    regime = np.array([_semantic_regime(int(c), float(r)) for c, r in zip(canonical, rec_ret)])
    df["regime"] = regime

    # Trend filter: close > EMA200 AND EMA200 slope (50-bar diff) > 0
    ema200 = df["close"].ewm(span=200, adjust=False).mean()
    ema_slope_pos = (ema200.diff(50) > 0).fillna(False).values
    above_ema = (df["close"] > ema200).values
    trend_ok = above_ema & ema_slope_pos

    # Base exposure from regime, GATED by trend filter for bull regimes
    base = np.array([REGIME_BASE_EXPOSURE.get(r, 0.0) for r in regime])
    bull_mask = np.isin(regime, ["bull_calm", "bull_neutral"])
    base[bull_mask & ~trend_ok] = 0.0  # gate, not halve

    # Smooth the regime-driven base before alpha modulation, to absorb
    # transient regime-label flips around the canonical-state boundary.
    base = pd.Series(base).ewm(span=200, adjust=False).mean().values

    # Alpha multiplier with regime-dependent bounds
    mult = _alpha_multiplier(df["alpha_prob"].values, regime)

    # Combined target
    target_raw = np.clip(base * mult, 0.0, max_target)

    # Kill switch (applied before smoothing — must be sharp)
    kill = (df["turbulence_pct"].fillna(0).values > kill_turb_pct) | (regime == "panic_selloff")
    target_raw[kill] = 0.0

    # Anti-churn smoothing: EWM the target so micro-fluctuations in alpha
    # multiplier and regime label don't trigger constant rebalances.
    target = pd.Series(target_raw).ewm(span=target_smooth_span, adjust=False).mean().values
    # Discretize to coarse exposure buckets (0%, 10%, 25%, 50%, 75%, 100%)
    buckets = np.array([0.0, 0.10, 0.25, 0.50, 0.75, 1.0])
    target = buckets[np.argmin(np.abs(target[:, None] - buckets[None, :]), axis=1)]
    # Re-apply kill switch after smoothing/discretization
    target[kill] = 0.0

    # ─── Diagnostics ────────────────────────────────────────────────────────
    print("\n=== Regime distribution ===")
    rd = pd.Series(regime).value_counts(normalize=True).sort_index()
    for r, p in rd.items():
        print(f"  {r:20s}  {p:6.1%}  base={REGIME_BASE_EXPOSURE.get(r, 0):.2f}")

    print(f"\n  trend_ok bars                : {trend_ok.sum()} ({trend_ok.mean():.1%})")
    print(f"  kill-switch fired (bars)     : {kill.sum()} ({kill.mean():.1%})")
    print(f"  target == 0 bars             : {(target == 0).sum()} ({(target == 0).mean():.1%})")
    print(f"  target > 0.5 bars            : {(target > 0.5).sum()} ({(target > 0.5).mean():.1%})")
    print(f"  mean target exposure         : {target.mean():.3f}")

    # ─── Simulation ─────────────────────────────────────────────────────────
    bar_return = df["close"].pct_change().fillna(0).values
    cost_rate = 0.0005 + 0.0002

    if "date" in df.columns:
        total_days = (df["date"].max() - df["date"].min()).total_seconds() / 86400
    else:
        total_days = len(df) / 17.6 / 24
    total_months = total_days / 30.44

    print("\n=== Strategy & Benchmarks ===")
    actual, cum_strat, pnl_strat, td_strat = _simulate(target, bar_return, cost_rate, rebal_threshold=rebal_threshold)
    s_strat = _summarize(f"Regime-Participation ({model_name})",
                         cum_strat, pnl_strat, td_strat, total_months)

    bh = np.ones(len(df)); bh[0] = 0
    _, cum_bh, pnl_bh, td_bh = _simulate(bh, bar_return, cost_rate, rebal_threshold=rebal_threshold)
    s_bh = _summarize("Buy & Hold", cum_bh, pnl_bh, td_bh, total_months)

    # Trend-only (no alpha) — sanity benchmark using just regime base * trend
    trend_only_target = np.clip(base, 0.0, max_target)
    trend_only_target[kill] = 0.0
    trend_only_target = pd.Series(trend_only_target).ewm(span=target_smooth_span, adjust=False).mean().values
    trend_only_target = buckets[np.argmin(np.abs(trend_only_target[:, None] - buckets[None, :]), axis=1)]
    trend_only_target[kill] = 0.0
    _, cum_trend, pnl_trend, td_trend = _simulate(trend_only_target, bar_return, cost_rate)
    s_trend = _summarize("Regime-only (no alpha mult)",
                         cum_trend, pnl_trend, td_trend, total_months)

    # Calmar
    def _calmar(s):
        return s["roi"] / abs(s["dd"]) if s["dd"] != 0 else 0.0
    print(f"\n  Calmar Strategy: {_calmar(s_strat):+.3f}  | B&H: {_calmar(s_bh):+.3f}  "
          f"| Regime-only: {_calmar(s_trend):+.3f}")

    # Capture ratio in bull regimes (vs B&H)
    bull_idx = np.isin(regime, ["bull_calm", "bull_neutral"])
    cr_bull = _capture_ratio(pnl_strat, pnl_bh, bull_idx)
    bear_idx = np.isin(regime, ["bear_calm", "bear_neutral", "panic_selloff"])
    cr_bear = _capture_ratio(pnl_strat, pnl_bh, bear_idx)
    print(f"  Bull capture ratio: {cr_bull:+.2%}   Bear capture ratio: {cr_bear:+.2%}")

    # ─── Plot ───────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    plt.figure(figsize=(13, 7))
    x = np.arange(len(df))
    plt.plot(x, cum_bh, label=f"B&H ({s_bh['roi']:+.1%})",
             color="gray", linestyle="--", alpha=0.6)
    plt.plot(x, cum_trend, label=f"Regime-only ({s_trend['roi']:+.1%})",
             color="orange", alpha=0.8)
    plt.plot(x, cum_strat, label=f"Regime+Alpha ({s_strat['roi']:+.1%})",
             color="blue", linewidth=2)
    cum_max = np.maximum.accumulate(cum_strat)
    plt.fill_between(x, cum_strat, cum_max, color="red", alpha=0.15, label="Strat DD")
    plt.title(f"Regime-Participation Replay: {Path(model_path).name}\n"
              f"ROI {s_strat['roi']:+.2%} | Calmar {_calmar(s_strat):+.2f} | "
              f"Bull capture {cr_bull:+.1%} | Trades {s_strat['trades']}")
    plt.xlabel("Dollar Bars"); plt.ylabel("Cumulative Return")
    plt.legend(loc="upper left"); plt.grid(True, alpha=0.2)
    Path("reports").mkdir(exist_ok=True)
    out = Path("reports") / f"regime_participation_{Path(model_path).stem}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"\nEquity curves saved to {out}")

    return s_strat, s_bh, s_trend


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="models/dollar_alpha_v1/latest_model.txt")
    p.add_argument("--data", type=str,
                   default="cache/dollar_bars_btc_2000000_features.feather")
    p.add_argument("--kill-turb-pct", type=float, default=0.95)
    p.add_argument("--max-target", type=float, default=1.0)
    p.add_argument("--smooth-span", type=int, default=100,
                   help="EWM span for target smoothing to control churn")
    p.add_argument("--rebal-threshold", type=float, default=0.20,
                   help="Min target change to trigger rebalance")
    args = p.parse_args()
    if not (os.path.exists(args.model) and os.path.exists(args.data)):
        print(f"Missing files\n  model: {args.model}\n  data: {args.data}")
        sys.exit(1)
    run(args.model, args.data, kill_turb_pct=args.kill_turb_pct,
        max_target=args.max_target, target_smooth_span=args.smooth_span,
        rebal_threshold=args.rebal_threshold)
