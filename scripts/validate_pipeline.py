#!/usr/bin/env python3
"""
scripts/validate_pipeline.py

Mandatory statistical validation gate before any live capital deployment.
Checks:
  1. Probability Calibration (Brier Score + Reliability Diagram)
  2. Deflated Sharpe Ratio (DSR) — penalizes Sharpe for number of strategy trials
  3. Triple Barrier label uniqueness audit (concurrent label overlap)

Usage:
    python scripts/validate_pipeline.py --data cache/dollar_bars_btc_2000000_regimes.feather
    python scripts/validate_pipeline.py --data cache/dollar_bars_btc_2000000_regimes.feather \
        --effective-trials-method correlation --trial-correlation 0.85
"""

import sys
import argparse
import heapq
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy.stats import norm
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))


# ─── 1. Probability Calibration ───────────────────────────────────────────────

def check_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """
    Reliability diagram + Brier Score.
    A well-calibrated model has mean predicted prob ≈ fraction of positives in each bin.
    """
    brier = brier_score_loss(y_true, y_prob)

    calib_df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).dropna()
    if calib_df.empty:
        raise ValueError("No valid probability rows available for calibration.")

    try:
        bins = pd.qcut(calib_df["y_prob"], q=n_bins, duplicates="drop")
    except ValueError:
        bins = pd.cut(calib_df["y_prob"], bins=min(n_bins, max(2, calib_df["y_prob"].nunique())))

    grouped = calib_df.groupby(bins, observed=False)
    mean_pred_val = grouped["y_prob"].mean().to_numpy()
    fraction_of_pos = grouped["y_true"].mean().to_numpy()
    bin_sizes = grouped.size().to_numpy()

    # Expected Calibration Error (ECE): weighted mean absolute deviation
    ece = np.sum(np.abs(fraction_of_pos - mean_pred_val) * bin_sizes) / bin_sizes.sum()

    print("\n[1] PROBABILITY CALIBRATION")
    print(f"    Brier Score : {brier:.5f}  (lower = better; ~0.25 = random, ~0.0 = perfect)")
    print(f"    ECE         : {ece:.5f}   (lower = better; 0 = perfect calibration)")
    print("\n    Reliability Diagram:")
    for fop, mpv in zip(fraction_of_pos, mean_pred_val):
        bar = "█" * int(fop * 30)
        print(f"    prob_bin={mpv:.2f}  actual_win_rate={fop:.3f}  {bar}")

    verdict = "PASS" if ece < 0.05 else ("WARN" if ece < 0.10 else "FAIL — Calibration Required")
    print(f"    → Verdict: {verdict}")
    return {"brier": brier, "ece": ece, "calibration_verdict": verdict}


# ─── 2. Deflated Sharpe Ratio ─────────────────────────────────────────────────

def deflated_sharpe_ratio(sharpe_observed: float, n_trials: int, n_obs: int,
                          skew: float = 0.0, kurt: float = 3.0) -> dict:
    """
    Bailey & Lopez de Prado (2014) Probabilistic Sharpe Ratio and Deflated Sharpe Ratio.
    Accounts for multiple strategy evaluations (selection bias / backtest overfitting).

    DSR converts the observed Sharpe into a probability that the true SR > 0,
    penalized by the number of independent trials tested.
    """
    # Expected maximum Sharpe under N_trials iid Gaussian strategies
    # E[max SR] ≈ (1 - γ) * Φ^{-1}(1 - 1/N) + γ * Φ^{-1}(1 - 1/(N*e))
    # Simplified: use the expected maximum of N standard normals
    E_max_sr = (1 - 0.5772) * norm.ppf(1 - 1 / n_trials) + 0.5772 * norm.ppf(1 - 1 / (n_trials * np.e))

    # Probabilistic Sharpe Ratio (PSR)
    # PSR(SR*) = Φ( (SR_hat - SR*) * sqrt(n-1) / sqrt(1 - skew*SR + ((kurt-1)/4)*SR^2) )
    sr_star = E_max_sr
    sigma_sr = np.sqrt((1 - skew * sharpe_observed + ((kurt - 1) / 4) * sharpe_observed ** 2) / (n_obs - 1))
    psr = norm.cdf((sharpe_observed - sr_star) / sigma_sr)

    print("\n[2] DEFLATED SHARPE RATIO")
    print(f"    Observed SR       : {sharpe_observed:.4f}")
    print(f"    N trials tested   : {n_trials}")
    print(f"    Expected max SR   : {E_max_sr:.4f}  (hurdle to beat)")
    print(f"    PSR (P[SR>0])     : {psr:.4f}  (higher = better; >0.95 is very strong)")

    verdict = "PASS" if psr > 0.95 else ("WARN" if psr > 0.80 else "FAIL — Likely Overfitted")
    print(f"    → Verdict: {verdict}")
    return {"psr": psr, "expected_max_sr": E_max_sr, "dsr_verdict": verdict}


def derive_event_return_proxy(df: pd.DataFrame, mode: str = "barrier") -> np.ndarray:
    """
    Build a conservative tradable-return proxy from the triple-barrier geometry.
    This is materially better than using ±1 labels as if they were PnL.
    """
    if mode == "label":
        return np.where(df["label"].to_numpy() == 1, 1.0, np.where(df["label"].to_numpy() == -1, -1.0, 0.0))

    if {"start_price", "upper_b", "lower_b", "label"}.issubset(df.columns):
        up_ret = df["upper_b"].to_numpy() / df["start_price"].to_numpy() - 1.0
        dn_ret = df["lower_b"].to_numpy() / df["start_price"].to_numpy() - 1.0
        return np.where(
            df["label"].to_numpy() == 1,
            up_ret,
            np.where(df["label"].to_numpy() == -1, dn_ret, 0.0),
        )

    if {"start_price", "end_price"}.issubset(df.columns):
        return df["end_price"].to_numpy() / df["start_price"].to_numpy() - 1.0

    print("WARN: event return geometry missing; falling back to signed labels as a weak PnL proxy.")
    return np.where(df["label"].to_numpy() == 1, 1.0, np.where(df["label"].to_numpy() == -1, -1.0, 0.0))


def infer_events_per_year(df: pd.DataFrame) -> float:
    """
    Infer annualisation factor from the actual event timestamps.
    """
    if "start_time" not in df.columns or len(df) < 2:
        return 252.0
    ts = pd.to_datetime(df["start_time"]).sort_values()
    span_days = max((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0, 1.0)
    return len(ts) / span_days * 365.25


def derive_effective_trials(
    n_trials: int,
    method: str = "raw",
    manual_value: int | None = None,
    trial_correlation: float = 0.0,
) -> int:
    """
    Convert a raw grid size into an estimated count of independent trials.

    Methods
    -------
    raw         : use n_trials exactly.
    manual      : use --effective-trials directly.
    sqrt        : heuristic shrinkage for highly overlapping sweeps.
    correlation : N_eff ≈ 1 + (N-1) * (1-rho), clipped to [1, N].
    """
    n_trials = max(int(n_trials), 1)

    if method == "raw":
        return n_trials
    if method == "manual":
        if manual_value is None:
            raise ValueError("--effective-trials is required when method=manual")
        return max(1, min(int(manual_value), n_trials))
    if method == "sqrt":
        return max(1, min(int(np.ceil(np.sqrt(n_trials))), n_trials))
    if method == "correlation":
        rho = float(np.clip(trial_correlation, 0.0, 0.999))
        eff = int(np.ceil(1 + (n_trials - 1) * (1 - rho)))
        return max(1, min(eff, n_trials))

    raise ValueError(f"Unknown effective-trials method: {method}")


# ─── 3. Triple Barrier Label Uniqueness Audit ────────────────────────────────

def audit_label_uniqueness(df: pd.DataFrame) -> dict:
    """
    Checks the fraction of non-overlapping samples.
    Triple Barrier labels on adjacent events often overlap in time,
    meaning they are not IID. Uniqueness weight = 1/concurrency.

    We approximate: if two consecutive labeled events both fired within
    the same rolling window, they share the same outcome period.
    """
    if not {"start_time", "end_time"}.issubset(df.columns):
        print("\n[3] LABEL UNIQUENESS: Skipped (need start_time/end_time)")
        return {}

    print("\n[3] TRIPLE BARRIER LABEL UNIQUENESS AUDIT")

    starts = pd.to_datetime(df["start_time"]).astype("int64").to_numpy()
    ends   = pd.to_datetime(df["end_time"]).astype("int64").to_numpy()
    active_ends: list[int] = []
    concurrency = np.ones(len(df), dtype=float)

    for i, (start_ns, end_ns) in enumerate(zip(starts, ends)):
        while active_ends and active_ends[0] < start_ns:
            heapq.heappop(active_ends)
        concurrency[i] = len(active_ends) + 1
        heapq.heappush(active_ends, end_ns)

    uniqueness = 1.0 / concurrency
    avg_uniqueness = uniqueness.mean()

    # Label distribution by regime
    if "hmm_semantic_regime" in df.columns:
        print("\n    Label Balance by Regime:")
        if "label" in df.columns:
            regime_balance = df.groupby("hmm_semantic_regime")["label"].mean()
            for regime, mean_label in regime_balance.items():
                bar = "█" * int(abs(mean_label) * 30)
                direction = "BULL BIAS" if mean_label > 0.1 else ("BEAR BIAS" if mean_label < -0.1 else "BALANCED")
                print(f"    {regime:20s}: mean_label={mean_label:+.3f}  {direction} {bar}")

    print(f"\n    Average Label Uniqueness: {avg_uniqueness:.4f}")
    print(f"    (1.0 = no overlaps, <0.5 = severe concurrency, needs uniqueness weighting)")

    verdict = "PASS" if avg_uniqueness > 0.5 else "WARN — Consider uniqueness sample weights in retraining"
    print(f"    → Verdict: {verdict}")
    return {"avg_uniqueness": avg_uniqueness, "uniqueness_verdict": verdict}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Pre-production statistical validation gate")
    ap.add_argument("--data",     default="cache/dollar_bars_btc_2000000_regimes.feather")
    ap.add_argument("--n-trials", type=int,   default=50,   help="Number of strategy combinations tested")
    ap.add_argument("--effective-trials", type=int, default=None,
                    help="Estimated number of independent trials. Used directly when --effective-trials-method=manual.")
    ap.add_argument("--effective-trials-method",
                    choices=["raw", "manual", "sqrt", "correlation"],
                    default="raw",
                    help="How to map raw trial count to effective independent trials for DSR.")
    ap.add_argument("--trial-correlation", type=float, default=0.0,
                    help="Assumed average correlation between tested variants when using --effective-trials-method=correlation.")
    ap.add_argument("--dsr-return-mode", choices=["barrier", "label"], default="barrier",
                    help="Return proxy used for the Sharpe/DSR calculation. 'barrier' uses triple-barrier geometry; 'label' uses signed labels only.")
    ap.add_argument("--events-per-year", type=float, default=None,
                    help="Override the inferred event frequency used for Sharpe annualisation.")
    ap.add_argument("--n-obs",    type=int,   default=None, help="Number of OOS observations (auto-detected)")
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print("  PRE-PRODUCTION VALIDATION GATE")
    print(f"{'='*60}")

    print(f"\nLoading {args.data}...")
    df = feather.read_feather(args.data)

    if "oof_pred" not in df.columns or "label" not in df.columns:
        print("ABORT: Need 'oof_pred' and 'label' columns. Run build_regime_labels.py first.")
        return

    # Use only rows explicitly marked as valid OOF predictions.
    if "oof_valid" in df.columns:
        valid_mask = df["oof_valid"].astype(bool) & df["oof_pred"].notna()
    else:
        valid_mask = df["oof_pred"].notna()
        print("WARN: oof_valid column missing. Falling back to non-null oof_pred rows only.")
    df_val = df[valid_mask].copy()
    if df_val.empty:
        print("ABORT: No valid OOF rows found. Run train_dollar_alpha.py so OOFs are persisted first.")
        return
    df_val["binary_target"] = (df_val["label"] == 1).astype(int)

    n_obs = args.n_obs or len(df_val)
    effective_trials = derive_effective_trials(
        n_trials=args.n_trials,
        method=args.effective_trials_method,
        manual_value=args.effective_trials,
        trial_correlation=args.trial_correlation,
    )

    results = {}

    # --- 1. Calibration ---
    results.update(check_calibration(df_val["binary_target"].values, df_val["oof_pred"].values))

    # --- 2. Compute proxy Sharpe from OOS predictions using event-return geometry ---
    # Use confidence-scaled direction times the event's implied barrier return.
    signals = np.clip(2 * df_val["oof_pred"].to_numpy() - 1.0, -1.0, 1.0)
    event_returns = derive_event_return_proxy(df_val, mode=args.dsr_return_mode)
    pnl_proxy = pd.Series(signals * event_returns)
    events_per_year = args.events_per_year or infer_events_per_year(df_val)
    sr_hat = pnl_proxy.mean() / (pnl_proxy.std() + 1e-8) * np.sqrt(events_per_year)
    skew   = float(pnl_proxy.skew())
    kurt   = float(pnl_proxy.kurt() + 3)  # convert excess kurtosis to full kurtosis
    print(f"\n    Events/year proxy : {events_per_year:.1f}")
    print(f"    DSR return mode   : {args.dsr_return_mode}")
    print(f"    Trials raw/effective: {args.n_trials}/{effective_trials} "
          f"(method={args.effective_trials_method})")
    results.update(deflated_sharpe_ratio(sr_hat, n_trials=effective_trials, n_obs=n_obs, skew=skew, kurt=kurt))

    # --- 3. Uniqueness ---
    results.update(audit_label_uniqueness(df_val))

    # Final gate
    print(f"\n{'='*60}")
    print("  FINAL GATE SUMMARY")
    print(f"{'='*60}")
    verdicts = {k: v for k, v in results.items() if "verdict" in k}
    for k, v in verdicts.items():
        status = "✓" if "FAIL" not in str(v) else "✗"
        print(f"  {status} {k:35s}: {v}")

    any_fail = any("FAIL" in str(v) for v in verdicts.values())
    print(f"\n  {'🔴 HOLD — Do not deploy capital.' if any_fail else '🟢 PROCEED — Cleared for shadow-live dry-run.'}")


if __name__ == "__main__":
    main()
