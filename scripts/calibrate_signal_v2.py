#!/usr/bin/env python3
"""
scripts/calibrate_signal_v2.py — Stretch LightGBM probabilities so the RL
agent sees decisive signals.

Problem
───────
The raw `signal_prob_v2_feature` column has std ≈ 0.050 around 0.5 —
even "strong" LGBM predictions land at 0.55 / 0.45. The RL policy
receives nearly flat observations and never learns to separate its
action between "mild signal" and "strong signal".

Fix
───
Apply a logit-stretch only to the v2 column on the training portion,
fit a scale, and write a new `signal_score_v2_feature` column so the
original probability is preserved. This is a monotonic RL score, NOT a
probability calibration. The transformation is strictly monotonic,
order-preserving, and zero-leakage (parameters come from TRAIN only).

    logit(p)        = log(p / (1 - p))
    p_cal           = sigmoid(scale * logit(p) + bias)

We solve for `scale` so the resulting train-set std hits a target
(default 0.15 — 3× the current v2 std, still well below the
theoretical max of ~0.29). `bias` keeps mean(p_cal) = mean(p_train)
so a 0.50 base-rate signal stays a 0.50 signal.

Output
──────
Adds `signal_score_v2_feature` to the parquet in-place. Prints
stats on TRAIN / VAL / TEST splits for before/after comparison.

Usage
─────
    python scripts/calibrate_signal_v2.py
    python scripts/calibrate_signal_v2.py --target-std 0.12
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import brentq

DATA_PATH  = "cache/data_v1_430340a861af5f8f9bcbd7a4ca16ba95.parquet"
SRC_COL    = "signal_prob_v2_feature"
DST_COL    = "signal_score_v2_feature"
LEGACY_ALIAS_COL = "signal_prob_v2_cal_feature"

TRAIN_END  = "2024-06-30"
VAL_START  = "2024-07-01"
VAL_END    = "2025-06-30"
TEST_START = "2025-07-01"

EPS        = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _fit_scale(logit_train: np.ndarray, mean_train: float,
               target_std: float) -> float:
    """Find the positive `scale` s.t. std(sigmoid(scale*z + bias)) = target_std,
    with bias chosen so mean(p_cal) ≈ mean_train."""
    target_mean_logit = np.log(mean_train / (1 - mean_train))

    def _std_given_scale(scale: float) -> float:
        z_shifted = scale * logit_train
        # Solve bias s.t. sigmoid(z_shifted + bias) has mean = mean_train.
        # Cheap 1-D root-find on bias.
        def _mean_err(b: float) -> float:
            return _sigmoid(z_shifted + b).mean() - mean_train
        # Bracket: logit_train roughly zero-centred, start wide.
        try:
            bias = brentq(_mean_err, -20.0, 20.0, xtol=1e-6)
        except ValueError:
            bias = target_mean_logit
        return _sigmoid(z_shifted + bias).std(), bias

    def _obj(scale: float) -> float:
        s, _ = _std_given_scale(scale)
        return s - target_std

    try:
        best_scale = brentq(_obj, 0.1, 20.0, xtol=1e-5)
    except ValueError as e:
        # If the bracket doesn't contain the root, the max achievable std
        # with any scale < 20 is lower than target — fall back to scale=20.
        s_hi, _ = _std_given_scale(20.0)
        s_lo, _ = _std_given_scale(0.1)
        raise RuntimeError(
            f"Target std {target_std} not achievable "
            f"(scale=0.1 → std={s_lo:.4f}, scale=20 → std={s_hi:.4f}). "
            "Pick a lower --target-std."
        ) from e

    _, bias = _std_given_scale(best_scale)
    return best_scale, bias


def _summary(label: str, p: np.ndarray) -> None:
    print(f"    {label:<8s}  "
          f"mean={p.mean():.4f}  std={p.std():.4f}  "
          f"min={p.min():.4f}  max={p.max():.4f}  "
          f"P(>0.60)={ (p>0.60).mean():.1%}  "
          f"P(<0.40)={ (p<0.40).mean():.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-std", type=float, default=0.15,
                    help="Target std for p_cal on TRAIN (default 0.15).")
    ap.add_argument("--parquet", type=str, default=DATA_PATH)
    ap.add_argument("--source-col", type=str, default=SRC_COL,
                    help="Column to calibrate (default signal_prob_v2_feature).")
    ap.add_argument("--target-col", type=str, default=DST_COL,
                    help="Column to write (default signal_score_v2_feature).")
    ap.add_argument("--legacy-alias-col", type=str, default=LEGACY_ALIAS_COL,
                    help="Optional compatibility alias. When set, writes the same RL score "
                         "to this legacy column name too.")
    ap.add_argument("--params-out", type=str, default="models/signal_v2_calibration.json",
                    help="Path to save fitted calibration parameters.")
    args = ap.parse_args()

    src_col = args.source_col
    dst_col = args.target_col

    print("=" * 65)
    print(f"  LightGBM signal calibration (logit-stretch)")
    print(f"  Source: {src_col}   Target: {dst_col}")
    print(f"  Target TRAIN std: {args.target_std}")
    print("=" * 65)

    df = pd.read_parquet(args.parquet)
    if src_col not in df.columns:
        raise RuntimeError(f"{src_col} missing — run the upstream training script first.")

    p = df[src_col].values.astype(np.float64)
    mask_train = df.index <= TRAIN_END
    p_train    = p[mask_train]

    z_train = _logit(p_train)
    mean_train = float(np.mean(p_train))
    print(f"\n  BEFORE calibration:")
    _summary("TRAIN", p_train)
    _summary("VAL",   p[(df.index >= VAL_START) & (df.index <= VAL_END)])
    _summary("TEST",  p[df.index >= TEST_START])

    scale, bias = _fit_scale(z_train, mean_train, args.target_std)
    print(f"\n  Fitted: scale = {scale:.4f}  bias = {bias:+.4f}  "
          f"(train mean preserved at {mean_train:.4f})")

    # Apply to ALL rows (TRAIN+VAL+TEST) using the same transformation.
    p_cal = _sigmoid(scale * _logit(p) + bias)

    print(f"\n  AFTER calibration:")
    _summary("TRAIN", p_cal[mask_train])
    _summary("VAL",   p_cal[(df.index >= VAL_START) & (df.index <= VAL_END)])
    _summary("TEST",  p_cal[df.index >= TEST_START])

    # Rank correlation sanity — must be ~1.0 (monotonic transform).
    from scipy.stats import spearmanr
    r, _ = spearmanr(p, p_cal)
    print(f"\n  Spearman rank corr (raw vs cal): {r:.6f}  "
          f"(must be 1.0 — monotonic transform)")

    df[dst_col] = p_cal
    if args.legacy_alias_col:
        df[args.legacy_alias_col] = p_cal
    df.to_parquet(args.parquet)
    print(f"\n  Wrote → {args.parquet}  (column: {dst_col})")
    if args.legacy_alias_col:
        print(f"  Compatibility alias → {args.legacy_alias_col} "
              f"(semantics = score, not calibrated probability)")

    # Persist the calibration params for reproducibility.
    params_path = Path(args.params_out)
    params_path.parent.mkdir(exist_ok=True, parents=True)
    import json
    params_path.write_text(json.dumps({
        "source_column": src_col,
        "output_column": dst_col,
        "semantics":     "rl_score_not_probability",
        "scale":         scale,
        "bias":          bias,
        "target_std":    args.target_std,
        "train_mean":    mean_train,
        "train_end":     TRAIN_END,
    }, indent=2))
    print(f"  Calibration params → {params_path}")


if __name__ == "__main__":
    main()
