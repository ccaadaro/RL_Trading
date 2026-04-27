#!/usr/bin/env python3
"""
scripts/train_signal_walkforward.py — Walk-forward LGBM signal training
════════════════════════════════════════════════════════════════════════

Why
───
The existing `retrain_signal_v2.py` fits a single "final" LightGBM on
2019-01 → 2024-06 and uses it to predict VAL+TEST. That's a single
train/test split — fragile, gives point estimates that the walk-forward
eval just showed are unreliable.

This script does the correct thing: refits LightGBM N times along the
timeline, each fold using only past data to predict a disjoint future
window. The final `signal_prob_v2_wf_feature` column is constructed
from fold-OOS predictions only — every probability is an honest OOS
forecast at the moment it could have been made live.

Fold layout (anchored walk-forward)
───────────────────────────────────
    anchor_start ────────────────────────→  (fixed, e.g. 2019-01-01)

    Fold 0:  train [anchor..val_start-1]  val 3mo  test 3mo
    Fold 1:  train [anchor..val_start-1]+step  val 3mo  test 3mo
    ...
    Fold N:  …                                                  (rolls up to end)

Each fold predicts its own disjoint TEST window, then all TEST preds
are stitched into a single column covering [first_test_start ... end].

For bars BEFORE first_test_start we fall back to the existing global
`signal_prob_v2_feature` so downstream code works end-to-end; those
bars are clearly marked.

Output
──────
- `signal_prob_v2_wf_feature` column in the parquet (honest OOS preds
  on rolling test windows, fallback to global v2 for early bars).
- `models/signal_lgbm_v2_wf/` directory containing one pickle per
  fold + a manifest json with fold bounds and per-fold metrics.

Usage
─────
    python scripts/train_signal_walkforward.py
    python scripts/train_signal_walkforward.py \
        --anchor-start 2019-01-01 \
        --first-test-start 2021-07-01 \
        --val-days 90 --test-days 90 --step-days 90
"""

import argparse
import json
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from utils.signal_features import (
    SIGNAL_FEAT_COLS_V2,
    build_feature_matrix,
    load_funding_series,
)
from utils.walkforward import anchored_walkforward, summarise_folds


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH    = "cache/data_v1_430340a861af5f8f9bcbd7a4ca16ba95.parquet"
ETH_PATH     = "../../data/binance/ETH_USDT-1h.feather"
FUNDING_PATH = "../../data/binance/futures/BTC_USDT_USDT-8h-funding_rate.feather"
OUT_DIR      = Path("models/signal_lgbm_v2_wf")
OUT_COL      = "signal_prob_v2_wf_feature"

HORIZON = 1   # predict sign of next bar

LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "boosting_type":     "gbdt",
    "n_estimators":      600,
    "learning_rate":     0.03,
    "max_depth":         5,
    "num_leaves":        31,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.6,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "random_state":      42,
    "verbose":           -1,
    "n_jobs":            -1,
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_prepared_df() -> pd.DataFrame:
    print(f"  Loading parquet: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    eth_df = pd.DataFrame()
    eth_p  = Path(ETH_PATH)
    if eth_p.exists():
        eth_df = pd.read_feather(str(eth_p))
        if "date" in eth_df.columns:
            eth_df = eth_df.set_index("date")
        eth_df.index = pd.to_datetime(eth_df.index, utc=True).tz_localize(None)

    funding = load_funding_series(FUNDING_PATH)
    if funding is not None:
        funding.index = funding.index.tz_localize(None)

    print(f"  Computing {len(SIGNAL_FEAT_COLS_V2)} features …")
    X_full = build_feature_matrix(df, eth_df=eth_df, funding_series=funding)
    df[SIGNAL_FEAT_COLS_V2] = X_full[SIGNAL_FEAT_COLS_V2].values

    future_ret = np.log(df["close"].shift(-HORIZON) / df["close"])
    df["target"] = (future_ret > 0).astype(int)
    return df.iloc[:-HORIZON].copy()


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fit_fold(train_df, val_df, feat_cols):
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        train_df[feat_cols], train_df["target"],
        eval_set=[(val_df[feat_cols], val_df["target"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def predict_slice(model, slice_df, feat_cols):
    return model.predict_proba(slice_df[feat_cols])[:, 1]


def eval_fold(model, slice_df, feat_cols, label):
    probs = predict_slice(model, slice_df, feat_cols)
    y     = slice_df["target"].values
    if len(set(y)) < 2:
        return probs, {"auc": 0.5, "acc": 0.5, "n": len(slice_df)}
    acc = accuracy_score(y, (probs > 0.5).astype(int))
    auc = roc_auc_score(y, probs)
    return probs, {"auc": auc, "acc": acc, "n": len(slice_df)}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-start",      type=str, default="2019-01-01")
    ap.add_argument("--first-test-start",  type=str, default="2021-07-01",
                    help="First fold's TEST start (must allow >=2y of TRAIN + val_days).")
    ap.add_argument("--val-days",          type=int, default=90)
    ap.add_argument("--test-days",         type=int, default=90,
                    help="Disjoint test windows stitched into the OOS series.")
    ap.add_argument("--step-days",         type=int, default=90,
                    help="Advance per fold. Equals test-days → non-overlapping tests.")
    ap.add_argument("--min-train-bars",    type=int, default=10_000,
                    help="Skip folds whose train slice is smaller than this.")
    ap.add_argument("--parquet",           type=str, default=DATA_PATH)
    ap.add_argument("--global-fallback-col", type=str,
                    default="signal_prob_v2_feature",
                    help="Column to use for bars BEFORE first_test_start "
                         "(no OOS prediction available there).")
    args = ap.parse_args()

    print("=" * 65)
    print("  Walk-forward LightGBM signal training")
    print(f"  Anchor  : {args.anchor_start}")
    print(f"  Folds   : from test_start={args.first_test_start}, "
          f"val={args.val_days}d  test={args.test_days}d  step={args.step_days}d")
    print("=" * 65)

    df = load_prepared_df()
    feat_cols = SIGNAL_FEAT_COLS_V2
    print(f"  Bars: {len(df)}  features: {len(feat_cols)}  "
          f"range: {df.index[0].date()} → {df.index[-1].date()}")

    # ── Storage ───────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wf_probs = pd.Series(np.nan, index=df.index, name=OUT_COL)
    fold_records = []

    # ── Iterate folds ─────────────────────────────────────────────────────
    print(f"\n  {'#':>3}  {'TrainEnd':<10}  {'ValEnd':<10}  {'TestEnd':<10}  "
          f"{'Train':>6}  {'Test':>4}  {'AUC':>5}  {'Acc':>5}  {'Regime':<8}")
    print(f"  {'─'*85}")

    wf_gen = anchored_walkforward(
        df, args.anchor_start, args.first_test_start,
        val_days=args.val_days, test_days=args.test_days,
        step_days=args.step_days, min_train_bars=args.min_train_bars,
    )

    for train_df, val_df, test_df, meta in wf_gen:
        # Trim last HORIZON row from train so its target was fully observed
        # when the fold would have been fit in live trading.
        tr = train_df.iloc[:-HORIZON] if len(train_df) > HORIZON else train_df
        vl = val_df.iloc[:-HORIZON] if len(val_df) > HORIZON else val_df

        model = fit_fold(tr, vl, feat_cols)
        probs, m_val  = eval_fold(model, vl,      feat_cols, "VAL")
        probs_test, m = eval_fold(model, test_df, feat_cols, "TEST")

        wf_probs.loc[test_df.index] = probs_test

        # Persist fold artefact + meta
        fold_pkl = OUT_DIR / f"fold_{meta.fold_idx:02d}.pkl"
        with open(fold_pkl, "wb") as f:
            pickle.dump({
                "model":        model,
                "feat_cols":    feat_cols,
                "meta":         meta.to_dict(),
                "val_metrics":  m_val,
                "test_metrics": m,
                "best_iteration": model.best_iteration_,
            }, f)

        print(f"  {meta.fold_idx:>3}  "
              f"{meta.train_end.strftime('%Y-%m-%d'):<10}  "
              f"{meta.val_end.strftime('%Y-%m-%d'):<10}  "
              f"{meta.test_end.strftime('%Y-%m-%d'):<10}  "
              f"{len(tr):>6}  {m['n']:>4}  "
              f"{m['auc']:>5.3f}  {m['acc']:>5.3f}  "
              f"{meta.regime_tag:<8}")

        fold_records.append({
            "fold_idx":     meta.fold_idx,
            "train_end":    str(meta.train_end.date()),
            "val_end":      str(meta.val_end.date()),
            "test_start":   str(meta.test_start.date()),
            "test_end":     str(meta.test_end.date()),
            "regime_tag":   meta.regime_tag,
            "market_return": meta.market_return,
            "auc":          m["auc"],
            "acc":          m["acc"],
            "n_test":       m["n"],
            "trees":        int(model.best_iteration_ or 0),
        })

    if not fold_records:
        print("  [ERROR] No folds generated — check your dates / min-train-bars.")
        return

    # ── Summary ───────────────────────────────────────────────────────────
    summary = summarise_folds([
        # synthesize an alpha-ish record so summarise_folds works; use AUC-0.5 as edge
        {"regime_tag": r["regime_tag"], "alpha": (r["auc"] - 0.5) * 100,
         "portfolio_return": 0, "market_return": r["market_return"],
         "sharpe": 0, "max_dd": 0}
        for r in fold_records
    ])
    aucs = [r["auc"] for r in fold_records]
    accs = [r["acc"] for r in fold_records]
    print(f"\n{'═'*65}")
    print(f"  Per-fold AUC: mean={np.mean(aucs):.4f}  std={np.std(aucs):.4f}  "
          f"min={min(aucs):.4f}  max={max(aucs):.4f}")
    print(f"  Per-fold Acc: mean={np.mean(accs):.4f}  std={np.std(accs):.4f}")
    print(f"  Folds with AUC>0.5: "
          f"{sum(1 for a in aucs if a > 0.5)}/{len(aucs)} "
          f"({100*sum(1 for a in aucs if a > 0.5)/len(aucs):.0f}%)")
    for tag in ("bull", "bear", "sideways"):
        tag_aucs = [r["auc"] for r in fold_records if r["regime_tag"] == tag]
        if tag_aucs:
            print(f"    {tag:<9}  AUĈ = {np.mean(tag_aucs):.4f}  (n={len(tag_aucs)})")

    # ── Fill pre-first-fold bars with fallback column ─────────────────────
    fallback_col = args.global_fallback_col
    if fallback_col not in df.columns:
        print(f"\n  [WARN] Fallback column {fallback_col} missing — early bars stay NaN.")
    else:
        mask_nan = wf_probs.isna()
        wf_probs.loc[mask_nan] = df.loc[mask_nan, fallback_col].values
        print(f"\n  Filled {int(mask_nan.sum())} pre-first-fold bars from {fallback_col}.")

    # Fill any remaining NaN with 0.5
    wf_probs.fillna(0.5, inplace=True)
    print(f"  WF prob stats: mean={wf_probs.mean():.4f}  "
          f"std={wf_probs.std():.4f}  min={wf_probs.min():.4f}  "
          f"max={wf_probs.max():.4f}")

    # ── Persist to parquet ────────────────────────────────────────────────
    df_out = pd.read_parquet(args.parquet)
    n = min(len(df_out), len(wf_probs))
    df_out = df_out.iloc[:n].copy()
    df_out[OUT_COL] = wf_probs.values[:n]
    df_out.to_parquet(args.parquet)
    print(f"\n  Parquet updated → {OUT_COL}")

    # ── Save manifest ─────────────────────────────────────────────────────
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps({
        "anchor_start":     args.anchor_start,
        "first_test_start": args.first_test_start,
        "val_days":         args.val_days,
        "test_days":        args.test_days,
        "step_days":        args.step_days,
        "n_folds":          len(fold_records),
        "folds":            fold_records,
        "summary":          summary,
        "lgb_params":       LGB_PARAMS,
        "feat_cols":        feat_cols,
        "output_column":    OUT_COL,
    }, indent=2, default=str))
    print(f"  Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
