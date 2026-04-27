#!/usr/bin/env python3
"""
scripts/train_signal_walkforward_15m.py — Walk-forward LightGBM signal
training on the 15m dataset (OHLCV + order-flow + futures).

Why a separate script
─────────────────────
`train_signal_walkforward.py` is hard-wired to `SIGNAL_FEAT_COLS_V2`
(58 features from the 4h pipeline). The 15m dataset has a different
feature set (52 features including the new `*_trade_feature` columns).
This script auto-detects features from the parquet to keep the 4h
pipeline untouched.

Everything else mirrors the 4h walk-forward (anchored expanding train,
disjoint 90-day test windows, per-fold refit, stitched OOS output).

Usage
─────
    python scripts/train_signal_walkforward_15m.py
    python scripts/train_signal_walkforward_15m.py \
        --first-test-start 2024-01-01 --val-days 60 --test-days 60
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

from utils.walkforward import anchored_walkforward, summarise_folds


DATA_PATH = "cache/dataset_btc_15m_v1.parquet"
OUT_DIR   = Path("models/signal_lgbm_15m_wf")
OUT_COL   = "signal_prob_15m_wf_feature"   # written back into dataset

HORIZON = 8   # predict sign of next 15m bar (8 bars = 2h holding period)

LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "boosting_type":     "gbdt",
    "n_estimators":      800,
    "learning_rate":     0.02,   # slower LR, more data to fit
    "max_depth":         6,
    "num_leaves":        63,
    "min_child_samples": 200,    # higher than 4h (more rows → more noise)
    "subsample":         0.8,
    "colsample_bytree":  0.6,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "random_state":      42,
    "verbose":           -1,
    "n_jobs":            -1,
}


def load_prepared_df(path: str) -> tuple[pd.DataFrame, list]:
    df = pd.read_parquet(path)
    feat_cols = sorted([c for c in df.columns if "feature" in c])
    print(f"  Data   : {len(df):,} rows  {df.index[0]} → {df.index[-1]}")
    print(f"  Features: {len(feat_cols)}")

    future_ret = np.log(df["close"].shift(-HORIZON) / df["close"])
    df["target"] = (future_ret > 0).astype(int)
    return df, feat_cols


def fit_fold(train_df, val_df, feat_cols):
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(
        train_df[feat_cols], train_df["target"],
        eval_set=[(val_df[feat_cols], val_df["target"])],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    return m


def eval_slice(model, slice_df, feat_cols):
    probs = model.predict_proba(slice_df[feat_cols])[:, 1]
    y     = slice_df["target"].values
    if len(set(y)) < 2:
        return probs, {"auc": 0.5, "acc": 0.5, "n": len(slice_df)}
    acc = accuracy_score(y, (probs > 0.5).astype(int))
    auc = roc_auc_score(y, probs)
    return probs, {"auc": auc, "acc": acc, "n": len(slice_df)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet",           type=str, default=DATA_PATH)
    ap.add_argument("--anchor-start",      type=str, default="2023-06-15",
                    help="Earliest train date. 2023-06-15 leaves ~15 days "
                         "of warmup for long rolling features.")
    ap.add_argument("--first-test-start",  type=str, default="2024-01-01",
                    help="Start of fold-0's test window. Needs ~6 months "
                         "of training before.")
    ap.add_argument("--val-days",          type=int, default=60)
    ap.add_argument("--test-days",         type=int, default=60)
    ap.add_argument("--step-days",         type=int, default=60)
    ap.add_argument("--min-train-bars",    type=int, default=10_000,
                    help="15m bars (1 day ≈ 96). 10k ≈ 104 days.")
    args = ap.parse_args()

    print("=" * 65)
    print("  Walk-forward LightGBM (15m dataset, order-flow features)")
    print(f"  Anchor  : {args.anchor_start}")
    print(f"  Folds   : test_start={args.first_test_start}  "
          f"val={args.val_days}d  test={args.test_days}d  step={args.step_days}d")
    print("=" * 65)

    df, feat_cols = load_prepared_df(args.parquet)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wf_probs = pd.Series(np.nan, index=df.index, name=OUT_COL)
    fold_records = []

    print(f"\n  {'#':>3}  {'TrainEnd':<10}  {'ValEnd':<10}  {'TestEnd':<10}  "
          f"{'Train':>6}  {'Test':>4}  {'AUC':>5}  {'Acc':>5}  {'Regime':<8}")
    print(f"  {'─'*85}")

    wf_gen = anchored_walkforward(
        df, args.anchor_start, args.first_test_start,
        val_days=args.val_days, test_days=args.test_days,
        step_days=args.step_days, min_train_bars=args.min_train_bars,
    )
    for train_df, val_df, test_df, meta in wf_gen:
        tr = train_df.iloc[:-HORIZON] if len(train_df) > HORIZON else train_df
        vl = val_df.iloc[:-HORIZON] if len(val_df) > HORIZON else val_df

        model = fit_fold(tr, vl, feat_cols)
        _, m_val  = eval_slice(model, vl,      feat_cols)
        probs_test, m = eval_slice(model, test_df, feat_cols)

        wf_probs.loc[test_df.index] = probs_test

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
            "fold_idx":      meta.fold_idx,
            "train_end":     str(meta.train_end.date()),
            "val_end":       str(meta.val_end.date()),
            "test_start":    str(meta.test_start.date()),
            "test_end":      str(meta.test_end.date()),
            "regime_tag":    meta.regime_tag,
            "market_return": meta.market_return,
            "auc":           m["auc"],
            "acc":           m["acc"],
            "n_test":        m["n"],
            "trees":         int(model.best_iteration_ or 0),
        })

    if not fold_records:
        print("  [ERROR] No folds generated.")
        return

    # Summary
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
            print(f"    {tag:<9}  AUĈ = {np.mean(tag_aucs):.4f}  (n={len(tag_aucs)})")

    # Fill pre-first-fold bars with 0.5 (no prediction available)
    wf_probs.fillna(0.5, inplace=True)
    print(f"\n  WF prob stats: mean={wf_probs.mean():.4f}  "
          f"std={wf_probs.std():.4f}  min={wf_probs.min():.4f}  "
          f"max={wf_probs.max():.4f}")

    # Persist back to dataset
    df_out = pd.read_parquet(args.parquet)
    df_out[OUT_COL] = wf_probs.values
    df_out.to_parquet(args.parquet)
    print(f"\n  Parquet updated → {OUT_COL}")

    # Feature importance aggregated across folds
    print(f"\n  Feature importance (summed gain across folds, top 20):")
    imp_sum = np.zeros(len(feat_cols))
    for i, r in enumerate(fold_records):
        with open(OUT_DIR / f"fold_{r['fold_idx']:02d}.pkl", "rb") as f:
            mdl = pickle.load(f)["model"]
        imp_sum += mdl.feature_importances_
    imp_rank = sorted(zip(feat_cols, imp_sum), key=lambda x: -x[1])
    max_i = max(1, imp_rank[0][1])
    for feat, s in imp_rank[:20]:
        bar = "█" * int(s / max_i * 30)
        print(f"    {int(s):>6d}  {bar:<30s}  {feat}")

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps({
        "parquet":          str(args.parquet),
        "anchor_start":     args.anchor_start,
        "first_test_start": args.first_test_start,
        "val_days":         args.val_days,
        "test_days":        args.test_days,
        "step_days":        args.step_days,
        "n_folds":          len(fold_records),
        "folds":            fold_records,
        "lgb_params":       LGB_PARAMS,
        "feat_cols":        feat_cols,
        "output_column":    OUT_COL,
    }, indent=2, default=str))
    print(f"  Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
