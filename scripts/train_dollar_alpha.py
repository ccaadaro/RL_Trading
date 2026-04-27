#!/usr/bin/env python3
"""
scripts/train_dollar_alpha.py

Trains a LightGBM Binary Classifier on Dollar Bars.
Implements a strict Purged Walk-Forward Cross Validation to avoid data leakage.
"""

import sys
import argparse
import time
import heapq
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

def collect_features(df: pd.DataFrame) -> list:
    """Identify all feature columns automatically."""
    return [c for c in df.columns if c.endswith("_feature")]

LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "auc",
    "boosting_type":     "gbdt",
    "extra_trees":       True,
    "n_estimators":      2000,
    "learning_rate":     0.02,         # Faster learning
    "max_depth":         8,
    "num_leaves":        63,
    "min_child_samples": 100,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.6,
    "reg_alpha":         2.0,          # More L1
    "reg_lambda":        10.0,         # More L2
    "n_jobs":            -1,
    "random_state":      42,
    "verbose":           -1,
}

def compute_uniqueness_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Approximate López de Prado label uniqueness using active concurrency
    at each event start. Lower weight for events that start while many
    prior labels are still alive.
    """
    if "start_time" not in df.columns or "end_time" not in df.columns:
        return np.ones(len(df), dtype=float)

    starts = pd.to_datetime(df["start_time"]).astype("int64").to_numpy()
    ends   = pd.to_datetime(df["end_time"]).astype("int64").to_numpy()

    active_ends: list[int] = []
    weights = np.empty(len(df), dtype=float)

    for i, (start_ns, end_ns) in enumerate(zip(starts, ends)):
        while active_ends and active_ends[0] < start_ns:
            heapq.heappop(active_ends)
        concurrency = len(active_ends) + 1
        weights[i] = 1.0 / concurrency
        heapq.heappush(active_ends, end_ns)

    # Normalise to mean 1.0 so LightGBM sees a stable effective sample size.
    weights *= len(weights) / weights.sum()
    return weights


def compute_recency_weights(df: pd.DataFrame, half_life_bars: int = 50000) -> np.ndarray:
    """
    Exponential decay weighting: recent bars get more weight.
    half_life_bars = number of bars for weight to halve.
    50000 bars ≈ 6 months → bars from 6 months ago have weight 0.5.
    """
    n = len(df)
    if n == 0:
        return np.ones(0, dtype=float)

    decay_rate = np.log(2) / half_life_bars
    positions = np.arange(n, dtype=float)
    weights = np.exp(decay_rate * (positions - (n - 1)))  # newest bar = 1.0

    # Normalise to mean 1.0
    weights *= n / weights.sum()
    return weights

def purged_walk_forward(df, features, folds=5, purge_pct=0.01,
                        sample_weights: np.ndarray | None = None):
    """
    Implements a robust sequential walk-forward over bar indices to absolutely block look-ahead bias.
    purge_pct acts as the 'Embargo' between Train and Test sets.
    """
    print(f"\n--- Starting Purged Walk-Forward ({folds} Folds) ---")
    
    n_samples = len(df)
    fold_size = int(n_samples / folds)
    embargo_size = int(n_samples * purge_pct)
    
    oof_preds = np.full(n_samples, np.nan, dtype=float)
    oof_fold = np.full(n_samples, -1, dtype=int)
    scores = []
    
    # We enforce an expanding window (Anchored Walk-Forward).
    for i in range(1, folds):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < folds - 1 else n_samples

        # True purge: any event whose label horizon reaches into the
        # validation window is excluded from training, even if its row index
        # is earlier than val_start.
        val_start_time = pd.to_datetime(df.iloc[val_start]["start_time"])
        train_mask = np.arange(n_samples) < val_start
        if "end_time" in df.columns:
            train_mask &= pd.to_datetime(df["end_time"]) < val_start_time

        train_idx = np.flatnonzero(train_mask)
        if embargo_size > 0 and len(train_idx) > embargo_size:
            train_idx = train_idx[:-embargo_size]
        val_idx = np.arange(val_start, val_end)
        
        X_train, y_train = df.iloc[train_idx][features], df.iloc[train_idx]["binary_target"]
        X_val, y_val = df.iloc[val_idx][features], df.iloc[val_idx]["binary_target"]
        
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        
        # Early stopping via callbacks in new LGBM versions
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "callbacks": callbacks,
        }
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights[train_idx]
            fit_kwargs["eval_sample_weight"] = [sample_weights[val_idx]]

        model.fit(X_train, y_train, **fit_kwargs)
        
        preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds
        oof_fold[val_idx] = i
        
        auc = roc_auc_score(y_val, preds)
        acc = accuracy_score(y_val, (preds > 0.5).astype(int))
        
        print(f"Fold {i} | Train rows: {len(train_idx):,} | Val: {val_start:,}->{val_end:,} | "
              f"AUC: {auc:.4f} | Acc: {acc:.4f}")
        scores.append(auc)
        
    avg_auc = np.mean(scores)
    print(f"\n=== Walk-Forward Complete | Average OOS AUC: {avg_auc:.4f} ===")
    
    return oof_preds, oof_fold, avg_auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="cache/dollar_bars_btc_2000000_features.feather")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--purge-pct", type=float, default=0.01,
                    help="Additional embargo fraction after the true end_time purge.")
    ap.add_argument("--no-uniqueness-weight", action="store_true",
                    help="Disable concurrency-based sample weights.")
    ap.add_argument(
        "--oof-output",
        type=str,
        default=None,
        help="Where to persist the dataset augmented with oof_pred/oof_valid/oof_fold. "
             "Defaults to overwriting --data so downstream scripts consume real OOFs.",
    )
    args = ap.parse_args()

    print(f"Loading {args.data}...")
    df = feather.read_feather(args.data)
    
    # Remove timeout labels (0) since they are extremely rare (0.0%) and muddy the gradient.
    df = df[df["label"] != 0].copy()
    
    # Map [-1, 1] -> [0, 1] for LightGBM Binary Classifiers
    df["binary_target"] = (df["label"] == 1).astype(int)
    
    # Ensure sequential index for TimeSeries walkforward
    df = df.sort_values('date').reset_index(drop=True)
    
    features = collect_features(df)
    print(f"\nTotal Dataset: {len(df):,} samples")
    print(f"Target Distribution:\n{df['binary_target'].value_counts(normalize=True)}")
    print(f"Features: {len(features)}")

    sample_weights = np.ones(len(df), dtype=float)
    if not args.no_uniqueness_weight:
        uniqueness_w = compute_uniqueness_weights(df)
        sample_weights *= uniqueness_w
        df["uniqueness_weight"] = uniqueness_w
        print(f"Uniqueness weights: min={uniqueness_w.min():.4f}  "
              f"mean={uniqueness_w.mean():.4f}  max={uniqueness_w.max():.4f}")

    # Recency decay: recent bars matter more
    recency_w = compute_recency_weights(df, half_life_bars=50000)
    sample_weights *= recency_w
    df["recency_weight"] = recency_w
    print(f"Recency weights: oldest={recency_w[0]:.4f}  newest={recency_w[-1]:.4f}  "
          f"half_life=50000 bars")

    # Final combined weights
    sample_weights *= len(sample_weights) / sample_weights.sum()  # re-normalize
    print(f"Combined weights: min={sample_weights.min():.4f}  "
          f"mean={sample_weights.mean():.4f}  max={sample_weights.max():.4f}")
    
    t0 = time.time()
    
    # Run the anchored OOS testing
    oof_preds, oof_fold, avg_auc = purged_walk_forward(
        df, features, folds=args.folds, purge_pct=args.purge_pct,
        sample_weights=sample_weights,
    )
    
    df["oof_pred"] = oof_preds
    df["oof_valid"] = pd.Series(oof_preds).notna().values
    df["oof_fold"] = oof_fold
    
    # Only calculate global metric on the validated portions (Folds 1 to N)
    valid_mask = df["oof_valid"]
    final_auc = roc_auc_score(df[valid_mask]["binary_target"], df[valid_mask]["oof_pred"])
    final_acc = accuracy_score(df[valid_mask]["binary_target"], (df[valid_mask]["oof_pred"] > 0.5).astype(int))
    
    print(f"\nGlobal OOS Metric (All Validated Bars): AUC {final_auc:.4f} | Acc {final_acc:.4f}")
    print(f"Total modeling time: {time.time()-t0:.1f}s")
    
    # Retrain on 100% of data to produce the final model
    print("\nRetraining final model on 100% of data for production use...")
    final_params = LGB_PARAMS.copy()
    final_params["n_estimators"] = 1000
    final_params["extra_trees"] = False
    final_model = lgb.LGBMClassifier(**final_params)
    fit_kwargs = {}
    if sample_weights is not None:
        fit_kwargs["sample_weight"] = sample_weights
    final_model.fit(df[features], df["binary_target"], **fit_kwargs)
    
    # Save the model
    out_dir = Path("models/dollar_alpha_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "latest_model.txt"
    final_model.booster_.save_model(str(out_file))
    print(f"Saved Global Alpha Model to {out_file}")

    data_out = Path(args.oof_output) if args.oof_output else Path(args.data)
    feather.write_feather(df.reset_index(drop=True), str(data_out))
    print(f"Persisted real OOF predictions to {data_out}")

if __name__ == "__main__":
    main()
