#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import time
import heapq
import json
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "learning_rate": 0.015,
    "max_depth": 7,
    "num_leaves": 31,
    "min_child_samples": 200,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_alpha": 5.0,
    "reg_lambda": 20.0,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

def compute_uniqueness_weights(df):
    if "start_time" not in df.columns or "end_time" not in df.columns:
        return np.ones(len(df), dtype=float)
    starts = pd.to_datetime(df["start_time"]).astype("int64").to_numpy()
    ends = pd.to_datetime(df["end_time"]).astype("int64").to_numpy()
    active_ends = []
    weights = np.empty(len(df), dtype=float)
    for i, (start_ns, end_ns) in enumerate(zip(starts, ends)):
        while active_ends and active_ends[0] < start_ns:
            heapq.heappop(active_ends)
        concurrency = len(active_ends) + 1
        weights[i] = 1.0 / concurrency
        heapq.heappush(active_ends, end_ns)
    weights *= len(weights) / weights.sum()
    return weights

def compute_recency_weights(df, half_life_bars=50000):
    n = len(df)
    decay_rate = np.log(2) / half_life_bars
    positions = np.arange(n, dtype=float)
    weights = np.exp(decay_rate * (positions - (n - 1)))
    weights *= n / weights.sum()
    return weights

def purged_walk_forward(df, features, folds=5, purge_pct=0.01, sample_weights=None):
    n_samples = len(df)
    fold_size = int(n_samples / folds)
    embargo_size = int(n_samples * purge_pct)
    alpha_probs = np.full(n_samples, np.nan, dtype=float)
    scores = []
    for i in range(1, folds):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < folds - 1 else n_samples
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
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        fit_kwargs = {"eval_set": [(X_val, y_val)], "callbacks": callbacks}
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights[train_idx]
        model.fit(X_train, y_train, **fit_kwargs)
        preds = model.predict_proba(X_val)[:, 1]
        alpha_probs[val_idx] = preds
        auc = roc_auc_score(y_val, preds)
        print(f"Fold {i} | AUC: {auc:.4f}")
        scores.append(auc)
    return alpha_probs, np.mean(scores)

def main():
    data_path = "cache/dollar_bars_btc_2000000_features.feather"
    df = feather.read_feather(data_path)
    df = df[df["label"] != 0].copy()
    df["binary_target"] = (df["label"] == 1).astype(int)
    df = df.sort_values('date').reset_index(drop=True)
    
    features = [
        "cvd_4h_sum_trade_feature", "aggressor_ratio_4h_mean_trade_feature",
        "whale_trades_4h_sum_trade_feature", "large_trades_4h_sum_trade_feature",
        "max_trade_usd_4h_max_trade_feature", "vwap_skew_4h_mean_trade_feature",
        "whale_intensity_4h_mean_trade_feature", "l2_imbalance_feature", 
        "liq_vola_feature", "cross_exchange_premium_feature",
        "tv_cvd_zscore_feature", "tv_cvd_slope_feature", 
        "tv_aggr_delta_feature", "tv_buy_sell_imbalance_feature",
        "tv_wvf_panic_feature", "tv_wvf_val_feature",
        "pr_exhaust_ob_feature", "pr_exhaust_os_feature",
        "pr_exhaust_ob_reversal_feature", "pr_exhaust_os_reversal_feature",
        "pr_spread_feature"
    ]
    print(f"Training on {len(features)} features...")
    
    sample_weights = compute_uniqueness_weights(df) * compute_recency_weights(df)
    sample_weights *= len(df) / sample_weights.sum()
    
    alpha_probs, avg_auc = purged_walk_forward(df, features, sample_weights=sample_weights)
    print(f"Avg AUC: {avg_auc:.4f}")
    
    final_model = lgb.LGBMClassifier(**LGB_PARAMS)
    final_model.fit(df[features], df["binary_target"], sample_weight=sample_weights)
    
    out_dir = Path("models/dollar_alpha_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    final_model.booster_.save_model(str(out_dir / "latest_model.txt"))
    print("Model saved.")

if __name__ == "__main__":
    main()
