#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05, # Higher learning rate for speed
    "max_depth": 5,        # Shallower for speed
    "num_leaves": 15,
    "min_child_samples": 500,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

ALPHA_FEATURES = [
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

def generate_alpha_oof(data_path, output_path, folds=6, embargo_pct=0.01):
    print(f"Loading features from {data_path}...")
    df = pd.read_feather(data_path)
    df = df.sort_values('date').reset_index(drop=True)
    
    labeled_mask = (df["label"] != 0).values
    X_all = df[ALPHA_FEATURES].values.astype(np.float32)
    y_all = (df["label"] == 1).astype(int).values
    
    n_samples = len(df)
    alpha_probs = np.full(n_samples, np.nan)
    
    # Simple walk-forward folds
    fold_size = n_samples // folds
    embargo_size = int(n_samples * embargo_pct)
    
    print(f"Starting Fast OOF generation with {folds} folds...")
    
    for i in range(1, folds):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < folds - 1 else n_samples
        
        # Training: everything before val_start - embargo
        train_end = val_start - embargo_size
        train_idx = np.where(labeled_mask[:train_end])[0]
        
        if len(train_idx) < 5000:
            print(f"Fold {i}: Skipping (not enough training data)")
            continue
            
        # Validation: use first 10% of test fold for early stopping
        val_len = (val_end - val_start) // 5
        val_idx = np.where(labeled_mask[val_start : val_start + val_len])[0] + val_start
        test_idx = np.arange(val_start, val_end)
        
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]
        
        print(f"Fold {i} | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(test_idx)}")
        
        train_set = lgb.Dataset(X_train, label=y_train)
        valid_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
        
        model = lgb.train(
            LGB_PARAMS,
            train_set,
            valid_sets=[valid_set],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=0)]
        )
        
        probs = model.predict(X_all[test_idx])
        alpha_probs[test_idx] = probs
        
        auc = roc_auc_score(y_all[test_idx][labeled_mask[test_idx]], probs[labeled_mask[test_idx]])
        print(f"Fold {i} AUC: {auc:.4f}")

    # Results
    df["alpha_prob_oof"] = alpha_probs
    df["alpha_prob_smooth_oof"] = df["alpha_prob_oof"].ewm(span=30, min_periods=1).mean()
    
    # Roll-rank for percentile (approximate to speed up)
    print("Computing rolling percentile (approx)...")
    df["alpha_prob_percentile_oof"] = df["alpha_prob_oof"].rolling(10000, min_periods=1000).rank(pct=True)
    
    print(f"Saving OOF results to {output_path}...")
    df[["date", "alpha_prob_oof", "alpha_prob_smooth_oof", "alpha_prob_percentile_oof"]].to_feather(output_path)
    print("OOF generation complete.")

if __name__ == "__main__":
    generate_alpha_oof("cache/dollar_bars_btc_2000000_features.feather", "cache/alpha_oof_features.feather")
