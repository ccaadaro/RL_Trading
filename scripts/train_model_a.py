#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import lightgbm as lgb
import heapq

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

def compute_weights(df):
    starts = pd.to_datetime(df["start_time"]).astype("int64").to_numpy()
    ends = pd.to_datetime(df["end_time"]).astype("int64").to_numpy()
    active_ends = []
    weights = np.empty(len(df), dtype=float)
    for i, (start_ns, end_ns) in enumerate(zip(starts, ends)):
        while active_ends and active_ends[0] < start_ns:
            heapq.heappop(active_ends)
        weights[i] = 1.0 / (len(active_ends) + 1)
        heapq.heappush(active_ends, end_ns)
    # Recency
    n = len(df)
    recency = np.exp(np.log(2) / 50000 * (np.arange(n) - (n - 1)))
    combined = weights * recency
    return combined * (n / combined.sum())

def main():
    df = feather.read_feather("cache/dollar_bars_btc_2000000_features.feather")
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
        "tv_aggr_delta_feature", "tv_buy_sell_imbalance_feature"
    ]
    print(f"Training Model A on {len(features)} features...")
    w = compute_weights(df)
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(df[features], df["binary_target"], sample_weight=w)
    model.booster_.save_model("models/dollar_alpha_v1/model_a_institutional.txt")
    print("Model A saved.")

if __name__ == "__main__":
    main()
