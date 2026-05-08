import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss
from pathlib import Path
import sys
import os
import time

# Add strategy root to path
_HERE = Path(__file__).resolve().parent.parent
sys.path.append(str(_HERE))

from scripts.train_dollar_alpha import LGB_PARAMS
from utils.signal_features import SIGNAL_FEAT_COLS_V2

def simulate_pnl(df, preds, threshold=0.01, rebalance_threshold=0.1):
    """Simulate PnL with Long-Only filter and 10% rebalance threshold."""
    n = len(df)
    target_pos = (preds > (0.5 + threshold)).astype(float)
    
    current_pos = 0.0
    actual_pos = np.zeros(n)
    
    for i in range(n):
        if abs(target_pos[i] - current_pos) > rebalance_threshold:
            current_pos = target_pos[i]
        actual_pos[i] = current_pos
        
    # Return proxy: 1% gain on correct bull signal, 1% loss on wrong bull signal
    returns = np.where(df['binary_target'] == 1, 0.01, -0.01)
    pnl = actual_pos * returns
    turnover = np.abs(np.diff(actual_pos, prepend=0)).sum() / n
    
    return pnl.sum(), actual_pos.mean(), turnover

def run_comparison():
    data_path = 'cache/dollar_bars_btc_2000000_features.feather'
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        return

    print(f"Loading dataset {data_path}...")
    df = pd.read_feather(data_path)
    df["binary_target"] = (df["label"] == 1).astype(int)
    
    # Feature Groups
    institutional = [
        "cvd_4h_sum_trade_feature", "aggressor_ratio_4h_mean_trade_feature",
        "whale_trades_4h_sum_trade_feature", "large_trades_4h_sum_trade_feature",
        "max_trade_usd_4h_max_trade_feature", "vwap_skew_4h_mean_trade_feature",
        "whale_intensity_4h_mean_trade_feature", "l2_imbalance_feature", 
        "liq_vola_feature", "cross_exchange_premium_feature",
        "tv_cvd_zscore_feature", "tv_cvd_slope_feature", 
        "tv_aggr_delta_feature", "tv_buy_sell_imbalance_feature"
    ]
    wvf = ["tv_wvf_panic_feature", "tv_wvf_val_feature"]
    percent_r = [
        "pr_exhaust_ob_feature", "pr_exhaust_os_feature",
        "pr_exhaust_ob_reversal_feature", "pr_exhaust_os_reversal_feature",
        "pr_spread_feature"
    ]
    other_tv = [
        "tv_hull_hma_55_feature", "tv_hull_hma_slope_feature",
        "tv_lag_fast_feature", "tv_lag_mid_feature", "tv_lag_slow_feature",
        "tv_lag_dispersion_feature", "tv_pvi_nvi_spread_feature", "tv_tsv_feature"
    ]

    models = {
        "A. Institutional": institutional,
        "B. Inst + WVF": institutional + wvf,
        "C. Inst + WVF + %R": institutional + wvf + percent_r,
        "D. Inst + All TV": institutional + wvf + percent_r + other_tv
    }
    
    # Validation Setup (Purged Walk-Forward 3-fold for speed)
    folds = 3
    n_samples = len(df)
    fold_size = n_samples // folds
    results = []

    for name, features in models.items():
        print(f"\n--- Evaluating Model: {name} ({len(features)} features) ---")
        
        alpha_probs = np.full(n_samples, np.nan)
        
        for i in range(1, folds):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < folds - 1 else n_samples
            
            train_idx = np.arange(0, val_start)
            val_idx = np.arange(val_start, val_end)
            
            X_train, y_train = df.iloc[train_idx][features], df.iloc[train_idx]["binary_target"]
            X_val, y_val = df.iloc[val_idx][features], df.iloc[val_idx]["binary_target"]
            
            model = lgb.LGBMClassifier(**LGB_PARAMS)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      callbacks=[lgb.early_stopping(30, verbose=False)])
            
            alpha_probs[val_idx] = model.predict_proba(X_val)[:, 1]
            
        oos_idx = ~np.isnan(alpha_probs)
        y_oos = df.loc[oos_idx, "binary_target"]
        p_oos = alpha_probs[oos_idx]
        
        auc = roc_auc_score(y_oos, p_oos)
        brier = brier_score_loss(y_oos, p_oos)
        pnl, exposure, turnover = simulate_pnl(df.loc[oos_idx], p_oos)
        
        print(f"Results: AUC={auc:.4f}, Brier={brier:.4f}, PnL={pnl:.4f}, Exposure={exposure:.2%}")
        results.append({
            "Model": name,
            "Features": len(features),
            "AUC": auc,
            "Brier": brier,
            "PnL": pnl,
            "Exposure": exposure,
            "Turnover": turnover
        })

    report = pd.DataFrame(results)
    print("\n--- Model Comparison Report ---")
    print(report.to_string(index=False))
    report.to_csv("logs/model_comparison_report.csv", index=False)

if __name__ == "__main__":
    run_comparison()
