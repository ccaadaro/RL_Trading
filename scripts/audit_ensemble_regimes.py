#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def compute_metrics(price, signals, v_bars=10, cost_rate=0.0007):
    n = len(price)
    rets = price.pct_change().fillna(0).values
    sigs = signals.values
    
    active_counts = np.zeros(n)
    for i in np.where(sigs == 1)[0]:
        active_counts[i : min(i + v_bars, n)] += 1
        
    target_exposure = np.where(active_counts > 0, 1.0, 0.0)
    p_rets = np.zeros(n)
    p_rets[1:] = target_exposure[:-1] * rets[1:]
    
    costs = np.abs(np.diff(target_exposure, prepend=0)) * cost_rate
    net_rets = p_rets - costs
    equity = np.cumprod(1 + net_rets)
    
    roi = equity[-1] - 1
    dd = np.min(equity / np.maximum.accumulate(equity) - 1) if n > 0 else 0
    tim = np.mean(active_counts > 0)
    return {"roi": roi, "dd": dd, "tim": tim, "net_rets": net_rets}

def main():
    feat_path = "cache/dollar_bars_btc_100000000_v2_features.feather"
    if not Path(feat_path).exists():
        print(f"Error: {feat_path} not found. Run battery first.")
        return

    df = pd.read_feather(feat_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['binary_target'] = (df['label'] == 1).astype(int)

    f_v1 = [
        "return_3_bars_feature", "return_5_bars_feature", "vol_10_feature",
        "cvd_slope_feature", "aggressor_imbalance_feature", "hma_dist_feature",
        "wvf_zscore_feature"
    ]
    f_v2 = f_v1 + ["cvd_divergence_feature", "hma_slope_feature", "vol_accel_feature"]

    # Training Split (2023)
    train_df = df[df['date'] < '2024-01-01'].copy()
    
    # Models
    params = {"max_depth": 2, "n_estimators": 200, "learning_rate": 0.05, "verbosity": -1}
    
    print("Training Specialist v1...")
    m_v1 = lgb.LGBMClassifier(**params)
    m_v1.fit(train_df[f_v1], train_df['binary_target'])
    
    print("Training Specialist v2...")
    m_v2 = lgb.LGBMClassifier(**params)
    m_v2.fit(train_df[f_v2], train_df['binary_target'])

    # Full Dataset for Evaluation (2024-2025)
    eval_df = df[df['date'] >= '2024-01-01'].copy()
    eval_df['prob_v1'] = m_v1.predict_proba(eval_df[f_v1])[:, 1]
    eval_df['prob_v2'] = m_v2.predict_proba(eval_df[f_v2])[:, 1]
    
    # Thresholds (60th percentile of probs in training or val)
    t_v1 = eval_df['prob_v1'].quantile(0.60)
    t_v2 = eval_df['prob_v2'].quantile(0.60)
    
    eval_df['sig_v1'] = (eval_df['prob_v1'] > t_v1).astype(int)
    eval_df['sig_v2'] = (eval_df['prob_v2'] > t_v2).astype(int)
    
    # Ensemble B: Mean
    eval_df['prob_mean'] = 0.5 * eval_df['prob_v1'] + 0.5 * eval_df['prob_v2']
    t_mean = eval_df['prob_mean'].quantile(0.60)
    eval_df['sig_mean'] = (eval_df['prob_mean'] > t_mean).astype(int)
    
    # Ensemble A: Consensus
    eval_df['sig_cons'] = (eval_df['sig_v1'] & eval_df['sig_v2']).astype(int)

    # Regime Definitions
    eval_df['reg_trend'] = eval_df['hma_slope_feature'].abs() > eval_df['hma_slope_feature'].abs().median()
    eval_df['reg_div'] = eval_df['cvd_divergence_feature'] == 1
    eval_df['reg_panic'] = eval_df['wvf_zscore_feature'] > 1.5
    eval_df['reg_vol'] = eval_df['vol_10_feature'] > eval_df['vol_10_feature'].median()

    periods = {
        "VAL_2024": eval_df[eval_df['date'] < '2025-01-01'],
        "OBS_2025": eval_df[eval_df['date'] >= '2025-01-01']
    }

    models = ["sig_v1", "sig_v2", "sig_mean", "sig_cons"]
    
    summary = []
    for p_name, p_df in periods.items():
        if len(p_df) == 0: continue
        for m in models:
            perf = compute_metrics(p_df['close'], p_df[m])
            summary.append({
                "period": p_name, "model": m, "roi": perf['roi'], "tim": perf['tim']
            })
            
        # Regime Breakdown for Consensus
        for reg in ['reg_trend', 'reg_div', 'reg_panic', 'reg_vol']:
            reg_df = p_df[p_df[reg]]
            if len(reg_df) < 50: continue
            perf = compute_metrics(reg_df['close'], reg_df['sig_cons'])
            summary.append({
                "period": f"{p_name}_{reg}", "model": "cons", "roi": perf['roi'], "tim": perf['tim']
            })

    print("\n--- ENSEMBLE AUDIT SUMMARY ---")
    print(pd.DataFrame(summary).to_string(index=False))

if __name__ == "__main__":
    main()
