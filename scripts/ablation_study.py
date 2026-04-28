import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss
from pathlib import Path
import sys
import os

# Add strategy root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.train_dollar_alpha import LGB_PARAMS

def simulate_pnl(df, preds, threshold=0.01, rebalance_threshold=0.1):
    """Simulate PnL with Long-Only filter and 10% rebalance threshold."""
    n = len(df)
    # Long-Only: position 1 if pred > 0.5+threshold, else 0
    target_pos = (preds > (0.5 + threshold)).astype(float)
    
    current_pos = 0.0
    actual_pos = np.zeros(n)
    
    for i in range(n):
        # 10% Rebalance Threshold logic
        if abs(target_pos[i] - current_pos) > rebalance_threshold:
            current_pos = target_pos[i]
        actual_pos[i] = current_pos
        
    # Proxy returns: use 'binary_target' (which is label == 1)
    returns = np.where(df['binary_target'] == 1, 0.01, -0.01)
    
    pnl = actual_pos * returns
    turnover = np.abs(np.diff(actual_pos, prepend=0)).sum() / n
    
    return pnl.sum(), actual_pos.mean(), turnover

def run_ablation():
    data_path = 'cache/dollar_bars_btc_2000000_features.feather'
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        return

    print(f"Loading dataset {data_path}...")
    df = pd.read_feather(data_path)
    
    # Sample for speed in ablation
    if len(df) > 20000:
        print("Sampling 20,000 rows for speed...")
        df = df.sample(20000, random_state=42).sort_index()
    
    # Create binary target (Long vs Rest)
    df["binary_target"] = (df["label"] == 1).astype(int)
    
    all_features = [c for c in df.columns if c.endswith("_feature")]
    
    blocks = {
        "Baseline": [c for c in all_features if not c.startswith("tv_") and not "exhaust" in c],
        "Trend": [c for c in all_features if "hull" in c or "lag" in c or "trend" in c],
        "Exhaustion": [c for c in all_features if "exhaust" in c or "rsi" in c],
        "Risk/Panic": [c for c in all_features if "wvf" in c or "volatility" in c],
        "Institutional": [c for c in all_features if any(x in c for x in ["cvd", "aggr", "imbalance", "whale", "large"])],
        "All": all_features
    }
    
    # Correlation Pruning
    print("\nChecking for high correlations (> 0.95)...")
    corr_matrix = df[all_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    if to_drop:
        print(f"Pruning {len(to_drop)} redundant features: {to_drop}")
        all_features = [f for f in all_features if f not in to_drop]
        # Re-update blocks
        for name in blocks:
            blocks[name] = [f for f in blocks[name] if f in all_features]

    # Clean NaNs and Infs
    df[all_features] = df[all_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    results = []
    # Use 3-fold for speed in ablation
    folds = 3
    n_samples = len(df)
    fold_size = n_samples // folds
    
    for name, features in blocks.items():
        if not features: continue
        print(f"\n--- Testing Block: {name} ({len(features)} features) ---")
        
        oof_preds = np.full(n_samples, 0.5)
        
        for i in range(1, folds):
            print(f"  Fold {i}/{folds-1}...")
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < folds - 1 else n_samples
            
            train_idx = np.arange(0, val_start)
            val_idx = np.arange(val_start, val_end)
            
            X_train, y_train = df.iloc[train_idx][features], df.iloc[train_idx]["binary_target"]
            X_val, y_val = df.iloc[val_idx][features], df.iloc[val_idx]["binary_target"]
            
            params = LGB_PARAMS.copy()
            params["n_estimators"] = 200
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      callbacks=[lgb.early_stopping(20, verbose=False)])
            
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            
        # Metrics on the OOS part (folds 1 and 2)
        oos_idx = np.arange(fold_size, n_samples)
        y_oos = df.iloc[oos_idx]["binary_target"]
        p_oos = oof_preds[oos_idx]
        
        auc = roc_auc_score(y_oos, p_oos)
        brier = brier_score_loss(y_oos, p_oos)
        pnl, exposure, turnover = simulate_pnl(df.iloc[oos_idx], p_oos)
        
        print(f"Results for {name}: AUC={auc:.4f}, Brier={brier:.4f}, PnL={pnl:.4f}, Exposure={exposure:.2%}, Turnover={turnover:.4f}")
        results.append({
            "Block": name,
            "Features": len(features),
            "AUC": auc,
            "Brier": brier,
            "PnL": pnl,
            "Exposure": exposure,
            "Turnover": turnover
        })

    print("\n--- Final Ablation Report ---")
    report = pd.DataFrame(results).sort_values("PnL", ascending=False)
    print(report.to_string(index=False))
    
    report.to_csv("logs/ablation_report.csv", index=False)

if __name__ == "__main__":
    run_ablation()
