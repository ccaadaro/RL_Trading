import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_score
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def train_metamodel(data_path, output_model_path):
    print(f"Loading Meta-model dataset from {data_path}...")
    df = pd.read_feather(data_path)
    
    # Filter for signals that pass the Alpha gate (e.g., top 10% of confidence)
    # This aligns with the "Gatekeeper" architecture
    df_signals = df[df["alpha_prob_percentile"] > 0.90].copy()
    
    print(f"Signals passing Alpha Gate: {len(df_signals)} (Total: {len(df)})")
    print(f"Positive samples (y_meta=1): {df_signals['y_meta'].sum()} ({df_signals['y_meta'].mean():.2%})")
    
    if len(df_signals) < 1000:
        print("Warning: Not enough signals to train a robust meta-model. Check labeling or thresholds.")
        return

    # Features for the Meta-model (context-aware)
    features = [
        "alpha_prob", "alpha_prob_smooth", "alpha_prob_zscore", "alpha_prob_percentile",
        "alpha_signal_persistence", "turbulence_score", "turbulence_percentile",
        "hmm_state", "volatility_24_feature", "aggressor_ratio", "l2_imbalance_feature",
        "spread_bps", "expected_cost_bps"
    ]
    
    # Categorical features
    cat_features = ["hmm_state"]
    
    # Encode 'regime' if it was a string, but we use 'hmm_state' (int)
    
    # 1. PURGED TIME-SERIES SPLIT WITH EMBARGO
    horizon = 50
    embargo = 100
    split_idx = int(len(df_signals) * 0.75)
    
    train_df = df_signals.iloc[:split_idx - horizon].copy()
    test_df  = df_signals.iloc[split_idx + embargo:].copy()
    
    # 2. RECENCY SAMPLE WEIGHTING (Exponential half-life decay)
    # 6-month half-life on a 2-year dataset = 4 half-lives. 
    # Start weight = 2^-4 (0.0625), End weight = 2^0 (1.0).
    n_train = len(train_df)
    time_indices = np.linspace(0, 4, n_train)
    weights = 2**(time_indices - 4)
    
    print(f"Training on {len(train_df)} samples (Half-life: 6m), testing on {len(test_df)} samples (purged)...")
    
    X_train, y_train = train_df[features], train_df["y_meta"]
    X_test,  y_test  = test_df[features],  test_df["y_meta"]
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
    
    # LightGBM Parameters (Refined for Meta-models)
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 15, 
        "max_depth": 4,   
        "learning_rate": 0.03,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "verbose": -1,
        "is_unbalance": True,
    }
    
    train_set = lgb.Dataset(X_train, label=y_train, weight=weights, categorical_feature=cat_features)
    test_set  = lgb.Dataset(X_test, label=y_test, categorical_feature=cat_features, reference=train_set)
    
    model = lgb.train(
        params,
        train_set,
        valid_sets=[train_set, test_set],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100), 
            lgb.log_evaluation(period=50)
        ]
    )
    
    # 3. EVALUATION
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.60).astype(int)
    
    print("\n=== Meta-model Evaluation (Hardened OOS) ===")
    print(f"AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")
    print(f"Precision (Tradeability): {precision_score(y_test, y_pred):.4f}")
    
    # Feature Importance Audit
    importances = pd.Series(model.feature_importance(importance_type='gain'), index=features).sort_values(ascending=False)
    print("\nFeature Importance (Gain):")
    print(importances.head(10))
    
    # Key Metric: Expected Net Bps per Trade
    test_df["meta_pred"] = y_pred
    
    net_bps_all = test_df["forward_ret_bps"] - test_df["expected_cost_bps"]
    net_bps_filtered = test_df[test_df["meta_pred"] == 1]["forward_ret_bps"] - test_df[test_df["meta_pred"] == 1]["expected_cost_bps"]
    
    print(f"\nAvg Net Bps (All Alpha Signals in Test): {net_bps_all.mean():.2f}")
    print(f"Avg Net Bps (Meta-Filtered in Test):    {net_bps_filtered.mean():.2f}")
    
    # Save model
    model_dir = Path(output_model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_model_path)
    print(f"Meta-model saved to {output_model_path}")

if __name__ == "__main__":
    data_path = "cache/metamodel_training_data.feather"
    output_model_path = "models/meta_model_v1/gatekeeper.txt"
    if os.path.exists(data_path):
        train_metamodel(data_path, output_model_path)
    else:
        print(f"Error: Dataset {data_path} not found.")
