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
    
    X = df_signals[features]
    y = df_signals["y_meta"]
    
    # Time-based split (No shuffling for time-series)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
    
    # LightGBM Parameters
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "is_unbalance": True, # To handle potential label imbalance
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    test_data = lgb.Dataset(X_test, label=y_test, categorical_feature=cat_features, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, test_data],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # Evaluation
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.60).astype(int) # Using the user's recommended threshold
    
    print("\n=== Meta-model Evaluation (OOS) ===")
    print(f"AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")
    print(f"Precision (Tradeability): {precision_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    # Key Metric: Expected Net Bps per Trade
    # We map the predictions back to the original df_signals to see net profit
    df_test = df_signals.iloc[split_idx:].copy()
    df_test["meta_pred"] = y_pred
    
    net_bps_all = df_test["forward_ret_bps"] - df_test["expected_cost_bps"]
    net_bps_filtered = df_test[df_test["meta_pred"] == 1]["forward_ret_bps"] - df_test[df_test["meta_pred"] == 1]["expected_cost_bps"]
    
    print(f"Avg Net Bps (All Alpha Signals): {net_bps_all.mean():.2f}")
    print(f"Avg Net Bps (Meta-Filtered):    {net_bps_filtered.mean():.2f}")
    
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
