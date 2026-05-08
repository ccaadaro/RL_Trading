import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.signal_features import SIGNAL_FEAT_COLS_V2
from utils.risk_directors import MahalanobisTurbulence, HMMRegimeModel

def generate_metamodel_dataset(data_path, model_path, horizon=50):
    print(f"Loading data from {data_path}...")
    df = pd.read_feather(data_path)
    
    print(f"Loading Alpha model from {model_path}...")
    alpha_model = lgb.Booster(model_file=model_path)
    
    # 1. Generate Alpha Signals (OOF if available)
    oof_path = "cache/alpha_oof_features.feather"
    if os.path.exists(oof_path):
        print(f"Merging OOF Alpha Features from {oof_path}...")
        df_oof = pd.read_feather(oof_path)
        df = df.merge(df_oof, on="date", how="left")
        df["alpha_prob"] = df["alpha_prob_oof"]
        df["alpha_prob_smooth"] = df["alpha_prob_smooth_oof"]
        df["alpha_prob_percentile"] = df["alpha_prob_percentile_oof"]
        print("Using OOF Alpha features for Meta-model data.")
    else:
        print("[WARNING] OOF Alpha Features not found. Falling back to in-sample.")
        features = alpha_model.feature_name()
        df["alpha_prob"] = alpha_model.predict(df[features])
        df["alpha_prob_smooth"] = df["alpha_prob"].ewm(span=30).mean()
        df["alpha_prob_percentile"] = df["alpha_prob"].rolling(10000).rank(pct=True)

    # 2. Derived Alpha Features
    df["alpha_prob_zscore"] = (df["alpha_prob"] - df["alpha_prob"].rolling(10000).mean()) / df["alpha_prob"].rolling(10000).std()
    df["alpha_signal_persistence"] = (df["alpha_prob"] > 0.55).astype(int).rolling(50).sum()
    
    # 3. Market Context (HMM & Turbulence)
    print("Computing Turbulence & HMM states with Leakage-Proof Windowing...")
    turb_engine = MahalanobisTurbulence(window=2000) # Increased window for stability
    hmm_engine = HMMRegimeModel(n_components=3)
    
    risk_vec = ["log_return_feature", "volatility_24_feature", "intraday_range_feature"]
    available_risk = [c for c in risk_vec if c in df.columns]
    
    # Compute turbulence
    df["turbulence_score"] = turb_engine.compute(df, available_risk).fillna(method='bfill').fillna(5.0)
    df["turbulence_percentile"] = df["turbulence_score"].rolling(2000).rank(pct=True)
    
    # HMM Walk-Forward (LEAKAGE FIX)
    hmm_feats = ["log_return_feature", "volatility_24_feature", "aggressor_ratio", "intraday_range_feature"]
    available_hmm = [c for c in hmm_feats if c in df.columns]
    
    n = len(df)
    hmm_states = np.full(n, -1, dtype=int)
    
    # Use 1-year of bars as initial fit (~100k bars if 5m, adjusting for dollar bars)
    # Let's assume 50,000 bars is enough for a baseline HMM
    initial_fit_len = 50000
    print(f"Fitting initial HMM on {initial_fit_len} bars...")
    hmm_engine.fit(df.iloc[:initial_fit_len], available_hmm)
    
    # Re-fit every block to adapt to market regimes without using future data
    block_size = 20000
    for i in range(initial_fit_len, n, block_size):
        end_idx = min(i + block_size, n)
        # Fit on all data up to 'i' (or a window if memory is an issue)
        # To be strictly non-leaking, we fit on df[:i] and predict on df[i:end_idx]
        train_window = df.iloc[max(0, i-100000):i] # Max 100k bars lookback for speed
        hmm_engine.fit(train_window, available_hmm)
        
        # Predict block
        X_block = df.iloc[i:end_idx][available_hmm].fillna(0).values
        hidden_states = hmm_engine.model.predict(X_block)
        hmm_states[i:end_idx] = [hmm_engine.state_map.get(s, -1) for s in hidden_states]
        
    df["hmm_state"] = hmm_states
    
    # 4. Target Labeling (y_meta)
    print(f"Generating Meta-labels (Horizon: {horizon} bars)...")
    
    df["spread_bps"] = df["pr_spread_feature"] * 10000 
    df["expected_cost_bps"] = 5 + df["spread_bps"] + 2 
    
    # Forward return (strictly shifted)
    df["forward_ret_bps"] = (df["close"].shift(-horizon) / df["close"] - 1) * 10000
    
    # y_meta labeling with margin
    # y_meta = 1 if forward_ret_bps > 2 * expected_cost_bps
    df["y_meta"] = (
        (df["forward_ret_bps"] > 2 * df["expected_cost_bps"]) & 
        (df["forward_ret_bps"] > 10.0) # Absolute minimum threshold (1 bps)
    ).astype(int)
    
    # Max Forward Return (MFE) and MAE for secondary analysis
    df["mfe_bps"] = (df["close"].rolling(window=horizon).max().shift(-horizon) / df["close"] - 1) * 10000
    df["mae_bps"] = (df["close"].rolling(window=horizon).min().shift(-horizon) / df["close"] - 1) * 10000
    
    # Save dataset
    output_path = "cache/metamodel_training_data.feather"
    # Keep only relevant columns for training
    meta_cols = [
        "alpha_prob", "alpha_prob_smooth", "alpha_prob_zscore", "alpha_prob_percentile",
        "alpha_signal_persistence", "turbulence_score", "turbulence_percentile",
        "hmm_state", "volatility_24_feature", "aggressor_ratio", "l2_imbalance_feature",
        "spread_bps", "expected_cost_bps", "y_meta", "forward_ret_bps", "mfe_bps", "mae_bps"
    ]
    df[meta_cols].dropna().reset_index(drop=True).to_feather(output_path)
    print(f"Meta-model dataset saved to {output_path}")

if __name__ == "__main__":
    data_path = "cache/dollar_bars_btc_2000000_features.feather"
    model_path = "models/dollar_alpha_v1/latest_model.txt"
    generate_metamodel_dataset(data_path, model_path)
