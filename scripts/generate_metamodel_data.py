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
    
    # 1. Generate Alpha Signals
    features = alpha_model.feature_name()
    print(f"Generating Alpha predictions ({len(features)} features)...")
    df["alpha_prob"] = alpha_model.predict(df[features])
    
    # 2. Alpha Signal Features
    df["alpha_prob_smooth"] = df["alpha_prob"].ewm(span=10).mean()
    df["alpha_prob_zscore"] = (df["alpha_prob"] - df["alpha_prob"].rolling(200).mean()) / df["alpha_prob"].rolling(200).std()
    df["alpha_prob_percentile"] = df["alpha_prob"].rolling(1000).rank(pct=True)
    df["alpha_signal_persistence"] = (df["alpha_prob"] > 0.55).astype(int).rolling(20).sum()
    
    # 3. Market Context (HMM & Turbulence)
    print("Computing Turbulence & HMM states (this may take a few minutes)...")
    turb_engine = MahalanobisTurbulence(window=1000)
    hmm_engine = HMMRegimeModel(n_components=3) # Consistent with 3 states logic in risk_directors
    
    risk_vec = ["log_return_feature", "volatility_24_feature", "intraday_range_feature"]
    available_risk = [c for c in risk_vec if c in df.columns]
    
    # Compute turbulence
    turb_series = turb_engine.compute(df, available_risk)
    df["turbulence_score"] = turb_series.fillna(method='bfill').fillna(5.0)
    df["turbulence_percentile"] = df["turbulence_score"].rolling(2000).rank(pct=True)
    
    # Generate HMM semantic labels
    hmm_feats = ["log_return_feature", "volatility_24_feature", "aggressor_ratio", "intraday_range_feature"]
    available_hmm = [c for c in hmm_feats if c in df.columns]
    
    print("Running HMM fit_predict...")
    # Using fit_predict for bulk processing
    canonical_states, semantic_regimes = hmm_engine.fit_predict(df, available_hmm)
    df["hmm_state"] = canonical_states
    df["regime"] = semantic_regimes
    
    # 4. Target Labeling (y_meta)
    print(f"Generating Meta-labels (Horizon: {horizon} bars)...")
    
    # Calculate costs per bar if possible, or use a realistic estimate
    df["spread_bps"] = df["pr_spread_feature"] * 10000 # Convert to bps
    df["expected_cost_bps"] = 5 + df["spread_bps"] + 2 # 5 fee + spread + 2 slippage
    
    # Forward return
    df["forward_ret_bps"] = (df["close"].shift(-horizon) / df["close"] - 1) * 10000
    
    # y_meta = 1 if forward_ret_bps > 2 * expected_cost_bps AND forward_ret_bps - expected_cost_bps > 0
    # This identifies trades where the move is large enough to cover costs and leave profit.
    df["y_meta"] = (
        (df["forward_ret_bps"] > 2 * df["expected_cost_bps"]) & 
        (df["forward_ret_bps"] > df["expected_cost_bps"])
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
        "hmm_state", "regime", "volatility_24_feature", "aggressor_ratio", "l2_imbalance_feature",
        "spread_bps", "expected_cost_bps", "y_meta", "forward_ret_bps", "mfe_bps", "mae_bps"
    ]
    df[meta_cols].dropna().reset_index(drop=True).to_feather(output_path)
    print(f"Meta-model dataset saved to {output_path}")

if __name__ == "__main__":
    data_path = "cache/dollar_bars_btc_2000000_features.feather"
    model_path = "models/dollar_alpha_v1/latest_model.txt"
    generate_metamodel_dataset(data_path, model_path)
