import sys
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from utils.risk_directors import HMMRegimeModel
from utils.position_sizer import FractionalKellySizer

def run_stacked_verification(df_path, fast_model_path, slow_model_path):
    print(f"Loading data from {df_path}...")
    df = pd.read_feather(df_path)
    
    # Merge Regimes
    reg_path = "cache/dollar_bars_btc_2000000_regimes.feather"
    if Path(reg_path).exists():
        print(f"Merging regimes from {reg_path}...")
        df_reg = pd.read_feather(reg_path)
        # Assuming both have 'date' or 'start_time'
        if 'start_time' in df_reg.columns:
            df = df.merge(df_reg[['start_time', 'hmm_semantic_regime']], on='start_time', how='left')
        elif 'date' in df_reg.columns:
            df = df.merge(df_reg[['date', 'hmm_semantic_regime']], on='date', how='left')
    
    print(f"Loading Fast Model from {fast_model_path}...")
    fast_model = lgb.Booster(model_file=str(fast_model_path))
    
    print(f"Loading Slow Model from {slow_model_path}...")
    with open(slow_model_path, "rb") as f:
        slow_data = pickle.load(f)
    
    # 1. Compute Fast Preds
    feat_fast = fast_model.feature_name()
    X_fast = df[feat_fast].fillna(0.0)
    df["alpha_prob_fast"] = fast_model.predict(X_fast)
    
    # 2. Compute Slow Preds
    slow_models = slow_data.get("models", {})
    weights = slow_data.get("weights", {})
    
    print("Computing Slow Alpha Ensemble...")
    logits = np.zeros(len(df))
    total_w = 0.0
    for h, model in slow_models.items():
        slow_feats = getattr(model, "feature_name", lambda: [])()
        if not slow_feats: continue
        X_s = df[slow_feats].fillna(0.0)
        # Assuming LGBMClassifier (needs predict_proba)
        p = model.predict_proba(X_s)[:, 1]
        p = np.clip(p, 1e-6, 1-1e-6)
        w = weights.get(h, 1.0)
        logits += w * np.log(p / (1 - p))
        total_w += w
    
    if total_w > 0:
        df["alpha_prob_slow"] = 1.0 / (1.0 + np.exp(-logits / total_w))
    else:
        df["alpha_prob_slow"] = 0.5

    # 3. Apply Stacking Logic
    # Consensus Gate: agreement between fast and slow
    is_bull = (df["alpha_prob_slow"] > 0.51) & (df["alpha_prob_fast"] > 0.51)
    is_bear = (df["alpha_prob_slow"] < 0.49) & (df["alpha_prob_fast"] < 0.49)
    df["alpha_prob_stacked"] = 0.5
    df.loc[is_bull | is_bear, "alpha_prob_stacked"] = df["alpha_prob_fast"]
    
    # 4. Run Vectorized Replay
    # We'll compare: Fast-Only vs Stacked
    
    def simulate(preds_col, label=""):
        # Barrier height from previous work
        daily_vol = df["log_return_feature"].rolling(288).std() * np.sqrt(288)
        daily_vol = daily_vol.fillna(0.02)
        barrier = np.maximum(0.003, 1.5 * daily_vol)
        
        edge = 2.0 * df[preds_col].values - 1.0
        # Simple PnL proxy (Return = Edge * Barrier if we trade every bar, which we don't)
        # Real replay is better, but this gives a quick relative ranking.
        
        # Filter: Kill-Switch (Panic/Bear)
        regime = df["hmm_semantic_regime"].values
        is_blackout = np.isin(regime, ["unknown", "panic_selloff", "bear_neutral"])
        
        # Filter: Volume Z-score
        vol_z = df["volume_zscore_24_feature"].fillna(0.0).values
        vol_ok = vol_z >= 0.5
        
        signal = (df[preds_col] > 0.52).values
        # Apply Logic
        final_signal = signal & (~is_blackout) & vol_ok
        
        # Returns (using log_return_feature as proxy for next bar, 
        # but Triple Barrier used actual future labels)
        # Let's use the 'label' from the feather (1 = TP, -1 = SL, 0 = Timeout)
        returns = df["label"].values.astype(float) * 0.04 # 4% target proxy
        
        trades = returns[final_signal]
        if len(trades) == 0:
            return 0, 0, 0
        
        net_pnl = np.sum(trades) - len(trades) * 0.0014 # comisiones
        win_rate = np.mean(trades > 0)
        
        return len(trades), win_rate, net_pnl

    c_fast, wr_fast, pnl_fast = simulate("alpha_prob_fast")
    c_stack, wr_stack, pnl_stack = simulate("alpha_prob_stacked")
    
    print("\n" + "="*50)
    print(f"{'Mode':<15} | {'#Tr':<5} | {'WR':<7} | {'TotalNet':<10}")
    print("-"*50)
    print(f"{'Fast-Only':<15} | {c_fast:<5} | {wr_fast:>6.1%} | {pnl_fast:>+9.2%}")
    print(f"{'Stacked (v5)':<15} | {c_stack:<5} | {wr_stack:>6.1%} | {pnl_stack:>+9.2%}")
    print("="*50)

if __name__ == "__main__":
    run_stacked_verification(
        "cache/dollar_bars_btc_2000000_features.feather",
        "models/dollar_alpha_v1/latest_model.txt",
        "models/signal_lgbm_v2_ensemble.pkl"
    )
