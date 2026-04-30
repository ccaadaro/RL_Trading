import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verifier")

# --- Configuration (Hardened V1) ---
DEPLOY_DIR = Path("deployments/baseline_hardened_v1")
META_MODEL_PATH  = DEPLOY_DIR / "gatekeeper.txt"
DATA_PATH = Path("cache/metamodel_training_data.feather")

def run_verification():
    logger.info("Starting Institutional Verification of Frozen Baseline Meta-Gate...")
    
    # 1. Load Data
    df = pd.read_feather(DATA_PATH)
    logger.info(f"Data loaded: {len(df)} bars")
    
    # 2. Load Meta Model
    meta = lgb.Booster(model_file=str(META_MODEL_PATH))
    
    # 3. Features
    meta_feats  = meta.feature_name()
    
    # 4. Meta Inference (The Gatekeeper)
    logger.info("Running Meta inference...")
    df["meta_prob_live"] = meta.predict(df[meta_feats].fillna(0.0))
    
    # 5. Trading Rules (Institutional Thresholds)
    entry_thr = 0.55
    meta_thr = 0.60
    
    df["signal"] = 0
    # Use alpha_prob from the OOF dataset
    df.loc[(df["alpha_prob"] > entry_thr) & (df["meta_prob_live"] >= meta_thr), "signal"] = 1
    
    # 6. Enforce Min Hold (50 bars)
    logger.info("Enforcing 50-bar min hold constraint...")
    signal_arr = df["signal"].values
    last_entry = -1000
    for i in range(len(signal_arr)):
        if signal_arr[i] == 1 and (i - last_entry) < 50:
            # Already in a trade or within hold period, keep signal 1
            pass
        elif signal_arr[i] == 1:
            # New entry
            last_entry = i
        elif (i - last_entry) < 50:
            # Forced hold
            signal_arr[i] = 1
    df["signal"] = signal_arr

    # 7. Performance Calculation
    # We use forward_ret_bps if available, or reconstruct from log_return_feature
    # In metamodel_training_data, forward_ret_bps is the next bar return
    df["trade_cost"] = 0.0007 # 7 bps
    
    df["is_entry"] = (df["signal"] == 1) & (df["signal"].shift(1) == 0)
    
    # Returns in bps -> convert to fraction
    # Sum only on entries to avoid triple-counting the trade horizon
    total_roi = (df.loc[df["is_entry"], "forward_ret_bps"] / 10000).sum() - (df["is_entry"].sum() * 0.0007)
    
    logger.info(f"Verification Result: Net ROI = {total_roi:.4%}")
    
    # Baseline check
    target_roi = 0.0246
    diff = abs(total_roi - target_roi)
    
    if diff <= 0.0005:
        logger.info("SUCCESS: Verification matches baseline within 5 bps tolerance.")
    else:
        logger.error(f"FAILURE: Verification mismatch! Diff = {diff:.4%}")

if __name__ == "__main__":
    run_verification()
