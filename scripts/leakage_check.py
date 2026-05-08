import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add strategy root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.signal_features import build_feature_matrix

def check_leakage():
    data_path = 'cache/dollar_bars_btc_2000000.feather'
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        return

    print(f"Loading Dollar Bars {data_path}...")
    df_full = pd.read_feather(data_path).head(5000) # Larger subset
    
    # 1. Compute features on the first 2000 bars
    df_subset = df_full.iloc[:2000].copy()
    feat_subset = build_feature_matrix(df_subset)
    
    # 2. Compute features on the first 2001 bars
    df_plus_one = df_full.iloc[:2001].copy()
    feat_plus_one = build_feature_matrix(df_plus_one)
    
    # 3. Compare the first 2000 bars of both
    # We ignore the very first rows due to warmup NaNs/Infs
    comparison_start = 500 
    comparison_end = 2000
    
    diffs = (feat_subset.iloc[comparison_start:comparison_end] != feat_plus_one.iloc[comparison_start:comparison_end])
    
    leakage_found = False
    for col in diffs.columns:
        n_diff = diffs[col].sum()
        if n_diff > 0:
            # Check if diff is significant (floating point noise)
            val1 = feat_subset.iloc[comparison_start:comparison_end][col].values
            val2 = feat_plus_one.iloc[comparison_start:comparison_end][col].values
            max_abs_diff = np.abs(val1 - val2).max()
            
            if max_abs_diff > 1e-10:
                print(f"[!] LEAKAGE DETECTED in feature: {col} (Max Diff: {max_abs_diff})")
                leakage_found = True
    
    if not leakage_found:
        print("[v] No lookahead bias detected in feature matrix (causality verified).")
    else:
        print("[x] Leakage check failed. Review the features listed above.")

if __name__ == "__main__":
    check_leakage()
