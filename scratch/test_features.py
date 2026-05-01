import pandas as pd
import numpy as np
import sys
import os

# Add strategy root to path
sys.path.append('/home/nosferatu/freqtrade/user_data/strategies/RL_Trading')

from utils.signal_features import compute_ohlcv_features

def main():
    feather_path = '/home/nosferatu/freqtrade/user_data/strategies/RL_Trading/cache/dollar_bars_btc_2000000.feather'
    if not os.path.exists(feather_path):
        print(f"File not found: {feather_path}")
        return
    
    print(f"Loading {feather_path}...")
    df = pd.read_feather(feather_path)
    
    if 'volume' not in df.columns:
        df['volume'] = df['buy_vol'] + df['sell_vol']
    
    # Ensure it has basic columns
    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(c in df.columns for c in required):
        print(f"Missing required columns: {required}")
        return
    
    # Take a sample
    df_sample = df.tail(1000).copy()
    
    print("Computing features...")
    df_feat = compute_ohlcv_features(df_sample)
    
    new_cols = [
        "hull_hma_55_feature", "hull_hma_slope_feature", "hull_hma_dist_feature",
        "wvf_panic_feature", "wvf_val_feature",
        "laguerre_trend_feature", "laguerre_dispersion_feature",
        "koncorde_osc_pos_feature", "koncorde_osc_neg_feature"
    ]
    
    print("\nFeature Status:")
    for col in new_cols:
        if col in df_feat.columns:
            val = df_feat[col].iloc[-1]
            nan_count = df_feat[col].isna().sum()
            inf_count = np.isinf(df_feat[col]).sum()
            print(f"  - {col:30}: Last Val={val:.6f}, NaNs={nan_count}, Infs={inf_count}")
        else:
            print(f"  - {col:30}: MISSING")

if __name__ == "__main__":
    main()
