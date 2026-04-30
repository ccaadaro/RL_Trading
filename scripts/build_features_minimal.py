#!/usr/bin/env python3
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pandas_ta as ta

def compute_hma(series, length):
    return ta.hma(series, length)

def compute_wvf_zscore(df, length=22, sma_len=20):
    # Williams Vix Fix
    highest_close = df['close'].rolling(length).max()
    wvf = (highest_close - df['low']) / highest_close * 100
    wvf_sma = wvf.rolling(sma_len).mean()
    wvf_std = wvf.rolling(sma_len).std()
    wvf_z = (wvf - wvf_sma) / wvf_std.clip(lower=1e-9)
    return wvf_z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to bars feather")
    ap.add_argument("--labels", type=str, required=True, help="Path to labels feather")
    args = ap.parse_args()

    print(f"Loading bars from {args.data}...")
    df = feather.read_feather(args.data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Ensure microstructure columns exist
    if 'buy_vol' not in df.columns or 'sell_vol' not in df.columns:
        print("WARNING: buy_vol/sell_vol missing. Microstructure features will be zero.")
        df['buy_vol'] = 0
        df['sell_vol'] = 0
    
    print("Computing minimal feature set (7 features)...")
    t0 = time.time()
    
    close = df['close']
    log_close = np.log(close.clip(lower=1e-9))
    
    # 1 & 2. Returns
    df['return_3_bars_feature'] = log_close.diff(3).fillna(0)
    df['return_5_bars_feature'] = log_close.diff(5).fillna(0)
    
    # 3. Realized Vol (10 bars)
    df['vol_10_feature'] = log_close.diff().rolling(10).std().fillna(0)
    
    # 4. CVD Slope (10 bars)
    cvd = (df['buy_vol'] - df['sell_vol']).cumsum()
    # Simplified slope: difference over window
    df['cvd_slope_feature'] = cvd.diff(10).fillna(0) / 10
    
    # 5. Aggressor Imbalance
    df['aggressor_imbalance_feature'] = ((df['buy_vol'] - df['sell_vol']) / 
                                        (df['buy_vol'] + df['sell_vol']).clip(lower=1e-9)).fillna(0)
    
    # 6. HMA Distance (20 bars)
    hma20 = compute_hma(close, 20)
    df['hma_dist_feature'] = (close / hma20 - 1).fillna(0)
    
    # 7. WVF Z-Score
    df['wvf_zscore_feature'] = compute_wvf_zscore(df).fillna(0)
    
    minimal_features = [
        'return_3_bars_feature', 'return_5_bars_feature', 'vol_10_feature',
        'cvd_slope_feature', 'aggressor_imbalance_feature', 'hma_dist_feature',
        'wvf_zscore_feature'
    ]
    
    print(f"Computed {len(minimal_features)} features in {time.time()-t0:.2f}s")
    
    # Drop rows with NaNs (warmup)
    df = df.dropna(subset=minimal_features)
    
    print(f"Loading Labeled Events from {args.labels}...")
    df_labels = feather.read_feather(args.labels)
    df_labels['start_time'] = pd.to_datetime(df_labels['start_time'])
    
    print("Mapping features to Labeled Events...")
    df_final = pd.merge(df_labels, df, left_on='start_time', right_on='date', how='inner')
    
    # Save output
    stem = Path(args.data).stem
    out_file = str(Path(args.data).with_name(f"{stem}_minimal_features.feather"))
    feather.write_feather(df_final.reset_index(drop=True), out_file)
    print(f"Saved {len(df_final):,} labeled-featured rows to {out_file}")

if __name__ == "__main__":
    main()
