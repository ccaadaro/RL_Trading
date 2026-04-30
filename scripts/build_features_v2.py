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
    
    if 'buy_vol' not in df.columns or 'sell_vol' not in df.columns:
        df['buy_vol'] = 0
        df['sell_vol'] = 0
    
    print("Computing v2 feature set (10 features)...")
    t0 = time.time()
    
    close = df['close']
    log_close = np.log(close.clip(lower=1e-9))
    
    # --- BASE 7 FEATURES ---
    df['return_3_bars_feature'] = log_close.diff(3).fillna(0)
    df['return_5_bars_feature'] = log_close.diff(5).fillna(0)
    df['vol_10_feature'] = log_close.diff().rolling(10).std().fillna(0)
    
    cvd = (df['buy_vol'] - df['sell_vol']).cumsum()
    df['cvd_slope_feature'] = cvd.diff(10).fillna(0) / 10
    
    df['aggressor_imbalance_feature'] = ((df['buy_vol'] - df['sell_vol']) / 
                                        (df['buy_vol'] + df['sell_vol']).clip(lower=1e-9)).fillna(0)
    
    hma20 = compute_hma(close, 20)
    df['hma_dist_feature'] = (close / hma20 - 1).fillna(0)
    df['wvf_zscore_feature'] = compute_wvf_zscore(df).fillna(0)
    
    # --- V2 SECOND-ORDER FEATURES ---
    # 8. CVD Divergence (Price slope vs CVD slope)
    price_slope = close.diff(10).fillna(0) / 10
    # Normalize by price to make it comparable to CVD units (scale matters)
    # Better: rank-based or sign-based divergence
    df['cvd_divergence_feature'] = (np.sign(price_slope) != np.sign(df['cvd_slope_feature'])).astype(float)
    
    # 9. HMA Slope (Rate of change of trend)
    df['hma_slope_feature'] = hma20.diff(5).fillna(0) / 5
    
    # 10. Volatility Acceleration
    df['vol_accel_feature'] = df['vol_10_feature'].diff(5).fillna(0)
    
    v2_features = [
        'return_3_bars_feature', 'return_5_bars_feature', 'vol_10_feature',
        'cvd_slope_feature', 'aggressor_imbalance_feature', 'hma_dist_feature',
        'wvf_zscore_feature', 'cvd_divergence_feature', 'hma_slope_feature',
        'vol_accel_feature'
    ]
    
    print(f"Computed {len(v2_features)} features in {time.time()-t0:.2f}s")
    
    df = df.dropna(subset=v2_features)
    df_labels = feather.read_feather(args.labels)
    df_labels['start_time'] = pd.to_datetime(df_labels['start_time'])
    
    df_final = pd.merge(df_labels, df, left_on='start_time', right_on='date', how='inner')
    
    stem = Path(args.data).stem
    out_file = str(Path(args.data).with_name(f"{stem}_v2_features.feather"))
    feather.write_feather(df_final.reset_index(drop=True), out_file)
    print(f"Saved {len(df_final):,} labeled-featured rows to {out_file}")

if __name__ == "__main__":
    main()
