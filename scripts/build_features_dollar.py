#!/usr/bin/env python3
"""
scripts/build_features_dollar.py

Applies the standard Freqtrade/LGBM feature definitions onto Dollar Bars.
Because 'bars' represent information ($2M transacted) rather than clock-time,
classical indicators (RSI_14) become 'Information RSI', which are 
significantly more robust and stationary.
"""

import sys
import argparse
import time
from pathlib import Path

import pandas as pd
import pyarrow.feather as feather

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from utils.signal_features import build_feature_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=str, default="cache/dollar_bars_btc_2000000.feather")
    ap.add_argument("--labels", type=str, default="cache/dollar_bars_btc_2000000_labeled.feather")
    args = ap.parse_args()

    print(f"Loading full Dollar Bars from {args.bars}...")
    df = feather.read_feather(args.bars)
    
    # Needs volume column
    if 'volume' not in df.columns:
        df['volume'] = df['buy_vol'] + df['sell_vol']

    # Rename buy_vol/sell_vol to match build_feature_matrix expectations
    renames = {}
    if "buy_vol" in df.columns and "buy_volume" not in df.columns:
        renames["buy_vol"] = "buy_volume"
    if "sell_vol" in df.columns and "sell_volume" not in df.columns:
        renames["sell_vol"] = "sell_volume"
    if renames:
        df = df.rename(columns=renames)

    original_cols = set(df.columns)
    
    print("Computing full feature matrix (OHLCV + microstructure)...")
    t0 = time.time()
    
    # build_feature_matrix computes ALL 68 features including:
    # - 53 OHLCV features (RSI, MA, BB, Volatility, etc)
    # - 10 microstructure/order flow features (CVD, aggressor, whale trades)
    # - 3 ETH cross-asset features (set to 0 if no ETH data)
    # - 2 funding features (set to 0 if no funding data)
    X = build_feature_matrix(df, eth_df=None, funding_series=None)
    
    # Merge features back into df for downstream join with labels
    for col in X.columns:
        df[col] = X[col].values
    
    new_cols = [c for c in df.columns if c not in original_cols and c.endswith('_feature')]
    print(f"Computed {len(new_cols)} features in {time.time()-t0:.1f}s")
    
    # Drop nan rows caused by rolling features initially
    df = df.dropna(subset=new_cols)
    print(f"Dropped warmup rows, remaining bars: {len(df):,}")
    
    print(f"\nLoading Labeled Events from {args.labels}...")
    df_labels = feather.read_feather(args.labels)
    
    print("Mapping Information-Time features to Labeled Events...")
    # df_labels has 'start_time' which maps directly to the 'date' column in our full df
    # Both are DataFrames, so we merge
    df_final = pd.merge(df_labels, df, left_on='start_time', right_on='date', how='inner')
    
    out_file = str(Path(args.bars).with_name(Path(args.bars).stem + "_features.feather"))
    
    # Save the consolidated actionable dataset
    df_out = df_final.reset_index(drop=True)
    feather.write_feather(df_out, out_file)
    print(f"Saved {len(df_out):,} labeled-featured rows × {len(df_out.columns)} columns to {out_file}")

if __name__ == "__main__":
    main()
