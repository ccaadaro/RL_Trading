#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys

def build_1h_candles(trades_path: str, output_path: str):
    print(f"Loading trades from {trades_path}...")
    df = pd.read_feather(trades_path)
    
    # Ensure datetime (Schema fix: use 'timestamp' instead of 'date')
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    elif not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], unit='ms', utc=True)
    
    df = df.set_index('date')
    
    print("Resampling to 1h candles...")
    ohlc = df['price'].resample('1h').ohlc()
    volume = df['amount'].resample('1h').sum()
    
    # Microstructure features
    # Note: If side is not available, we use heuristics.
    # Based on the daemon log, we have 'price' and 'amount'.
    # If we have 'side', we use it.
    if 'side' in df.columns:
        buys = df[df['side'] == 'buy']['amount'].resample('1h').sum().fillna(0)
        sells = df[df['side'] == 'sell']['amount'].resample('1h').sum().fillna(0)
        cvd = (buys - sells).cumsum()
        aggr_ratio = buys / (buys + sells + 1e-9)
    else:
        # Heuristic: tick-rule (last price change)
        df['diff'] = df['price'].diff()
        df['side_h'] = np.sign(df['diff']).replace(0, method='ffill')
        buys = df[df['side_h'] > 0]['amount'].resample('1h').sum().fillna(0)
        sells = df[df['side_h'] < 0]['amount'].resample('1h').sum().fillna(0)
        cvd = (buys - sells).cumsum()
        aggr_ratio = buys / (buys + sells + 1e-9)

    res = pd.concat([ohlc, volume, cvd.rename('cvd'), aggr_ratio.rename('aggressor_ratio')], axis=1)
    res = res.dropna(subset=['close'])
    res = res.reset_index()
    
    print(f"Saving {len(res)} candles to {output_path}...")
    res.to_feather(output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    build_1h_candles(args.trades, args.output)
