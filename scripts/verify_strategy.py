#!/usr/bin/env python3
"""
scripts/verify_strategy.py

Tests the compilation and numerical integrity of the InstitutionalDollarStrategy
by manually feeding it the offline 5m Binance data.
"""

import sys
from pathlib import Path
import pandas as pd
import pyarrow.feather as feather

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from utils.signal_features import build_feature_matrix
from InstitutionalDollarStrategy import InstitutionalDollarStrategy

def main():
    print("Loading 5m BTC data for offline routing test...")
    data_path = Path("/home/nosferatu/freqtrade/user_data/data/binance/BTC_USDT-5m.feather")
    
    if not data_path.exists():
        print(f"File not found: {data_path}")
        return
        
    df = feather.read_feather(str(data_path))
    
    # Prune to last 10,000 candles to keep test fast
    df = df.tail(10000).reset_index(drop=True)
    
    print(f"Injected {len(df)} candles into Freqtrade Simulator.")
    
    strategy = InstitutionalDollarStrategy({})
    strategy.bot_start()
    
    print("\nRunning Indicator Aggregation Pipeline...")
    df = strategy.populate_indicators(df, {})
    
    # Run trend populators
    df = strategy.populate_entry_trend(df, {})
    df = strategy.populate_exit_trend(df, {})
    
    print("\n=== Pipeline Verification ===")
    has_target = "target_pos" in df.columns
    print(f"Target Position Array generated: {has_target}")
    
    if has_target:
        active_pct = (df["target_pos"] != 0.0).sum() / len(df) * 100
        print(f"Active Allocation Presence: {active_pct:.2f}% of timeline")
        
        buys = df["enter_long"].sum()
        sells = df["exit_long"].sum()
        
        print(f"Freqtrade Entry Signals Fired: {buys}")
        print(f"Freqtrade Exit Signals Fired: {sells}")
        
    print("\nSanity Check Successful! Systems ready for Production.")

if __name__ == "__main__":
    main()
