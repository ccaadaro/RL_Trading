#!/usr/bin/env python3
import os
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pyarrow.feather as feather

# Config
THETA = 2_000_000 # 2M USDT
DATA_DIR = Path("/home/nosferatu/freqtrade/user_data/data/binance/aggTrades")
OUT_FILE = Path("/home/nosferatu/freqtrade/user_data/strategies/RL_Trading/cache/dollar_bars_tick_v1.feather")

def process_zip(zip_path):
    """Processes a single aggTrade ZIP and returns trades dataframe."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_name = z.namelist()[0]
        # agg_trade_id,price,qty,first_id,last_id,time,is_buyer_maker,is_best_match
        cols = ["id", "price", "quantity", "f", "l", "ts", "is_buyer_maker", "m"]
        with z.open(csv_name) as f:
            df = pd.read_csv(f, names=cols, header=None)
            
    # Cleanup and format
    df["notional"] = df["price"] * df["quantity"]
    # Aggressor Logic: is_buyer_maker=True means Taker Sell (Maker was buyer)
    df["buy_vol"] = np.where(df["is_buyer_maker"] == False, df["quantity"], 0.0)
    df["sell_vol"] = np.where(df["is_buyer_maker"] == True, df["quantity"], 0.0)
    df["buy_notional"] = np.where(df["is_buyer_maker"] == False, df["notional"], 0.0)
    df["sell_notional"] = np.where(df["is_buyer_maker"] == True, df["notional"], 0.0)
    
    return df[["ts", "price", "quantity", "notional", "buy_vol", "sell_vol", "buy_notional", "sell_notional"]]

def main():
    # 1. Process ETH first to create a lookup table
    print("\n[ETH] Resampling ETH ticks to 1m lookup...")
    eth_files = sorted(list(DATA_DIR.glob("ETHUSDT-aggTrades-*.zip")))
    eth_records = []
    
    for eth_f in tqdm(eth_files, desc="Resampling ETH"):
        df_eth = process_zip(eth_f)
        df_eth["date"] = pd.to_datetime(df_eth["ts"], unit="us", utc=True)
        # Resample to 1m to have a continuous series for cross-asset features
        df_eth_1m = df_eth.set_index("date")["price"].resample("1m").last().ffill()
        eth_records.append(df_eth_1m)
    
    eth_lookup = pd.concat(eth_records).sort_index()

    # 2. Process BTC and build Dollar Bars
    print("\n[BTC] Reconstructing BTC Dollar Bars...")
    btc_files = sorted(list(DATA_DIR.glob("BTCUSDT-aggTrades-*.zip")))
    
    if not btc_files:
        print("No BTC ZIP files found. Wait for the downloader.")
        return

    bars = []
    leftover_notional = 0.0
    
    for zip_f in tqdm(btc_files, desc="Building BTC Bars"):
        df = process_zip(zip_f)
        df["cum_notional"] = leftover_notional + df["notional"].cumsum()
        
        # Exact Partitioning
        df["global_bar_id"] = (df["cum_notional"] // THETA).astype(int)
        
        day_bars = df.groupby("global_bar_id").agg(
            date=("ts", "last"),
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("quantity", "sum"),
            notional=("notional", "sum"),
            buy_vol=("buy_vol", "sum"),
            sell_vol=("sell_vol", "sum"),
            buy_notional=("buy_notional", "sum"),
            sell_notional=("sell_notional", "sum"),
            trade_count=("price", "count")
        )
        
        day_bars["date"] = pd.to_datetime(day_bars["date"], unit="us", utc=True)
        bars.append(day_bars)
        leftover_notional = df["cum_notional"].iloc[-1] % THETA

    # 3. Final Assembly and Cross-Asset Join
    print("\n[MERGE] Aligning BTC and ETH...")
    full_df = pd.concat(bars).sort_values("date")
    
    # CVD and Aggressor Ratio (Real features)
    full_df["cvd"] = full_df["buy_notional"] - full_df["sell_notional"]
    full_df["aggressor_ratio"] = full_df["buy_vol"] / full_df["volume"]
    
    # Join ETH (Asof join to match BTC bar timestamp)
    full_df = full_df.reset_index(drop=True)
    full_df = pd.merge_asof(
        full_df.sort_values("date"), 
        eth_lookup.rename("eth_close").to_frame(),
        on="date", 
        direction="backward"
    )
    
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(full_df.reset_index(drop=True), str(OUT_FILE))
    print(f"\n[SUCCESS] Reconstructed {len(full_df):,} High-Fidelity Multi-Asset Bars.")

if __name__ == "__main__":
    main()
