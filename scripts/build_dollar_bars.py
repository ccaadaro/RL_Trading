#!/usr/bin/env python3
"""
scripts/build_dollar_bars.py — aggregate 26GB tick trades into Dollar Bars.

Instead of fixed time windows (cron-clock), this builds bars that close
every time a specified `theta` worth of USD volume is transacted 
(volume-clock / dollar-clock).

Input:  BTC_USDT-trades.feather (contains timestamp, price, amount, cost, side)
Output: cache/dollar_bars_btc_${THETA_USD}.feather

Usage:
    python scripts/build_dollar_bars.py --theta 2000000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather


TRADES_PATH = "/home/nosferatu/freqtrade/user_data/data/binance/BTC_USDT-trades.feather"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", type=str, default=TRADES_PATH)
    ap.add_argument("--theta", type=float, default=2_000_000.0,
                    help="Target Dollar Volume per bar (e.g., 2000000 for 2M USDT)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit number of batches to process for quick testing. 0 = all.")
    args = ap.parse_args()

    out_file = f"cache/dollar_bars_btc_{int(args.theta)}.feather"
    print(f"  Reading : {args.trades}")
    print(f"  Theta   : ${args.theta:,.0f} per bar")
    print(f"  Output  : {out_file}")

    reader = pa.ipc.RecordBatchFileReader(pa.memory_map(args.trades))
    n_batches = reader.num_record_batches
    print(f"  Batches : {n_batches:,}")
    if args.limit > 0:
        n_batches = min(n_batches, args.limit)
        print(f"  (Limiting to first {n_batches} batches)")

    t0 = time.time()
    last_report = t0
    total_rows = 0

    running_stats = {} # dictionary of stats across boundaries
    leftover_cusum = 0.0

    for batch_idx in range(n_batches):
        b = reader.get_batch(batch_idx).to_pandas()
        
        # Calculate continuous cumulative sum of cost
        b_cost = b["cost"].to_numpy(dtype=np.float64)
        c_sum = leftover_cusum + b_cost.cumsum()
        
        # Determine the target bar index for each trade
        b["bar_id"] = (c_sum // args.theta).astype(np.int64)
        leftover_cusum = c_sum[-1]

        # Extract features
        b["ts"] = pd.to_datetime(b["timestamp"], unit="ms", utc=True)

        # D-08 FIX: Use is_buyer_maker (true aggressor flag) when available.
        # is_buyer_maker=False means a buyer TOOK liquidity (aggressive buy).
        # is_buyer_maker=True  means a seller TOOK liquidity (aggressive sell).
        # The daemon live uses is_buyer_maker for CVD. Training data must match.
        # Fallback: side=="buy" is a proxy but has different semantics
        # (it classifies maker+taker buys together, diluting the aggressor signal).
        if "is_buyer_maker" in b.columns:
            # Aggressive buy = NOT buyer_maker (taker bought from maker)
            is_aggressor_buy = (~b["is_buyer_maker"].astype(bool)).to_numpy()
        elif "side" in b.columns:
            is_aggressor_buy = (b["side"] == "buy").to_numpy()
            if batch_idx == 0:  # warn only once
                print(
                    "  [D-08 WARNING] is_buyer_maker not found — using side=='buy' as fallback.\n"
                    "  CVD will not distinguish maker vs taker buys. "
                    "This creates a train/live feature mismatch for microstructure features.\n"
                    "  Fix: export aggTrade data with is_buyer_maker from Binance directly."
                )
        else:
            is_aggressor_buy = np.ones(len(b), dtype=bool)  # worst-case: assume all buys
            if batch_idx == 0:
                print("  [D-08 WARNING] No side or is_buyer_maker column found. CVD set to all-buy.")

        amt    = b["amount"].to_numpy(dtype=np.float64)

        b["buy_vol"]   = np.where(is_aggressor_buy, amt, 0.0)
        b["sell_vol"]  = np.where(is_aggressor_buy, 0.0, amt)
        b["buy_cost"]  = np.where(is_aggressor_buy, b_cost, 0.0)
        b["sell_cost"] = np.where(is_aggressor_buy, 0.0, b_cost)
        b["is_buy"]    = is_aggressor_buy.astype(np.uint32)


        # Vectorized GroupBy for the batch
        agg = b.groupby("bar_id", sort=False).agg(
            open_price   = ("price", "first"),
            high_price   = ("price", "max"),
            low_price    = ("price", "min"),
            close_price  = ("price", "last"),
            start_ts     = ("ts", "first"),
            end_ts       = ("ts", "last"),
            buy_vol      = ("buy_vol", "sum"),
            sell_vol     = ("sell_vol", "sum"),
            buy_cost     = ("buy_cost", "sum"),
            sell_cost    = ("sell_cost", "sum"),
            total_cost   = ("cost", "sum"),
            trade_count  = ("amount", "size"),
            buy_count    = ("is_buy", "sum")
        )

        for bar_idx, row in agg.iterrows():
            if bar_idx not in running_stats:
                running_stats[bar_idx] = {
                    "open": row["open_price"],
                    "high": row["high_price"],
                    "low": row["low_price"],
                    "close": row["close_price"],
                    "start_ts": row["start_ts"],
                    "end_ts": row["end_ts"],
                    "buy_vol": 0.0, "sell_vol": 0.0,
                    "buy_cost": 0.0, "sell_cost": 0.0,
                    "total_cost": 0.0,
                    "trade_count": 0, "buy_count": 0
                }
            r = running_stats[bar_idx]
            
            # For bars that straddle boundary, max/min might be higher/lower
            r["high"]  = max(r["high"], row["high_price"])
            r["low"]   = min(r["low"], row["low_price"])
            r["close"] = row["close_price"] # latest trade wins
            r["end_ts"] = row["end_ts"]     # latest time wins
            
            # Cumulative values
            r["buy_vol"]     += row["buy_vol"]
            r["sell_vol"]    += row["sell_vol"]
            r["buy_cost"]    += row["buy_cost"]
            r["sell_cost"]   += row["sell_cost"]
            r["total_cost"]  += row["total_cost"]
            r["trade_count"] += int(row["trade_count"])
            r["buy_count"]   += int(row["buy_count"])

        total_rows += len(b)
        elapsed = time.time() - t0
        
        if elapsed > 0 and (time.time() - last_report > 15):
            pct = (batch_idx + 1) / n_batches * 100
            eta = elapsed * (n_batches / (batch_idx + 1) - 1)
            print(f"  [{pct:5.1f}%] batch {batch_idx+1:>6,}/{n_batches:,}  "
                  f"rows={total_rows/1e6:.1f}M  bars={len(running_stats):,}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
            last_report = time.time()

    print(f"\n  Total trades processed : {total_rows:,}")
    print(f"  Total dollar bars gen  : {len(running_stats):,}")
    print(f"  Processing time        : {time.time()-t0:.1f}s")

    # Drop the last bar if it hasn't reached the full volume yet.
    # The last bar ID is int(leftover_cusum // args.theta)
    last_bar_id = int(leftover_cusum // args.theta)
    if last_bar_id in running_stats:
        if running_stats[last_bar_id]["total_cost"] < args.theta * 0.99:
            print(f"  Dropping incomplete last bar (id={last_bar_id})")
            del running_stats[last_bar_id]

    df = pd.DataFrame.from_dict(running_stats, orient="index").sort_index()
    
    # Calculate derived features analogous to the 15m format
    df["sell_count"]      = df["trade_count"] - df["buy_count"]
    total_vol             = df["buy_vol"] + df["sell_vol"]
    df["cvd"]             = df["buy_vol"] - df["sell_vol"]
    df["aggressor_ratio"] = (df["buy_vol"] / total_vol.replace(0, np.nan)).fillna(0.5)
    df["vwap"]            = df["total_cost"] / total_vol.replace(0, np.nan)
    
    # Standardize output for Freqtrade / LGBM
    df["date"] = df["end_ts"].dt.tz_localize(None)  # naive datetime matching feather
    df = df.drop(columns=["end_ts"])
    
    # Reorder columns
    cols = ["date", "open", "high", "low", "close", "vwap", "buy_vol", "sell_vol", "cvd", 
            "aggressor_ratio", "trade_count", "buy_count", "sell_count", "total_cost"]
    df = df[cols]

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Feather requires a RangeIndex
    df_out = df.reset_index(drop=True)
    feather.write_feather(df_out, str(out_path))
    
    print(f"\n  [SUCCESS] Wrote {len(df_out):,} rows → {out_path}")

if __name__ == "__main__":
    main()
