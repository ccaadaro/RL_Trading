#!/usr/bin/env python3
"""
scripts/deploy_execution_sim.py

Simulates the Execution Router over the portfolio target sizes.
Proves that large allocations do not teleport instantly into the market,
but are sliced gracefully via the Participation Rate limits.
"""

import sys
import argparse
import time
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow.feather as feather

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from utils.execution_algos import DeterministicPOVRouter


def infer_q_lob_safe(row: pd.Series, direction: float,
                     router: DeterministicPOVRouter,
                     liquidity_haircut: float = 0.25) -> float:
    """
    Conservative liquidity proxy.
    Prefer explicit L2 depth when present; otherwise fall back to a haircut
    of observed opposite-side flow or bar notional.
    """
    if direction > 0:
        if "safe_buy_depth_usdt" in row and pd.notna(row["safe_buy_depth_usdt"]):
            return max(float(row["safe_buy_depth_usdt"]), 0.0)
        if {"sell_vol", "close"}.issubset(row.index):
            return max(float(row["sell_vol"]) * float(row["close"]) * liquidity_haircut, 0.0)
    elif direction < 0:
        if "safe_sell_depth_usdt" in row and pd.notna(row["safe_sell_depth_usdt"]):
            return max(float(row["safe_sell_depth_usdt"]), 0.0)
        if {"buy_vol", "close"}.issubset(row.index):
            return max(float(row["buy_vol"]) * float(row["close"]) * liquidity_haircut, 0.0)

    if "notional" in row and pd.notna(row["notional"]):
        return max(float(row["notional"]) * liquidity_haircut, 0.0)
    return router.V_bar * 0.05

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="cache/dollar_bars_btc_2000000_sizing.feather")
    ap.add_argument("--account", type=float, default=1_000_000.0, help="Simulated account size in USD")
    args = ap.parse_args()

    print(f"Loading {args.data}...")
    df = feather.read_feather(args.data)
    
    # We will simulate the timeline
    account_size = args.account
    router = DeterministicPOVRouter(V_bar=2_000_000.0)
    
    current_notional = 0.0
    active_schedule = None  # { 'Q_rem': float, 'N_rem': int, 'dir': int }
    
    actual_positions = np.zeros(len(df))
    execution_volumes = np.zeros(len(df))
    q_lob_safe_series = np.zeros(len(df))
    
    print("\n[Layer 5] Executing Order Router Simulation...")
    t0 = time.time()
    
    # We convert dataframe to dict/records for fast iteration or use arrays
    target_sizes = df["target_position_size"].values
    regimes = df["hmm_semantic_regime"].values
    if {"expected_net_return", "risk_scale"}.issubset(df.columns):
        urgencies = (
            np.abs(df["expected_net_return"].values) /
            np.maximum(df["risk_scale"].values, 1e-6)
        )
    else:
        urgencies = np.abs(df["oof_pred"].fillna(0.5).values - 0.5) * 2.0
    
    for i in range(len(df)):
        target_notional = target_sizes[i] * account_size
        regime = regimes[i]
        urgency = abs(urgencies[i])
        
        # 1. Assess if we need to update our macroscopic schedule
        deficit = target_notional - current_notional
        
        # Only open a new schedule if the deficit is material (e.g., > 1% of account) -> $10,000
        if abs(deficit) > (0.01 * account_size):
            direction = np.sign(deficit)
            if active_schedule and active_schedule["dir"] == direction:
                active_schedule["Q_rem"] = abs(deficit)
            else:
                # We schedule over 4 Dollar Bars (default horizon)
                active_schedule = {
                    'Q_rem': abs(deficit),
                    'N_rem': router.default_horizon_bars,
                    'dir': direction
                }
        
        executed_this_bar = 0.0
        
        # 2. Process active schedule (Simulating the Intrabar micro-fill)
        if active_schedule and active_schedule['Q_rem'] > 0 and active_schedule['N_rem'] > 0:
            row = df.iloc[i]  # BUG FIX: 'row' was referenced but never defined in this scope
            q_lob_safe = infer_q_lob_safe(row, active_schedule["dir"], router)
            q_b = router.slice_order(
                Q_remaining=active_schedule['Q_rem'],
                N_remaining=active_schedule['N_rem'],
                regime=regime,
                urgency=urgency,
                q_lob_safe=q_lob_safe,
            )
            
            # Update state
            executed_this_bar = q_b * active_schedule['dir']
            current_notional += executed_this_bar
            q_lob_safe_series[i] = q_lob_safe
            
            active_schedule['Q_rem'] -= q_b
            active_schedule['N_rem'] -= 1
            
            # Clean up finished schedules
            if active_schedule['Q_rem'] <= 1.0 or active_schedule['N_rem'] <= 0:
                active_schedule = None
                
        actual_positions[i] = current_notional / account_size
        execution_volumes[i] = executed_this_bar
        
    df["actual_position_size"] = actual_positions
    df["execution_volume_usd"] = execution_volumes
    df["q_lob_safe_usd"] = q_lob_safe_series
    
    print(f"Simulation completed across {len(df):,} events in {time.time()-t0:.1f}s")
    
    # Statistics
    blocks_executed = (execution_volumes != 0.0).sum()
    print(f"Total Execution Blocks fired: {blocks_executed:,} ({blocks_executed/len(df)*100:.1f}%)")
    if blocks_executed > 0:
        print(f"Average q_lob_safe used: {df.loc[df['q_lob_safe_usd'] > 0, 'q_lob_safe_usd'].mean():,.0f} USD")
    
    # Measure typical slippage via delays: How often did the Actual Position lag the Target Position?
    lags = np.abs(df["target_position_size"] - df["actual_position_size"])
    mean_lag = lags.mean()
    print(f"Average Execution Schedule Lag: {mean_lag*100:.2f}% of Account")
    
    out_file = str(Path(args.data).with_name("dollar_bars_btc_2000000_execution.feather"))
    feather.write_feather(df, out_file)
    print(f"\nFinal Execution Timeline saved to {out_file}")

if __name__ == "__main__":
    main()
