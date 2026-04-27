#!/usr/bin/env python3
"""
scripts/deploy_hedge_sim.py

Simulates the Delta-Neutral Hedge Manager. 
Overlays aggressive Market Order shorts to artificially kill Spot exposition 
safeguarding edge-down systemic risks.
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

from utils.hedge_manager import DynamicHedger

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="cache/dollar_bars_btc_2000000_execution.feather")
    args = ap.parse_args()

    print(f"Loading executed timeline {args.data}...")
    df = feather.read_feather(args.data)
    
    print("\n[Layer 6] Sizing Tactical Hedge (Delta Neutralization)...")
    t0 = time.time()
    
    # Instantiate Hedger with dynamic scaling
    hedger = DynamicHedger(critical_turbulence=9.48, high_stress_turbulence=5.0)
    
    hedge_sizes = np.zeros(len(df))
    
    # We use numpy arrays for extremely fast temporal iteration
    spot_invs = df["actual_position_size"].values
    target_invs = df["target_position_size"].values
    regimes = df["hmm_semantic_regime"].values
    turbulences = df["turbulence_score"].values
    
    for i in range(len(df)):
        h = hedger.calculate_hedge(
            spot_inventory=spot_invs[i],
            target_inventory=target_invs[i],
            regime=regimes[i],
            turbulence=turbulences[i]
        )
        # Assuming Market Maker executes hedge instantly
        hedge_sizes[i] = h
        
    df["hedge_position_size"] = hedge_sizes
    df["net_delta_exposure"] = df["actual_position_size"] + df["hedge_position_size"]
    
    print(f"Hedge overlay computed in {time.time()-t0:.1f}s")
    
    # Metrics
    activations = (df["hedge_position_size"] != 0.0).sum()
    print(f"\nHedge Activations: {activations:,} ticks ({(activations/len(df))*100:.2f}% of timeline)")
    
    # Calculate Risk Reductions
    # When hedge activated, what was the lag vs the net delta?
    mask = df["hedge_position_size"] != 0.0
    if mask.sum() > 0:
        avg_exposed_lag = np.abs(df.loc[mask, "actual_position_size"] - df.loc[mask, "target_position_size"]).mean()
        avg_net_lag = np.abs(df.loc[mask, "net_delta_exposure"] - df.loc[mask, "target_position_size"]).mean()
        
        print("\nDuring Crisis Environments:")
        print(f"-> Unhedged Average Delta Lag : {avg_exposed_lag*100:.2f}%")
        print(f"-> Hedged Average Delta Lag   : {avg_net_lag*100:.2f}% (Reduction in Market Risk!)")
        
    out_file = str(Path(args.data).with_name("dollar_bars_btc_2000000_hedged.feather"))
    feather.write_feather(df, out_file)
    print(f"\nFinal Hedged Portfolio saved to {out_file}")

if __name__ == "__main__":
    main()
