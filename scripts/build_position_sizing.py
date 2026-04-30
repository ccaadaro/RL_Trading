#!/usr/bin/env python3
"""
scripts/build_position_sizing.py

Integrates the predictions and regime signals to output final capital allocation weights.
Produces the fundamental target size that execution engines will use for routing trading orders.
"""

import sys
import argparse
import time
from pathlib import Path

import pandas as pd
import pyarrow.feather as feather

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from utils.position_sizer import FractionalKellySizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="cache/dollar_bars_btc_2000000_regimes.feather")
    ap.add_argument("--kelly", type=float, default=0.5, help="Kelly fraction multiplier (e.g., 0.5 for Half-Kelly)")
    ap.add_argument("--fee-rate", type=float, default=5e-4)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    args = ap.parse_args()

    print(f"Loading {args.data}...")
    df = feather.read_feather(args.data)
    
    if "alpha_prob" not in df.columns or "hmm_semantic_regime" not in df.columns:
        raise ValueError("Dataframe must contain 'alpha_prob', 'turbulence_score', and 'hmm_semantic_regime'")
        
    print(f"Total Rows: {len(df):,}")
    
    print("\n[Layer 4] Executing Portfolio Allocation (Fractional Kelly + CVaR proxy)...")
    t0 = time.time()
    
    allocator = FractionalKellySizer(
        kelly_fraction=args.kelly,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
    )

    probs = df["alpha_prob"].fillna(0.5).clip(0.0, 1.0)
    risk_scale_col = next(
        (c for c in ["volatility", "volatility_24_feature", "realized_vol_1h_feature"] if c in df.columns),
        None,
    )
    risk_scales = (
        df[risk_scale_col].astype(float).clip(lower=allocator.min_risk_scale)
        if risk_scale_col is not None
        else pd.Series(allocator.min_risk_scale, index=df.index)
    )
    expected_net = allocator.estimate_expected_net_return(probs, risk_scales)
    df["signal_edge"] = 2.0 * probs - 1.0
    df["risk_scale"] = risk_scales
    df["expected_net_return"] = expected_net
    
    target_sizes = allocator.size_portfolio(
        probabilities=None,
        regimes=df["hmm_semantic_regime"],
        turbulence=df["turbulence_score"]
        ,
        expected_net_returns=expected_net,
        risk_scales=risk_scales,
    )
    
    df["target_position_size"] = target_sizes
    
    print(f"Position sizes computed in {time.time()-t0:.1f}s")
    if risk_scale_col:
        print(f"Risk scale column: {risk_scale_col}")
    print(f"Expected net return: mean={df['expected_net_return'].mean():.5f}  "
          f"std={df['expected_net_return'].std():.5f}")
    
    print("\nTarget Allocation Distribution Statistics:")
    print("------------------------------------------")
    # Absolute exposure stats
    abs_sizes = df["target_position_size"].abs()
    print(f"Zero Exposure (Kill-Switch Active) : {(abs_sizes == 0.0).mean()*100:.2f}%")
    print(f"Micro Exposure (0% - 5%)           : {((abs_sizes > 0.0) & (abs_sizes <= 0.05)).mean()*100:.2f}%")
    print(f"Mild Exposure (5% - 15%)           : {((abs_sizes > 0.05) & (abs_sizes <= 0.15)).mean()*100:.2f}%")
    print(f"Max Exposure (15% - 25%)           : {(abs_sizes > 0.15).mean()*100:.2f}%")
    
    # Save the sized dataset
    out_file = str(Path(args.data).with_name("dollar_bars_btc_2000000_sizing.feather"))
    feather.write_feather(df, out_file)
    print(f"\nSaved {len(df):,} rows with allocated position sizes to {out_file}")

if __name__ == "__main__":
    main()
