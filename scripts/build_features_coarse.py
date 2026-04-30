#!/usr/bin/env python3
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
    ap.add_argument("--data", type=str, required=True, help="Path to bars feather (OHLCV + microstructure)")
    ap.add_argument("--labels", type=str, required=True, help="Path to labels feather (Triple Barrier)")
    ap.add_argument("--bar-type", type=str, choices=["dollar", "time"], default="dollar")
    args = ap.parse_args()

    print(f"Loading bars from {args.data}...")
    df = feather.read_feather(args.data)
    
    # Ensure volume and buy/sell renames for build_feature_matrix
    if 'volume' not in df.columns:
        if 'buy_vol' in df.columns and 'sell_vol' in df.columns:
            df['volume'] = df['buy_vol'] + df['sell_vol']
        elif 'amount' in df.columns:
            df['volume'] = df['amount']
    
    renames = {}
    if "buy_vol" in df.columns: renames["buy_vol"] = "buy_volume"
    if "sell_vol" in df.columns: renames["sell_vol"] = "sell_volume"
    if renames:
        df = df.rename(columns=renames)

    original_cols = set(df.columns)
    
    print(f"Computing {args.bar_type} feature matrix...")
    t0 = time.time()
    
    X = build_feature_matrix(df, eth_df=None, funding_series=None)
    
    for col in X.columns:
        df[col] = X[col].values
    
    new_cols = [c for c in df.columns if c not in original_cols and c.endswith('_feature')]
    print(f"Computed {len(new_cols)} features in {time.time()-t0:.1f}s")
    
    df = df.dropna(subset=new_cols)
    print(f"Dropped warmup rows, remaining bars: {len(df):,}")
    
    print(f"\nLoading Labeled Events from {args.labels}...")
    df_labels = feather.read_feather(args.labels)
    
    print("Mapping features to Labeled Events...")
    # Join on 'date'/'start_time'
    df_final = pd.merge(df_labels, df, left_on='start_time', right_on='date', how='inner')
    
    stem = Path(args.data).stem
    out_file = str(Path(args.data).with_name(f"{stem}_features.feather"))
    
    df_out = df_final.reset_index(drop=True)
    feather.write_feather(df_out, out_file)
    print(f"Saved {len(df_out):,} labeled-featured rows to {out_file}")

if __name__ == "__main__":
    main()
