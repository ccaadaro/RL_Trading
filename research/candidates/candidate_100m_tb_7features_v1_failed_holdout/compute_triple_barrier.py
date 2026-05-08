#!/usr/bin/env python3
"""
scripts/compute_triple_barrier.py

Implements Marcos López de Prado's Triple-Barrier Method and Symmetric CUSUM filter.

1. Calculates Exponential Volatility on Dollar Bars.
2. Applies Symmetric CUSUM to select 'events' (periods where price varied enough to trade).
3. Applies Triple Barrier on those events to determine if the first touch is:
   - Upper barrier (take profit) -> Label 1
   - Lower barrier (stop loss) -> Label -1
   - Vertical barrier (timeout) -> Label 0
"""

import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyarrow.feather as feather

def get_volatility(prices: pd.Series, span: int = 100) -> pd.Series:
    """
    Computes exponential moving standard deviation of log returns.
    This is the "micro" volatility used for CUSUM event selection.
    """
    log_ret = np.log(prices).diff()
    vol = log_ret.ewm(span=span).std()
    return vol


def get_daily_volatility(prices: pd.Series, window: int = 288) -> pd.Series:
    """
    Computes rolling daily volatility on dollar bars.
    window=288 ≈ 1 day at typical $2M BTC bar cadence.
    Used for barrier sizing — captures the *daily* price swing amplitude
    rather than per-bar micro noise.
    """
    if len(prices) < 100:
        # Fallback for very small datasets (e.g. $500M bars on short history)
        return pd.Series(0.02, index=prices.index)
        
    log_ret = np.log(prices).diff()
    effective_min_periods = min(window, 50)
    daily_vol = log_ret.rolling(window, min_periods=effective_min_periods).std() * np.sqrt(window)
    return daily_vol

def cusum_filter(prices: pd.Series, threshold: pd.Series) -> pd.Index:
    """
    Symmetric CUSUM Filter.
    Selects events where cumulative price difference exceeds the threshold.
    Returns integer indices of the events.
    """
    t_events = []
    s_pos = 0
    s_neg = 0
    
    diff = prices.diff()
    
    # We must iterate. Though slow, for ~1M rows it takes just seconds in standard numba/python.
    # We will use vectorized forms if possible, but standard iteration is robust here.
    diff_values = diff.values
    threshold_values = threshold.values
    
    for i in range(1, len(prices)):
        vol = threshold_values[i-1]
        if np.isnan(vol) or vol == 0:
            continue
            
        # Using log-prices or prices diff. We'll use log diff for stationary
        d = diff_values[i] / prices.values[i-1] # % diff
        
        s_pos = max(0, s_pos + d)
        s_neg = min(0, s_neg + d)
        
        if s_pos > vol:
            s_pos = 0
            t_events.append(i)
        elif s_neg < -vol:
            s_neg = 0
            t_events.append(i)
            
    return pd.Index(t_events)

def get_triple_barrier_labels(
    prices: pd.Series, 
    dates: pd.Series,
    events: pd.Index, 
    volatility: pd.Series,
    daily_volatility: pd.Series = None,
    pt_sl: list = [1.0, 1.0], 
    min_ret: float = 0.001,
    vertical_bars: int = 216,
    use_daily_vol: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Compute triple barriers for given events.
    
    When use_daily_vol=True (default), barriers are sized against daily_volatility
    (wider, captures real directional moves). When False, uses per-bar EWM volatility
    (original narrow behavior).
    
    pt_sl: [Profit Take multiplier, Stop Loss multiplier] against volatility
    vertical_bars: max holding period in bars (216 ≈ 9h at $2M BTC bars)
    """
    out = pd.DataFrame(index=events)
    out['start_time'] = dates.iloc[events].values
    out['start_price'] = prices.iloc[events].values
    out['volatility'] = volatility.iloc[events].values
    
    # Choose barrier sizing volatility
    if use_daily_vol and daily_volatility is not None:
        barrier_vol = daily_volatility.iloc[events].values
        # Fallback to micro vol if daily vol is NaN (early bars)
        micro_vol = volatility.iloc[events].values
        barrier_vol = np.where(np.isnan(barrier_vol), micro_vol * np.sqrt(288), barrier_vol)
        out['barrier_vol'] = barrier_vol
    else:
        barrier_vol = volatility.iloc[events].values
        out['barrier_vol'] = barrier_vol
    
    # Define barriers using the chosen volatility scale
    pt_threshold = pt_sl[0] * barrier_vol
    sl_threshold = pt_sl[1] * barrier_vol
    
    # Floor: separate minimum barrier widths to avoid noise and cover costs
    pt_floor = kwargs.get('pt_floor', min_ret)
    sl_floor = kwargs.get('sl_floor', min_ret)
    
    pt_threshold = np.maximum(pt_threshold, pt_floor)
    sl_threshold = np.maximum(sl_threshold, sl_floor)
    
    out['upper_b'] = out['start_price'] * (1 + pt_threshold)
    out['lower_b'] = out['start_price'] * (1 - sl_threshold)

    labels = []
    end_times = []
    
    # Note: Optimization could be done here with numpy arrays and argmax.
    # Since Dollar Bars are granular, we can slice forward
    price_array = prices.values
    date_array = dates.values
    
    t0 = time.time()
    for loc in events:
        limit_loc = min(loc + vertical_bars, len(price_array) - 1)
        window = price_array[loc:limit_loc + 1]
        
        # We need the values
        upper = out.at[loc, 'upper_b']
        lower = out.at[loc, 'lower_b']
        
        # Find first index where cross happens
        cross_upper = np.where(window >= upper)[0]
        cross_lower = np.where(window <= lower)[0]
        
        idx_upper = cross_upper[0] if len(cross_upper) > 0 else np.inf
        idx_lower = cross_lower[0] if len(cross_lower) > 0 else np.inf
        
        if idx_upper < idx_lower and idx_upper != np.inf:
            labels.append(1)
            end_times.append(date_array[loc + idx_upper])
        elif idx_lower < idx_upper and idx_lower != np.inf:
            labels.append(-1)
            end_times.append(date_array[loc + idx_lower])
        else:
            # Hit vertical barrier or end of array
            labels.append(0)
            end_times.append(date_array[limit_loc])
            
    out['end_time'] = end_times
    out['label'] = labels
    
    # Store diagnostics
    out['barrier_width_pt'] = pt_threshold
    out['barrier_width_sl'] = sl_threshold
    
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="cache/dollar_bars_btc_2000000.feather")
    ap.add_argument("--span", type=int, default=100, help="Volatility EWMA span (for CUSUM)")
    ap.add_argument("--daily-window", type=int, default=288,
                    help="Rolling window for daily vol estimation (288 ≈ 1 day)")
    ap.add_argument("--pt", type=float, default=1.5,
                    help="Profit Take multiplier × daily_vol")
    ap.add_argument("--sl", type=float, default=1.0,
                    help="Stop Loss multiplier × daily_vol")
    ap.add_argument("--v-bars", type=int, default=6,
                    help="Vertical Barrier in dollar bars (6 ≈ 1 day at $100M)")
    ap.add_argument("--min-ret", type=float, default=0.003,
                    help="Legacy minimum barrier width (replaced by floors if provided)")
    ap.add_argument("--pt-floor", type=float, default=0.0060,
                    help="Minimum PT width (60 bps)")
    ap.add_argument("--sl-floor", type=float, default=0.0040,
                    help="Minimum SL width (40 bps)")
    ap.add_argument("--use-micro-vol", action="store_true",
                    help="Use per-bar EWM volatility instead of daily vol (legacy mode)")
    args = ap.parse_args()

    print(f"Loading {args.data}...")
    df = feather.read_feather(args.data)
    
    MIN_LABELING = 100
    MIN_TRAINING = 1000

    if len(df) < MIN_LABELING:
        raise ValueError(f"CRITICAL: Not enough bars ({len(df)}) for labeling. Minimum: {MIN_LABELING}")
    
    if len(df) < MIN_TRAINING:
        print(f"  [WARNING] Only {len(df)} bars found. Below training threshold ({MIN_TRAINING}).")
        print(f"            Treating as DIAGNOSTIC ONLY resolution ($500M context).")
        
    print(f"Total bars: {len(df):,}")
    
    print("Computing EWM Volatility (for CUSUM)...")
    close_prices = df['close']
    dates = df['date']
    vol = get_volatility(close_prices, span=args.span)
    
    daily_vol = None
    if not args.use_micro_vol:
        print(f"Computing Daily Volatility (window={args.daily_window} bars)...")
        daily_vol = get_daily_volatility(close_prices, window=args.daily_window)
        dv_valid = daily_vol.dropna()
        if len(dv_valid) > 0:
            print(f"  Daily vol: mean={dv_valid.mean()*100:.3f}%  "
                  f"median={dv_valid.median()*100:.3f}%  "
                  f"p90={dv_valid.quantile(0.90)*100:.3f}%")
    
    print("Applying Symmetric CUSUM filter...")
    t0 = time.time()
    events = cusum_filter(close_prices, vol)
    print(f"CUSUM extracted {len(events):,} events ({(len(events)/len(df))*100:.1f}%) in {time.time()-t0:.1f}s")
    
    print(f"Applying Triple Barrier Labeling (pt={args.pt}, sl={args.sl}, v_bars={args.v_bars})...")
    t0 = time.time()
    out = get_triple_barrier_labels(
        prices=close_prices,
        dates=dates,
        events=events,
        volatility=vol,
        daily_volatility=daily_vol,
        pt_sl=[args.pt, args.sl],
        min_ret=args.min_ret,
        vertical_bars=args.v_bars,
        use_daily_vol=(not args.use_micro_vol),
        pt_floor=args.pt_floor,
        sl_floor=args.sl_floor,
    )
    print(f"Triple barrier labeling completed in {time.time()-t0:.1f}s")
    
    label_counts = out['label'].value_counts()
    print("\nTarget Label Distribution:")
    for lbl in [1, -1, 0]:
        count = label_counts.get(lbl, 0)
        print(f"  Label {lbl:>2}: {count:>8,} ({count/len(out)*100:4.1f}%)")
        
    out_file = str(Path(args.data).with_name(Path(args.data).stem + "_labeled.feather"))
    
    # Print barrier width diagnostics
    barrier_width = (out['upper_b'] - out['lower_b']) / out['start_price']
    print(f"\nBarrier Width Stats:")
    print(f"  Mean:   {barrier_width.mean()*100:.4f}%")
    print(f"  Median: {barrier_width.median()*100:.4f}%")
    print(f"  p25:    {barrier_width.quantile(0.25)*100:.4f}%")
    print(f"  p75:    {barrier_width.quantile(0.75)*100:.4f}%")
    
    # We save only the labeled events
    out_df = out.reset_index(drop=True)
    feather.write_feather(out_df, out_file)
    print(f"\nSaved labeled events to {out_file}")

if __name__ == "__main__":
    main()
