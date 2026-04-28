import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import sys
import os
from typing import Optional

# Add strategy root to path
_HERE = Path(__file__).resolve().parent.parent
sys.path.append(str(_HERE))

from utils.signal_features import SIGNAL_FEAT_COLS_V2, FEATURE_SET_INSTITUTIONAL

def run_replay(model_path: str, data_path: str, threshold: float = 0.01):
    print(f"Loading data from {data_path}...")
    df = pd.read_feather(data_path)
    
    print(f"Loading model from {model_path}...")
    model = lgb.Booster(model_file=model_path)
    
    # Auto-detect feature set
    n_features = len(model.feature_name())
    if n_features == 21:
        features = SIGNAL_FEAT_COLS_V2
        model_name = "Elite v2.1"
    elif n_features == 14:
        features = FEATURE_SET_INSTITUTIONAL
        model_name = "Institutional Base"
    else:
        print(f"Warning: Model has {n_features} features. Using generic detection.")
        features = [c for c in df.columns if c.endswith("_feature") and c in model.feature_name()]
        model_name = "Unknown Architecture"
    
    print(f"Executing replay for {model_name} on {len(df)} samples...")
    
    # Generate predictions
    preds = model.predict(df[features])
    df["alpha_signal"] = preds
    df["binary_target"] = (df["label"] == 1).astype(int)
    
    # 1. Signal Smoothing (EMA)
    pred_series = pd.Series(preds)
    pred_smooth = pred_series.ewm(span=10).mean().values
    
    # 2. Dynamic Percentile Threshold
    # We use a 1000-bar rolling window to find the 90th percentile of smoothed predictions
    dynamic_threshold = pred_series.rolling(window=1000, min_periods=100).quantile(0.90).fillna(0.60).values
    
    # 3. Advanced Hysteresis & Min Hold Logic
    n = len(df)
    target_pos = np.zeros(n)
    curr_t = 0.0
    
    # Hardcoded parameters based on the matrix idea
    # We lower the hard floor to 0.52 to allow the dynamic percentile to dictate entry, 
    # ensuring even lower-range models (like Model A) get tested on their highest-conviction trades.
    min_entry = 0.52
    exit_threshold = 0.50
    min_hold_bars = 10
    
    bars_since_entry = 0
    
    for i in range(n):
        current_pred = pred_smooth[i]
        # Entry requires being in the 90th percentile AND above a minimum probability floor
        entry_threshold_t = max(min_entry, dynamic_threshold[i])
        
        if curr_t == 0.0:
            if current_pred > entry_threshold_t:
                curr_t = 1.0
                bars_since_entry = 0
        else:
            bars_since_entry += 1
            # Block exit if minimum hold period hasn't passed, unless it's a catastrophic drop
            if current_pred < exit_threshold:
                if bars_since_entry >= min_hold_bars or current_pred < (exit_threshold - 0.05):
                    curr_t = 0.0
        
        target_pos[i] = curr_t

    actual_pos = np.zeros(n)
    current_pos = 0.0
    rebal_threshold = 0.1
    fee_rate = 0.0005  # 0.05%
    slippage = 0.0002  # 0.02%
    total_cost_rate = fee_rate + slippage
    
    total_costs = 0.0
    for i in range(n):
        diff = abs(target_pos[i] - current_pos)
        if diff > rebal_threshold:
            total_costs += diff * total_cost_rate
            current_pos = target_pos[i]
        actual_pos[i] = current_pos
        
    df["position"] = actual_pos
    # Return proxy (normalized per bar)
    df["bar_return"] = df["close"].pct_change().fillna(0)
    # Strategy return = position * market_return - costs (applied on change)
    trade_diff = np.abs(np.diff(actual_pos, prepend=0))
    df["costs"] = trade_diff * total_cost_rate
    df["strategy_return"] = (df["position"].shift(1).fillna(0) * df["bar_return"]) - df["costs"]
    
    # Cumulative stats
    df["cum_return"] = (1 + df["strategy_return"]).cumprod()
    df["cum_market"] = (1 + df["bar_return"]).cumprod()
    
    total_ret = df["cum_return"].iloc[-1] - 1
    market_ret = df["cum_market"].iloc[-1] - 1
    sharpe = (df["strategy_return"].mean() / df["strategy_return"].std()) * np.sqrt(365 * 24) if df["strategy_return"].std() > 0 else 0
    
    # Profit concentration check
    trade_returns = df[df["costs"] > 0]["strategy_return"]
    top_3_pct = trade_returns.nlargest(3).sum() / trade_returns.sum() if trade_returns.sum() > 0 else 0

    print("\n=== Global Alpha Replay Results ===")
    print(f"Total Strategy Return (Net): {total_ret:.2%}")
    print(f"Total Market Return:         {market_ret:.2%}")
    print(f"Annualized Sharpe (Net):     {sharpe:.2f}")
    print(f"Max Drawdown:                {(1 - df['cum_return'] / df['cum_return'].cummax()).max():.2%}")
    print(f"Total Turnover:              {trade_diff.sum():.2f}")
    print(f"Monthly Turnover (Est):      {trade_diff.sum() / (n / (30*24)):.2f}x")
    print(f"Profit Concentration (Top 3): {top_3_pct:.2%}")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 7))
    
    # Cumulative returns
    plt.plot(df["cum_market"], label="Market (BTC)", alpha=0.4, color="gray", linestyle="--")
    plt.plot(df["cum_return"], label="Strategy (Net)", color="blue", linewidth=2)
    
    # Drawdown shading
    cum_max = df["cum_return"].cummax()
    drawdown = (df["cum_return"] / cum_max) - 1
    plt.fill_between(df.index, df["cum_return"], cum_max, color="red", alpha=0.2, label="Drawdown")
    
    plt.title(f"Global Alpha Replay: {Path(model_path).name}\nNet ROI: {total_ret:.2%} | Max DD: {drawdown.min():.2%}")
    plt.xlabel("Dollar Bars")
    plt.ylabel("Cumulative Return")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.2)
    
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    plot_file = reports_dir / f"equity_{Path(model_path).stem}.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Equity curve with Drawdown shading saved to {plot_file}")
    
    output_path = f"cache/replay_{Path(model_path).stem}.feather"
    df[["alpha_signal", "position", "strategy_return", "costs", "cum_return", "cum_market"]].reset_index().to_feather(output_path)
    print(f"Replay results saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/dollar_alpha_v1/latest_model.txt")
    parser.add_argument("--threshold", type=float, default=0.08)
    args = parser.parse_args()
    
    data_path = "cache/dollar_bars_btc_2000000_features.feather"
    if os.path.exists(args.model) and os.path.exists(data_path):
        run_replay(args.model, data_path, threshold=args.threshold)
    else:
        print(f"Error: Required files not found.\nModel: {args.model}\nData: {data_path}")
