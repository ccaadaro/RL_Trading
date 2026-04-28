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
from utils.risk_directors import MahalanobisTurbulence, HMMRegimeModel

def run_replay(model_path: str, data_path: str, meta_model_path: Optional[str] = None):
    print(f"Loading data from {data_path}...")
    df = pd.read_feather(data_path)
    
    print(f"Loading Alpha model from {model_path}...")
    alpha_model = lgb.Booster(model_file=model_path)
    
    meta_model = None
    if meta_model_path:
        print(f"Loading Meta-model from {meta_model_path}...")
        meta_model = lgb.Booster(model_file=meta_model_path)

    # Auto-detect feature set
    n_features = len(alpha_model.feature_name())
    if n_features == 21:
        features = SIGNAL_FEAT_COLS_V2
        model_name = "Elite v2.1"
    elif n_features == 14:
        features = FEATURE_SET_INSTITUTIONAL
        model_name = "Institutional Base"
    else:
        print(f"Warning: Model has {n_features} features. Using generic detection.")
        features = [c for c in df.columns if c.endswith("_feature") and c in alpha_model.feature_name()]
        model_name = "Unknown Architecture"
    
    print(f"Executing replay for {model_name} on {len(df)} samples...")
    
    # Generate Alpha Predictions
    alpha_preds = alpha_model.predict(df[features])
    df["alpha_signal"] = alpha_preds
    
    # Smooth Alpha
    df["alpha_prob_smooth"] = pd.Series(alpha_preds).ewm(span=10).mean().values
    df["alpha_prob_zscore"] = (pd.Series(alpha_preds) - pd.Series(alpha_preds).rolling(200).mean()) / pd.Series(alpha_preds).rolling(200).std()
    df["alpha_prob_percentile"] = pd.Series(alpha_preds).rolling(1000).rank(pct=True)
    df["alpha_signal_persistence"] = (pd.Series(alpha_preds) > 0.55).astype(int).rolling(20).sum()

    # Dynamic Threshold
    dynamic_threshold = pd.Series(alpha_preds).rolling(window=1000, min_periods=100).quantile(0.90).fillna(0.60).values

    # Market Context (HMM & Turbulence)
    print("Computing Context Features...")
    turb_engine = MahalanobisTurbulence(window=1000)
    hmm_engine = HMMRegimeModel(n_components=3)
    
    risk_feats = ["log_return_feature", "volatility_24_feature", "intraday_range_feature"]
    available_risk = [c for c in risk_feats if c in df.columns]
    df["turbulence_score"] = turb_engine.compute(df, available_risk).fillna(method='bfill').fillna(5.0)
    df["turbulence_percentile"] = df["turbulence_score"].rolling(2000).rank(pct=True)

    hmm_feats = ["log_return_feature", "volatility_24_feature", "aggressor_ratio", "intraday_range_feature"]
    available_hmm = [c for c in hmm_feats if c in df.columns]
    canonical_states, _ = hmm_engine.fit_predict(df, available_hmm)
    df["hmm_state"] = canonical_states

    # Meta-Inference (Pre-calculate if possible to avoid loop overhead)
    df["meta_pass"] = 1.0
    if meta_model:
        print("Running Meta-Gatekeeper inference...")
        df["spread_bps"] = df["pr_spread_feature"] * 10000
        df["expected_cost_bps"] = 5 + df["spread_bps"] + 2
        
        meta_feats_df = pd.DataFrame({
            "alpha_prob":              df["alpha_signal"], # Raw probability
            "alpha_prob_smooth":       df["alpha_prob_smooth"],
            "alpha_prob_zscore":       df["alpha_prob_zscore"],
            "alpha_prob_percentile":   df["alpha_prob_percentile"],
            "alpha_signal_persistence": df["alpha_signal_persistence"],
            "turbulence_score":        df["turbulence_score"],
            "turbulence_percentile":   df["turbulence_percentile"],
            "hmm_state":               df["hmm_state"].astype(int),
            "volatility_24_feature":   df["volatility_24_feature"],
            "aggressor_ratio":         df["aggressor_ratio"] if "aggressor_ratio" in df.columns else 0.0,
            "l2_imbalance_feature":    df["l2_imbalance_feature"] if "l2_imbalance_feature" in df.columns else 0.0,
            "spread_bps":              df["spread_bps"],
            "expected_cost_bps":       df["expected_cost_bps"]
        }).fillna(0)
        
        meta_probs = meta_model.predict(meta_feats_df)
        df["meta_prob"] = meta_probs
        # Institutional Selective Gatekeeper
        df["meta_pass"] = (df["meta_prob"] >= 0.65).astype(int)
    else:
        df["meta_pass"] = 1.0
    
    # Simulation Loop
    n = len(df)
    target_pos = np.zeros(n)
    curr_t = 0.0
    
    min_entry = 0.65     # High conviction entry
    exit_threshold = 0.50 # Neutral exit
    min_hold_bars = 150   # Institutional hold (~9 hours)
    bars_since_entry = 0
    
    print(f"Executing replay for {Path(model_path).name} on {n} samples...")
    print(f"Parameters: MinEntry={min_entry}, ExitThresh={exit_threshold}, MinHold={min_hold_bars}")
    
    for i in range(n):
        current_pred = df["alpha_prob_smooth"].iloc[i]
        meta_ok = df["meta_pass"].iloc[i]
        entry_threshold_t = max(min_entry, dynamic_threshold[i])
        
        if curr_t == 0.0:
            # Entry requires Alpha AND Meta agreement
            if current_pred > entry_threshold_t and meta_ok:
                curr_t = 1.0
                bars_since_entry = 0
        else:
            bars_since_entry += 1
            # Exit logic
            if current_pred < exit_threshold and bars_since_entry >= min_hold_bars:
                curr_t = 0.0
        
        target_pos[i] = curr_t

    # Rest of calculation (unchanged logic)
    actual_pos = np.zeros(n)
    current_pos = 0.0
    rebal_threshold = 0.1
    fee_rate = 0.0005  # 0.05%
    slippage = 0.0002  # 0.02%
    total_cost_rate = fee_rate + slippage
    
    for i in range(n):
        diff = abs(target_pos[i] - current_pos)
        if diff > rebal_threshold:
            current_pos = target_pos[i]
        actual_pos[i] = current_pos
        
    df["position"] = actual_pos
    df["bar_return"] = df["close"].pct_change().fillna(0)
    trade_diff = np.abs(np.diff(actual_pos, prepend=0))
    df["costs"] = trade_diff * total_cost_rate
    df["strategy_return"] = (df["position"].shift(1).fillna(0) * df["bar_return"]) - df["costs"]
    
    df["cum_return"] = (1 + df["strategy_return"]).cumprod()
    df["cum_market"] = (1 + df["bar_return"]).cumprod()
    
    total_ret = df["cum_return"].iloc[-1] - 1
    market_ret = df["cum_market"].iloc[-1] - 1
    sharpe = (df["strategy_return"].mean() / df["strategy_return"].std()) * np.sqrt(365 * 24) if df["strategy_return"].std() > 0 else 0
    
    # Correct turnover calculation using actual time elapsed
    if 'date' in df.columns:
        total_days = (df['date'].max() - df['date'].min()).total_seconds() / (24 * 3600)
    else:
        total_days = n / 17.6 / 24
    
    total_months = total_days / 30.44
    monthly_turnover = trade_diff.sum() / total_months
    
    print("\n=== Global Alpha Replay Results ===")
    print(f"Meta-Model Active:           {'YES' if meta_model else 'NO'}")
    print(f"Total Strategy Return (Net): {total_ret:.2%}")
    print(f"Total Market Return:         {market_ret:.2%}")
    print(f"Annualized Sharpe (Net):     {sharpe:.2f}")
    print(f"Max Drawdown:                {(1 - df['cum_return'] / df['cum_return'].cummax()).max():.2%}")
    print(f"Monthly Turnover (Est):      {monthly_turnover:.2f}x")
    
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
    parser.add_argument("--model", type=str, default="models/dollar_alpha_v1/model_a_institutional.txt")
    parser.add_argument("--meta", type=str, default="models/meta_model_v1/gatekeeper.txt")
    args = parser.parse_args()
    
    data_path = "cache/dollar_bars_btc_2000000_features.feather"
    
    # Check meta model path
    meta_path = args.meta if os.path.exists(args.meta) else None
    
    if os.path.exists(args.model) and os.path.exists(data_path):
        run_replay(args.model, data_path, meta_model_path=meta_path)
    else:
        print(f"Error: Required files not found.\nModel: {args.model}\nData: {data_path}")
