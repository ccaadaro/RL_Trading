#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    feat_path = "cache/dollar_bars_btc_100000000_minimal_features.feather"
    if not Path(feat_path).exists():
        print(f"Error: {feat_path} not found.")
        return

    print(f"Loading features from {feat_path}...")
    df = pd.read_feather(feat_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Define periods
    df_24 = df[(df['date'] >= '2024-01-01') & (df['date'] < '2025-01-01')].copy()
    df_25 = df[df['date'] >= '2025-01-01'].copy()
    
    features = [
        "return_3_bars_feature", "return_5_bars_feature", "vol_10_feature",
        "cvd_slope_feature", "aggressor_imbalance_feature", "hma_dist_feature",
        "wvf_zscore_feature"
    ]
    
    print(f"Analysis: 2024 (N={len(df_24)}) vs 2025 (N={len(df_25)})")
    
    # 1. Feature Distributions
    stats = []
    for f in features:
        m24, s24 = df_24[f].mean(), df_24[f].std()
        m25, s25 = df_25[f].mean(), df_25[f].std()
        stats.append({
            "feature": f,
            "mean_24": m24, "std_24": s24,
            "mean_25": m25, "std_25": s25,
            "drift_z": (m25 - m24) / (s24 if s24 > 0 else 1e-9)
        })
    
    df_stats = pd.DataFrame(stats)
    print("\n--- FEATURE DRIFT ANALYSIS ---")
    print(df_stats[["feature", "mean_24", "mean_25", "drift_z"]].to_string(index=False))
    
    # 2. Target Properties
    def analyze_labels(df_p, name):
        counts = df_p['label'].value_counts(normalize=True).sort_index()
        return {
            "name": name,
            "long_ratio": counts.get(1, 0),
            "short_ratio": counts.get(-1, 0),
            "neutral_ratio": counts.get(0, 0)
        }
    
    label_stats = [analyze_labels(df_24, "2024"), analyze_labels(df_25, "2025")]
    print("\n--- LABEL DISTRIBUTION ---")
    print(pd.DataFrame(label_stats).to_string(index=False))
    
    # 3. Correlation to Target
    corr_stats = []
    for f in features:
        c24 = df_24[f].corr((df_24['label'] == 1).astype(int))
        c25 = df_25[f].corr((df_25['label'] == 1).astype(int))
        corr_stats.append({
            "feature": f,
            "corr_24": c24,
            "corr_25": c25,
            "corr_decay": c25 / (c24 if abs(c24) > 1e-6 else 1e-9)
        })
    
    print("\n--- FEATURE CORRELATION TO TARGET (LONG) ---")
    print(pd.DataFrame(corr_stats)[["feature", "corr_24", "corr_25", "corr_decay"]].to_string(index=False))

    # 4. Volatility Regime
    print("\n--- VOLATILITY REGIME ---")
    print(f"2024 Median Realized Vol (10 bars): {df_24['vol_10_feature'].median():.4f}")
    print(f"2025 Median Realized Vol (10 bars): {df_25['vol_10_feature'].median():.4f}")

    # Output to file
    out_path = "research/candidates/candidate_100m_tb_7features_v1_failed_holdout/regime_drift_analysis.csv"
    df_stats.to_csv(out_path, index=False)
    print(f"\nAnalysis saved to {out_path}")

if __name__ == "__main__":
    main()
