import pandas as pd
import numpy as np

df = pd.read_feather("cache/metamodel_training_data.feather")
print("=== Meta-model Training Data Analysis ===")
print(f"Total samples: {len(df)}")
print(f"y_meta distribution:\n{df['y_meta'].value_counts(normalize=True)}")
print(f"forward_ret_bps stats:\n{df['forward_ret_bps'].describe()}")

# Percentiles of forward_ret_bps
print(f"forward_ret_bps 90th: {df['forward_ret_bps'].quantile(0.90):.2f}")
print(f"forward_ret_bps 95th: {df['forward_ret_bps'].quantile(0.95):.2f}")

# Check correlations
features = [
    "alpha_prob", "alpha_prob_smooth", "alpha_prob_zscore", "alpha_prob_percentile",
    "alpha_signal_persistence", "turbulence_score", "hmm_state", "expected_cost_bps"
]
print("\nCorrelation with forward_ret_bps:")
print(df[features + ["forward_ret_bps"]].corr()["forward_ret_bps"].sort_values(ascending=False))
