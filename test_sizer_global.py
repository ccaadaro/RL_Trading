import pandas as pd
import numpy as np
from utils.position_sizer import FractionalKellySizer

# Load data
df = pd.read_feather('cache/dollar_bars_btc_2000000_sizing.feather')

# Precompute adaptive_threshold
turb = df["turbulence_score"].fillna(0.0)
adaptive_thr = turb.expanding(100).quantile(0.95).fillna(5.0)

# Initialize sizer 
sizer = FractionalKellySizer(kelly_fraction=0.5, max_drawdown=0.10)

# Run sizes globally
targets = sizer.size_portfolio(
    probabilities=df["oof_pred"],
    regimes=df["hmm_semantic_regime"],
    turbulence=turb,
    adaptive_threshold=adaptive_thr,
    risk_scales=df["volatility_24_feature"],
    dof=3
)

print(f"Non-zero targets: {(targets > 0).sum()}")
print(f"Targets passing 0.10 threshold: {(targets >= 0.10).sum()}")
