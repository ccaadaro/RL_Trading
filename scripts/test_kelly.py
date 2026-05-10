import pandas as pd
from utils.position_sizer import FractionalKellySizer

# Initialize sizer with defaults used in fast_global_replay
sizer = FractionalKellySizer(kelly_fraction=0.5, max_drawdown=0.10, min_risk_scale=0.001)

# Sample a row with pred > 0.65
df = pd.read_feather('cache/dollar_bars_btc_2000000_sizing.feather')
df_high = df[df['alpha_prob'] > 0.65].iloc[0]

pred = pd.Series([df_high['alpha_prob']])
reg = pd.Series(['bull_calm'])
turb = pd.Series([df_high['turbulence_score']])
athr = pd.Series([5.0])
risk = pd.Series([df_high['volatility_24_feature']])

print(f"Pred: {pred.iloc[0]:.4f}, Risk: {risk.iloc[0]:.6f}, Turb: {turb.iloc[0]:.2f}")
res = sizer.size_portfolio(pred, reg, turb, athr, risk_scales=risk, dof=3)
print(f"Kelly Target: {res.iloc[0]}")
