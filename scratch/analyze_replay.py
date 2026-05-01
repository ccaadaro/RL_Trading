import pandas as pd
import numpy as np

df = pd.read_feather("cache/replay_model_a_institutional.feather")
print("=== Replay Cache Analysis ===")
print(f"Total samples: {len(df)}")
print(f"Position distribution:\n{df['position'].value_counts(normalize=True)}")

# We don't have meta_prob in the output feather yet, let's check what we have
print(f"Columns: {df.columns.tolist()}")

# Let's check trade frequency
df['change'] = df['position'].diff().abs()
trades = df[df['change'] > 0.1]
print(f"Total trades: {len(trades)}")
print(f"Avg time between trades: {len(df)/len(trades) if len(trades) > 0 else 0:.1f} bars")

# Check costs
print(f"Total costs: {df['costs'].sum():.4f}")
print(f"Total return (gross): {df['strategy_return'].sum() + df['costs'].sum():.4f}")
