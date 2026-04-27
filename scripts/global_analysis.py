import pandas as pd
import numpy as np

COST_BPS = 12

def extract_trades(d):
    d = d.reset_index(drop=True).copy()
    trades = []
    in_pos = False
    entry_idx = entry_price = entry_pos = None
    for i, row in d.iterrows():
        if not in_pos and row["target_pos"] > 0:
            in_pos = True
            entry_idx, entry_price, entry_pos = i, row["close"], row["target_pos"]
        elif in_pos and row["target_pos"] == 0:
            in_pos = False
            ret_gross = (row["close"] - entry_price) / entry_price
            ret_net = ret_gross * entry_pos - (COST_BPS / 1e4) * entry_pos
            trades.append({
                "entry_ts": d.at[entry_idx,"timestamp"],
                "exit_ts":  row["timestamp"],
                "entry_px": entry_price,
                "exit_px":  row["close"],
                "ticks":    i - entry_idx,
                "kelly":    entry_pos,
                "regime":   d.at[entry_idx,"regime_committed"],
                "alpha":    d.at[entry_idx,"alpha"],
                "bypass":   bool(d.at[entry_idx,"bypass"]),
                "ret_gross_pct": ret_gross * 100,
                "ret_net_pct":   ret_net * 100,
            })
    # Close any open trade at the end
    if in_pos:
        row = d.iloc[-1]
        ret_gross = (row["close"] - entry_price) / entry_price
        ret_net = ret_gross * entry_pos - (COST_BPS / 1e4) * entry_pos
        trades.append({
            "entry_ts": d.at[entry_idx,"timestamp"],
            "exit_ts":  row["timestamp"],
            "entry_px": entry_price,
            "exit_px":  row["close"],
            "ticks":    len(d) - 1 - entry_idx,
            "kelly":    entry_pos,
            "regime":   d.at[entry_idx,"regime_committed"],
            "alpha":    d.at[entry_idx,"alpha"],
            "bypass":   bool(d.at[entry_idx,"bypass"]),
            "ret_gross_pct": ret_gross * 100,
            "ret_net_pct":   ret_net * 100,
        })
    return pd.DataFrame(trades)


df = pd.read_csv("reports/replay_global.csv")

agg = {}
for variant in ["old", "new"]:
    sub = df[df["logic"] == variant]
    trades = extract_trades(sub)
    n = len(trades)
    if n == 0:
        agg[variant] = {"trades": 0, "wins": 0, "ret": 0.0, "bypass": 0, "best": 0, "worst": 0, "avg": 0}
    else:
        wins = (trades["ret_net_pct"] > 0).sum()
        total = trades["ret_net_pct"].sum()
        bp = trades["bypass"].sum()
        agg[variant] = {
            "trades": n, 
            "wins": wins, 
            "ret": total, 
            "bypass": bp,
            "best": trades["ret_net_pct"].max(),
            "worst": trades["ret_net_pct"].min(),
            "avg": trades["ret_net_pct"].mean()
        }

print(f"{'='*100}")
print(f"{'Var':<5} {'#Tr':<4} {'WR':>6} {'Avg':>8} {'Best':>8} {'Worst':>8} {'TotalNet':>10} {'Bypass':>7}")
print(f"{'='*100}")

for v in ["old", "new"]:
    a = agg[v]
    if a["trades"] == 0:
        print(f"{v:<5} 0 trades")
        continue
    wr = a["wins"] / a["trades"] * 100
    print(f"{v:<5} {a['trades']:<4} {wr:>5.1f}% {a['avg']:>7.3f}% {a['best']:>7.3f}% {a['worst']:>7.3f}% {a['ret']:>9.3f}% {a['bypass']:>5}/{a['trades']:<2}")

print(f"\n{'='*100}")
print("ALL NEW-MODEL TRADES (chronological):")
print(f"{'='*100}")
sub = df[df["logic"] == "new"]
trades = extract_trades(sub)
if not trades.empty:
    print(trades.to_string(index=False))
