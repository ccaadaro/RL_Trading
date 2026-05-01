import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import sys
import os
from typing import Optional

_HERE = Path(__file__).resolve().parent.parent
sys.path.append(str(_HERE))

from utils.signal_features import SIGNAL_FEAT_COLS_V2, FEATURE_SET_INSTITUTIONAL
from utils.risk_directors import MahalanobisTurbulence, HMMRegimeModel

# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering constants
# ─────────────────────────────────────────────────────────────────────────────
ALPHA_SMOOTH_SPAN = 30
ALPHA_ZSCORE_WIN = 10000
ALPHA_PCTILE_WIN = 10000
ALPHA_PERSIST_WIN = 50
TURB_WINDOW = 2000
TURB_PCTILE_WIN = 2000
HMM_INITIAL_FIT = 50000
HMM_BLOCK = 20000
HMM_LOOKBACK = 100000

def _build_alpha_derived_features(df: pd.DataFrame, alpha_col: str, multiplier: float) -> None:
    # Apply multiplier to raw alpha before smoothing (mirroring live strategy)
    s = 0.5 + (df[alpha_col] - 0.5) * multiplier
    df["alpha_prob"] = s
    df["alpha_prob_smooth"] = s.ewm(span=ALPHA_SMOOTH_SPAN).mean()
    df["alpha_prob_zscore"] = ((s - s.rolling(ALPHA_ZSCORE_WIN).mean()) / s.rolling(ALPHA_ZSCORE_WIN).std())
    df["alpha_prob_percentile"] = s.rolling(ALPHA_PCTILE_WIN).rank(pct=True)
    df["alpha_signal_persistence"] = (s > 0.55).astype(int).rolling(ALPHA_PERSIST_WIN).sum()

def _walk_forward_hmm(df: pd.DataFrame, hmm_feats: list) -> np.ndarray:
    n = len(df)
    states = np.full(n, -1, dtype=int)
    eng = HMMRegimeModel(n_components=3, verbose=False)
    eng.fit(df.iloc[:HMM_INITIAL_FIT], hmm_feats)
    X_init = df.iloc[:HMM_INITIAL_FIT][hmm_feats].fillna(0).values
    init_states = eng.model.predict(X_init)
    states[:HMM_INITIAL_FIT] = [eng.state_map.get(s, -1) for s in init_states]
    for i in range(HMM_INITIAL_FIT, n, HMM_BLOCK):
        end_idx = min(i + HMM_BLOCK, n)
        train_window = df.iloc[max(0, i - HMM_LOOKBACK): i]
        eng.fit(train_window, hmm_feats)
        X_block = df.iloc[i:end_idx][hmm_feats].fillna(0).values
        hidden = eng.model.predict(X_block)
        states[i:end_idx] = [eng.state_map.get(s, -1) for s in hidden]
    return states

def _simulate(target_pos: np.ndarray, bar_return: np.ndarray, total_cost_rate: float, rebal_threshold: float = 0.1):
    n = len(target_pos)
    actual = np.zeros(n)
    cur = 0.0
    for i in range(n):
        if abs(target_pos[i] - cur) > rebal_threshold:
            cur = target_pos[i]
        actual[i] = cur
    trade_diff = np.abs(np.diff(actual, prepend=0))
    costs = trade_diff * total_cost_rate
    pnl = (np.roll(actual, 1) * bar_return) - costs
    pnl[0] = 0
    cum = (1 + pnl).cumprod()
    return actual, cum, pnl, trade_diff

def _summarize(name: str, cum: np.ndarray, pnl: np.ndarray, trade_diff: np.ndarray, total_months: float, multiplier: float):
    total_ret = cum[-1] - 1
    cum_max = np.maximum.accumulate(cum)
    dd = (cum / cum_max) - 1
    n_trades = int((trade_diff > 0).sum() / 2)
    monthly_to = trade_diff.sum() / total_months if total_months > 0 else 0.0
    time_in_market = (np.cumsum(trade_diff > 0) % 2 == 1).mean()
    net_bps = (total_ret / n_trades * 10000) if n_trades > 0 else 0.0
    
    return {
        "multiplier": multiplier,
        "roi": total_ret,
        "dd": dd.min(),
        "trades": n_trades,
        "to_monthly": monthly_to,
        "time_in_market": time_in_market,
        "net_bps": net_bps
    }

def run_replay(model_path: str, data_path: str, meta_model_path: Optional[str] = None,
               meta_threshold: float = 0.60, min_entry: float = 0.55,
               exit_threshold: float = 0.50, min_hold_bars: int = 50,
               use_dynamic_threshold: bool = True, alpha_multiplier: float = 1.0):
    df = pd.read_feather(data_path)
    alpha_model = lgb.Booster(model_file=model_path)
    meta_model = None
    if meta_model_path and os.path.exists(meta_model_path):
        meta_model = lgb.Booster(model_file=meta_model_path)

    features = FEATURE_SET_INSTITUTIONAL if len(alpha_model.feature_name()) == 14 else SIGNAL_FEAT_COLS_V2
    alpha_preds = alpha_model.predict(df[features])
    df["alpha_signal"] = alpha_preds
    _build_alpha_derived_features(df, "alpha_signal", alpha_multiplier)

    if use_dynamic_threshold:
        dyn = pd.Series(alpha_preds).rolling(window=1000, min_periods=100).quantile(0.90).fillna(0.60).values
    else:
        dyn = np.full(len(df), min_entry)

    turb_engine = MahalanobisTurbulence(window=TURB_WINDOW)
    risk_feats = ["log_return_feature", "volatility_24_feature", "intraday_range_feature"]
    df["turbulence_score"] = turb_engine.compute(df, risk_feats).bfill().fillna(5.0)
    df["turbulence_percentile"] = df["turbulence_score"].rolling(TURB_PCTILE_WIN).rank(pct=True)
    df["hmm_state"] = _walk_forward_hmm(df, ["log_return_feature", "volatility_24_feature", "aggressor_ratio", "intraday_range_feature"])

    df["meta_pass"] = 1
    if meta_model is not None:
        df["spread_bps"] = df["pr_spread_feature"] * 10000 if "pr_spread_feature" in df.columns else 2.0
        df["expected_cost_bps"] = 5 + df["spread_bps"] + 2
        meta_feats = pd.DataFrame({
            "alpha_prob": df["alpha_prob"], "alpha_prob_smooth": df["alpha_prob_smooth"],
            "alpha_prob_zscore": df["alpha_prob_zscore"], "alpha_prob_percentile": df["alpha_prob_percentile"],
            "alpha_signal_persistence": df["alpha_signal_persistence"], "turbulence_score": df["turbulence_score"],
            "turbulence_percentile": df["turbulence_percentile"], "hmm_state": df["hmm_state"].astype(int),
            "volatility_24_feature": df["volatility_24_feature"], "aggressor_ratio": df.get("aggressor_ratio", 0.0),
            "l2_imbalance_feature": df.get("l2_imbalance_feature", 0.0), "spread_bps": df["spread_bps"],
            "expected_cost_bps": df["expected_cost_bps"],
        }).fillna(0)
        df["meta_prob"] = meta_model.predict(meta_feats)
        df["meta_pass"] = (df["meta_prob"] >= meta_threshold).astype(int)

    target_pos = np.zeros(len(df))
    cur_t, bars_in = 0.0, 0
    smooth = df["alpha_prob_smooth"].values
    meta_ok = df["meta_pass"].values
    for i in range(len(df)):
        thr = max(min_entry, dyn[i]) if use_dynamic_threshold else min_entry
        if cur_t == 0.0:
            if smooth[i] > thr and meta_ok[i] == 1:
                cur_t = 1.0
                bars_in = 0
        else:
            bars_in += 1
            if (smooth[i] < exit_threshold or meta_ok[i] == 0) and bars_in >= min_hold_bars:
                cur_t = 0.0
        target_pos[i] = cur_t

    bar_return = df["close"].pct_change().fillna(0).values
    total_cost_rate = 0.0007 # 7 bps per side
    total_days = len(df) / 17.6 / 24 # approx for 2M dollar bars
    total_months = total_days / 30.44
    
    actual_pos, cum, pnl, trade_diff = _simulate(target_pos, bar_return, total_cost_rate)
    return _summarize("Strategy", cum, pnl, trade_diff, total_months, alpha_multiplier)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--multipliers", type=str, default="0.3,0.5,0.7,1.0")
    parser.add_argument("--model", type=str, default="models/dollar_alpha_v1/model_a_institutional.txt")
    parser.add_argument("--meta", type=str, default="models/meta_model_v1/gatekeeper.txt")
    args = parser.parse_args()

    data_path = "cache/dollar_bars_btc_2000000_features.feather"
    mults = [float(m) for m in args.multipliers.split(",")]
    results = []
    for m in mults:
        print(f"Testing multiplier: {m}")
        results.append(run_replay(args.model, data_path, meta_model_path=args.meta, alpha_multiplier=m))

    print("\n" + "="*80)
    print(f"{'Mult':<6} | {'ROI':<10} | {'Trades':<8} | {'TO/m':<8} | {'BPS/T':<8} | {'DD':<8} | {'Market%':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['multiplier']:<6.1f} | {r['roi']:<10.2%} | {r['trades']:<8d} | {r['to_monthly']:<8.2f} | {r['net_bps']:<8.1f} | {r['dd']:<8.1%} | {r['time_in_market']:<8.1%}")
    print("="*80)
