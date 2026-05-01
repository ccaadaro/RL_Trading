import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_RL_DIR = _SCRIPT_DIR.parent
for _p in [str(_RL_DIR), str(_RL_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.risk_directors import HMMRegimeModel
from utils.position_sizer import FractionalKellySizer
from utils.filters import SymmetricCUSUMFilter

_HTF_CONFIG_NEW = {
    "bull_calm":        (1.00, 0.00),
    "bull_neutral":     (0.90, 0.00),
    "high_vol_rebound": (0.75, 0.05),
    "bear_neutral":     (0.50, 0.00),
    "bear_calm":        (0.30, 0.05),
    "panic_selloff":    (0.00, 0.00),
    "unknown":          (0.50, 0.00),
}
_HTF_MULTIPLIERS_OLD = {k: v[0] for k, v in _HTF_CONFIG_NEW.items()}
_BYPASS_REGIMES_OLD = {"bull_calm", "bull_neutral"}
_BYPASS_REGIMES_NEW = {"bull_calm", "bull_neutral", "high_vol_rebound"}

HYSTERESIS_TO_BULL_OLD = 3
HYSTERESIS_TO_BULL_NEW = 3
HYSTERESIS_TO_BEAR = 6
_BULL_REGIMES = {"bull_calm", "bull_neutral", "high_vol_rebound"}

def fast_replay(df, use_new_logic=True):
    t0 = time.time()
    n = len(df)
    
    # Extract numpy arrays
    close = df["close"].values.astype(float)
    alpha_prob = df["alpha_prob"].values.astype(float)
    turb = df["turbulence_score"].values.astype(float)
    log_ret = df["log_return_feature"].values.astype(float)
    hmm_regime = df["hmm_semantic_regime"].values
    imbalance = df.get("book_imbalance", pd.Series(np.zeros(n))).values.astype(float)

    # Pre-calculate adaptive threshold
    turb_series = pd.Series(turb)
    adaptive_thr = turb_series.expanding(100).quantile(0.95).fillna(5.0).values
    
    # Pre-calculate Daily Volatility
    daily_vol = pd.Series(log_ret).rolling(288, min_periods=20).std().values * np.sqrt(288)
    daily_vol = np.nan_to_num(daily_vol, nan=0.02)
    barrier_heights = np.maximum(0.003, 1.5 * daily_vol)

    # Pre-calculate CUSUM sigma_eff
    sigma = pd.Series(log_ret).rolling(100).std().values
    if use_new_logic:
        sigma_robust = pd.Series(log_ret).rolling(100).apply(lambda x: np.nanmedian(np.abs(x)) * 1.4826).values
        sigma_eff = np.minimum(sigma, sigma_robust * 2.0)
    else:
        sigma_eff = sigma

    # Kelly components
    sizer = FractionalKellySizer(kelly_fraction=0.5, max_drawdown=0.10, min_risk_scale=1.0)
    
    # Kelly Base per bar
    probs = np.clip(alpha_prob, 0.0, 1.0)
    edge = 2.0 * probs - 1.0
    gross = edge * barrier_heights
    net_return = np.sign(gross) * np.maximum(np.abs(gross) - sizer.round_trip_cost_rate, 0.0)
    
    sharpe_proxy = net_return / barrier_heights
    kelly_base = np.tanh(sharpe_proxy)
    if sizer.long_only:
        kelly_base = np.maximum(kelly_base, 0.0)
    
    # turbulence penalties
    from scipy.stats import chi2
    p95_anchor = chi2.ppf(0.95, df=3)
    anchor = np.where(np.isnan(adaptive_thr), p95_anchor, adaptive_thr)
    excess_turb = np.maximum(turb - anchor, 0.0)
    g_penalties = np.exp(-excess_turb / 10.0)
    g_penalties = np.where(g_penalties < 0.05, 0.0, g_penalties)

    # HTF Regime
    if "end_time" in df.columns:
        ts_col = pd.to_datetime(df["end_time"])
    elif "t_close" in df.columns:
        ts_col = pd.to_datetime(df["t_close"], unit="ms", utc=True)
    else:
        ts_col = pd.to_datetime(df["timestamp"])
        
    df_htf = df.assign(_hour=ts_col.dt.floor("1h")).groupby("_hour", as_index=False).agg(
        log_return_feature=("log_return_feature", "sum"),
        volatility_24_feature=("volatility_24_feature", "mean"),
    )
    hmm_htf = HMMRegimeModel(n_components=3, n_init=3)
    _, htf_regimes = hmm_htf.fit_predict(df_htf, ["log_return_feature", "volatility_24_feature"])
    df_htf["htf_regime"] = htf_regimes.values

    df_temp = df.assign(_hour=ts_col.dt.floor("1h")).merge(df_htf[["_hour", "htf_regime"]], on="_hour", how="left")
    htf_regime_array = df_temp["htf_regime"].fillna("unknown").values

    # Setup Loop
    cusum = SymmetricCUSUMFilter()
    current_target_pos = 0.0
    pending_regime = None
    pending_regime_count = 0
    committed_regime = "unknown"
    hysteresis_to_bull = HYSTERESIS_TO_BULL_NEW if use_new_logic else HYSTERESIS_TO_BULL_OLD
    bypass_hold_bars = 0
    bypass_regimes = _BYPASS_REGIMES_NEW if use_new_logic else _BYPASS_REGIMES_OLD

    results = []
    # AGGRESSIVE REGIME CAPS for Phase 5
    regime_limits = {
        "bull_calm":        1.00,
        "bull_neutral":     0.75,
        "high_vol_rebound": 0.50,
        "bear_calm":        0.40,
        "bear_neutral":     0.20,
        "panic_selloff":    0.00,
        "unknown":          0.20,
    }
    ts_array = ts_col.values

    print("Starting loop...")
    for i in range(1, n):
        raw_reg = hmm_regime[i]
        hyst_needed = hysteresis_to_bull if raw_reg in _BULL_REGIMES else HYSTERESIS_TO_BEAR
        
        if raw_reg == committed_regime:
            pending_regime = None
            pending_regime_count = 0
        elif raw_reg == pending_regime:
            pending_regime_count += 1
            if pending_regime_count >= hyst_needed:
                committed_regime = raw_reg
                pending_regime = None
                pending_regime_count = 0
        else:
            pending_regime = raw_reg
            pending_regime_count = 1

        lr = log_ret[i]
        se = sigma_eff[i]
        h_t = 3.5 * se if (not np.isnan(se) and se > 0) else 0.005
        is_event = cusum.check(lr, h_t)

        if not is_event:
            continue

        # Kelly Sizing
        f_max = regime_limits.get(committed_regime, 0.0)
        raw_target_pos = kelly_base[i] * sizer.c_kelly * f_max * g_penalties[i]
        
        # Confidence scale (relaxed)
        conf = min(1.0, (2.0 * abs(alpha_prob[i] - 0.5)) / 0.10)
        raw_target_pos *= conf
        
        # HTF
        htf = htf_regime_array[i]
        if use_new_logic:
            htf_mult, htf_floor = _HTF_CONFIG_NEW.get(htf, (0.50, 0.00))
            ltf_rec = committed_regime in ("high_vol_rebound", "bull_calm", "bull_neutral")
            if htf_mult < 1.0:
                mult = raw_target_pos * htf_mult
                if ltf_rec and htf_floor > 0:
                    raw_target_pos = max(mult, min(raw_target_pos, htf_floor))
                else:
                    raw_target_pos = mult
        else:
            htf_mult = _HTF_MULTIPLIERS_OLD.get(htf, 0.50)
            raw_target_pos = raw_target_pos * htf_mult if htf_mult < 1.0 else raw_target_pos
            
        # Bypass
        bypass_applied = False
        _turb_ok = (np.isnan(adaptive_thr[i])) or (turb[i] < adaptive_thr[i])
        _imb_ok = imbalance[i] >= -0.20
        
        if use_new_logic and bypass_hold_bars > 0:
            _bull = {"bull_calm", "bull_neutral", "high_vol_rebound"}
            if (pending_regime in _bull or committed_regime in _bull) and alpha_prob[i] >= 0.45:
                if raw_target_pos < 0.10: raw_target_pos = 0.10
                bypass_hold_bars -= 1
            else:
                bypass_hold_bars = 0
                
        if alpha_prob[i] >= 0.65 and _turb_ok and _imb_ok and pending_regime in bypass_regimes and raw_target_pos < 0.10:
            raw_target_pos = 0.10
            bypass_applied = True
            if use_new_logic:
                bypass_hold_bars = hysteresis_to_bull

        # Gate (also opening it to 0.02)
        pos_diff = abs(raw_target_pos - current_target_pos)
        should_update = False
        if current_target_pos == 0:
            if raw_target_pos >= 0.02: should_update = True
        elif raw_target_pos < current_target_pos:
            if pos_diff >= 0.02 or raw_target_pos == 0: should_update = True
        elif raw_target_pos > current_target_pos:
            if pos_diff >= 0.05: should_update = True

        if should_update:
            current_target_pos = raw_target_pos
            results.append({
                "bar_idx": i,
                "timestamp": ts_array[i],
                "close": close[i],
                "alpha": alpha_prob[i],
                "regime_committed": committed_regime,
                "target_pos": current_target_pos,
                "bypass": bypass_applied
            })

    print(f"Elapsed: {time.time()-t0:.2f}s")
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = pd.read_feather("cache/dollar_bars_btc_2000000_sizing.feather")
    print("NEW LOGIC:")
    res_new = fast_replay(df, use_new_logic=True)
    res_new["logic"] = "new"
    print("OLD LOGIC:")
    res_old = fast_replay(df, use_new_logic=False)
    res_old["logic"] = "old"
    
    combined = pd.concat([res_old, res_new], ignore_index=True)
    combined.to_csv("reports/replay_global.csv", index=False)
    print(f"Saved to reports/replay_global.csv - Total events: {len(combined)}")
