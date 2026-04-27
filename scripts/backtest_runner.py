#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
from tqdm import tqdm

# Path setup
RL_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(RL_DIR))

from utils.signal_features import build_feature_matrix
from utils.risk_directors  import MahalanobisTurbulence, HMMRegimeModel
from utils.position_sizer  import FractionalKellySizer
from utils.filters         import SymmetricCUSUMFilter

# Config
THETA = 2_000_000 # 2M USDT per Dollar Bar
FEES = 0.001       # 0.1% Binance spot fee
DATA_DIR = Path("/home/nosferatu/freqtrade/user_data/data/binance")
MODEL_PATH = RL_DIR / "models" / "dollar_alpha_v1" / "latest_model.txt"

def build_synthetic_dollar_bars(df_1m: pd.DataFrame, theta: float):
    print(f"  Aggregating {len(df_1m):,} 1m candles into Dollar Bars (Theta=${theta:,.0f})...")
    df_1m = df_1m.sort_values("date")
    
    # Estimate dollar volume per minute
    df_1m["dollar_vol"] = df_1m["volume"] * df_1m["close"]
    df_1m["cum_vol"]    = df_1m["dollar_vol"].cumsum()
    df_1m["bar_id"]     = (df_1m["cum_vol"] // theta).astype(int)
    
    # Group into Dollar Bars
    df_bars = df_1m.groupby("bar_id").agg({
        "date": "last",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "dollar_vol": "sum"
    }).rename(columns={"dollar_vol": "notional"})
    
    # Add synthetic features for build_feature_matrix compatibility
    df_bars["buy_volume"] = df_bars["volume"] * 0.51
    df_bars["aggressor_ratio"] = 0.51
    df_bars["trade_count"] = 1000
    
    return df_bars.reset_index(drop=True)

def run_backtest():
    print("=== Institutional Priority 2 (High-Fidelity) Backtest ===")

    # ── Config ─────────────────────────────────────────────────────────────────
    # D-04 FIX: This date must match the training cutoff used in train_dollar_alpha.py.
    # Predictions BEFORE this date are potentially in-sample for the LGBM model.
    # Predictions AFTER this date are out-of-sample (safe for metric reporting).
    TRAIN_END = pd.Timestamp("2024-06-30")

    # 1. Load Reconstructed Tick Bars
    bars_path = RL_DIR / "cache" / "dollar_bars_tick_v1.feather"
    if not bars_path.exists():
        print(f"Error: {bars_path} not found. Run process_aggtrades_to_bars.py first.")
        return

    df_bars = pd.read_feather(str(bars_path))

    # 2. Map ETH features
    eth_df = pd.DataFrame({"date": df_bars["date"], "close": df_bars["eth_close"]}).set_index("date")

    # 3. Vectorized Feature Engineering
    print(f"  Feature Engineering on {len(df_bars):,} Tick Bars...")
    df_bars = df_bars.set_index("date")
    X_full = build_feature_matrix(df_bars.copy(), eth_df=eth_df, funding_series=None)

    # Overwrite synthetic OF features with real tick-level data
    X_full["aggressor_ratio_4h_mean_trade_feature"] = df_bars["aggressor_ratio"].rolling(100, min_periods=1).mean()
    X_full["cvd_4h_sum_trade_feature"] = df_bars["cvd"].rolling(100, min_periods=1).sum()

    # ── D-04 FIX: Alpha Oracle prediction strategy ─────────────────────────────
    # Priority 1: use pre-computed OOF predictions if available in the feather.
    #             These are the ONLY truly honest predictions (generated via CV).
    # Priority 2: for the strictly OOS segment (post TRAIN_END), run the model.
    #             These are valid: the model never saw this data.
    # Priority 3: in-sample fallback — marked with oof_valid=False, excluded from
    #             metric reporting to avoid overfitting illusion.
    regimes_path = RL_DIR / "cache" / "dollar_bars_btc_2000000_regimes.feather"
    has_real_oof = False

    if regimes_path.exists():
        df_reg = pd.read_feather(str(regimes_path)).set_index("date") if "date" in pd.read_feather(str(regimes_path)).columns else pd.read_feather(str(regimes_path))
        if "oof_pred" in df_reg.columns and "oof_valid" in df_reg.columns:
            oof_aligned = df_reg["oof_pred"].reindex(df_bars.index)
            oof_valid   = df_reg["oof_valid"].reindex(df_bars.index).fillna(False)
            if oof_aligned.notna().sum() > 0:
                df_bars["pred"]      = oof_aligned
                df_bars["oof_valid"] = oof_valid
                df_bars["pred_source"] = "oof_cv"
                has_real_oof = True
                n_valid = int(oof_valid.sum())
                print(f"  Alpha Oracle: using {n_valid:,} real OOF predictions from regimes feather.")

    if not has_real_oof:
        alpha         = lgb.Booster(model_file=str(MODEL_PATH))
        model_features = alpha.feature_name()
        for feat in model_features:
            if feat not in X_full.columns:
                X_full[feat] = 0.0

        # Run model only on the OOS segment (strictly after TRAIN_END)
        is_oos = df_bars.index > TRAIN_END
        preds  = np.full(len(df_bars), np.nan)
        valid  = np.zeros(len(df_bars), dtype=bool)

        if is_oos.sum() > 0:
            preds[is_oos]  = alpha.predict(X_full.loc[is_oos, model_features].fillna(0.0))
            valid[is_oos]  = True

        # In-sample fallback (before TRAIN_END): predict but flag as invalid
        is_ins = ~is_oos
        if is_ins.sum() > 0:
            preds[is_ins] = alpha.predict(X_full.loc[is_ins, model_features].fillna(0.0))
            valid[is_ins] = False
            print(f"  WARNING: {is_ins.sum():,} bars are IN-SAMPLE ({is_ins.sum()/len(df_bars)*100:.1f}%). "
                  "Their metrics will be excluded from the final report.")

        df_bars["pred"]       = preds
        df_bars["oof_valid"]  = valid
        df_bars["pred_source"] = np.where(valid, "oos_model", "insample_model")
        n_oos = int(is_oos.sum())
        print(f"  Alpha Oracle: {n_oos:,} OOS bars ({n_oos/len(df_bars)*100:.1f}% of dataset).")

    # Fill residual NaN predictions with 0.5 (flat / no edge)
    df_bars["pred"] = df_bars["pred"].fillna(0.5)

    # 4. Global Indicators
    turb_engine  = MahalanobisTurbulence(window=1000, step=250)
    hmm_model    = HMMRegimeModel(n_components=3, n_init=3)  # n_init aligned with production
    sizer        = FractionalKellySizer(kelly_fraction=0.5, long_only=True)
    cusum_filter = SymmetricCUSUMFilter()

    risk_vec = ["log_return_feature", "volatility_24_feature", "intraday_range_feature"]
    available_risk = [c for c in risk_vec if c in X_full.columns]
    for col in available_risk + ["log_return_feature", "volatility_24_feature"]:
        df_bars[col] = X_full[col].values

    turb_series = turb_engine.compute(df_bars, available_risk)
    # D-05 aligned: warmup NaN → P95 (conservative), not 0
    turb_p95 = float(turb_series.quantile(0.95)) if turb_series.notna().sum() > 10 else 5.0
    df_bars["turb_score"]     = turb_series.fillna(turb_p95)
    df_bars["turb_threshold"] = turb_engine.rolling_threshold(df_bars["turb_score"]).fillna(9.48)

    # 5. Priority 3: Causal HMM Inference (strictly on past data — no leakage)
    print("  Fitting causal HMM Regimes...")
    hmm_feats  = ["log_return_feature", "volatility_24_feature"]
    hmm_fitted = False
    bars_since_fit = 0
    df_bars["regime"] = "unknown"

    for i in tqdm(range(500, len(df_bars)), desc="HMM Causal Inference"):
        if not hmm_fitted or bars_since_fit >= 500:
            window = df_bars.iloc[max(0, i-1000): i]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hmm_model.fit(window, hmm_feats)
                hmm_fitted = True
                bars_since_fit = 0
            except Exception:
                pass

        tail   = df_bars.iloc[max(0, i-50): i+1]
        regime = hmm_model.predict_current(tail)

        regime_col_idx = df_bars.columns.get_loc("regime")
        df_bars.iloc[i, regime_col_idx] = regime
        bars_since_fit += 1

    # 6. Priority 1 & 4 Logic: Clock Decoupling + Asymmetric Hysteresis
    print("  Simulating Event-Driven Decision Clock...")
    df_bars["target_pos"] = 0.0
    current_target = 0.0
    last_close = None
    event_count = 0
    
    for i in tqdm(range(len(df_bars)), desc="Executing Decisions"):
        row = df_bars.iloc[i]
        close = row["close"]
        
        # Kelly Target from Sizer (Long-Only is handled inside sizer)
        risk_scale = max(float(row.get("volatility_24_feature", 0.0)), sizer.min_risk_scale)
        expected_net = sizer.estimate_expected_net_return(
            probabilities=pd.Series([row["pred"]]),
            risk_scales=pd.Series([risk_scale]),
        )
        raw_target = sizer.size_portfolio(
            probabilities=None,
            regimes=pd.Series([row["regime"]]),
            turbulence=pd.Series([row["turb_score"]]),
            adaptive_threshold=pd.Series([row["turb_threshold"]]),
            expected_net_returns=expected_net,
            risk_scales=pd.Series([risk_scale]),
        ).iloc[0]
        
        # CUSUM Filter (Priority 4: k=3.5)
        is_event = False
        cusum_h = 0.0
        if last_close is not None:
            log_ret = np.log(close / last_close)
            # sigma = dynamic 100-bar rolling std
            sigma = df_bars["log_return_feature"].iloc[max(0, i-100):i+1].std()
            cusum_h = 3.5 * sigma if (not np.isnan(sigma) and sigma > 0) else 0.005
            is_event = cusum_filter.check(log_ret, cusum_h)
            
        # Asymmetric Hysteresis (Priority 4 User Spec)
        pos_diff = abs(raw_target - current_target)
        should_update = False
        
        if is_event:
            # Case A: Entry (currently flat)
            if current_target == 0:
                if raw_target >= 0.10:
                    should_update = True
            
            # Case B: Exit / Reduction (reducing exposure)
            elif raw_target < current_target:
                if (pos_diff >= 0.05) or (raw_target == 0):
                    should_update = True
            
            # Case C: Increase / Rebalance
            elif raw_target > current_target:
                if pos_diff >= 0.10:
                    should_update = True
        
        if should_update:
            current_target = raw_target
            event_count += 1
            
        df_bars.iloc[i, df_bars.columns.get_loc("target_pos")] = current_target
        last_close = close

    # 7. Final Metrics
    print("  Calculating Post-Hardening Performance...")
    last_date = df_bars.index.max()
    df_full30 = df_bars[df_bars.index > last_date - pd.Timedelta(days=30)].copy()
    df_full30["returns"]      = df_full30["close"].pct_change().fillna(0)
    df_full30["strat_returns"] = df_full30["target_pos"].shift(1).fillna(0) * df_full30["returns"]
    df_full30["pos_change"]   = df_full30["target_pos"].diff().abs().fillna(0)
    df_full30["strat_returns"] -= df_full30["pos_change"] * FEES

    # D-04 FIX: Only compute metrics on OOS bars (oof_valid=True)
    if "oof_valid" in df_full30.columns:
        df_bt_valid = df_full30[df_full30["oof_valid"] == True].copy()
        n_insample   = int((df_full30["oof_valid"] == False).sum())
        n_total      = len(df_full30)
        oos_pct      = len(df_bt_valid) / n_total * 100
        validity_tag = f"OOS-ONLY ({len(df_bt_valid):,}/{n_total:,} bars, {oos_pct:.0f}%)"
        if n_insample > 0:
            print(f"\n  ⚠  {n_insample} in-sample bars in the 30-day window were EXCLUDED from metrics.")
    else:
        df_bt_valid  = df_full30.copy()
        validity_tag = "UNVERIFIED (no oof_valid column)"

    if len(df_bt_valid) < 10:
        print(f"\n  ⚠  Not enough OOS bars for reliable metrics ({len(df_bt_valid)} bars). "
              "Extend your OOS window or retrain with a shorter TRAIN_END cutoff.\n")
        return

    df_bt_valid["cum_returns"] = (1 + df_bt_valid["strat_returns"]).cumprod()
    df_bt_valid["bh_returns"]  = (1 + df_bt_valid["returns"]).cumprod()

    total_ret = (df_bt_valid["cum_returns"].iloc[-1] - 1) * 100
    bh_ret    = (df_bt_valid["bh_returns"].iloc[-1] - 1) * 100
    annfactor = len(df_bt_valid) / 30 * 365
    vol       = df_bt_valid["strat_returns"].std() * np.sqrt(annfactor)
    sharpe    = (df_bt_valid["strat_returns"].mean() * annfactor) / vol if vol > 0 else 0
    mdd       = (1 - df_bt_valid["cum_returns"] / df_bt_valid["cum_returns"].cummax()).max() * 100
    turnover  = df_full30["pos_change"].sum()  # turnover on full 30d

    print("\n" + "="*50)
    print(f"  PRIORITY 1 VALIDATION REPORT — {validity_tag}")
    print("="*50)
    print(f"  Strategy Return:        {total_ret:7.2f}%")
    print(f"  Buy & Hold Return:      {bh_ret:7.2f}%")
    print(f"  Sharpe Ratio:           {sharpe:7.2f}")
    print(f"  Max Drawdown:           {mdd:7.2f}%")
    print(f"  Total Position Adjusts: {event_count}")
    print(f"  Monthly Turnover:       {turnover:7.2f}x (Target < 10x)")
    print("="*50)

    df_full30.to_csv("backtest_priority_1_results.csv", index=False)
    print(f"\n  Results saved → backtest_priority_1_results.csv")
    print(f"  Columns include 'pred_source' for per-bar validity audit.")

if __name__ == "__main__":
    run_backtest()
