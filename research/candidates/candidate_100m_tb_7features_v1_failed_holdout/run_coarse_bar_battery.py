#!/usr/bin/env python3
"""
scripts/run_coarse_bar_battery.py

Experiment battery: test alpha signal quality at coarser dollar bar resolutions.

Motivation: At $2M bars (308k bars / 2 years = 422 bars/day), any model with
AUC ~0.52 produces negative net return at 14bps/roundtrip. The hypothesis is
that coarser bars carry more persistent (and therefore more tradeable) signals.

Pipeline per theta:
  1. build_dollar_bars.py --theta T    → raw OHLCV + microstructure
  2. compute_triple_barrier.py         → CUSUM events + TB labels (auto-calibrated)
  3. build_features_dollar.py          → feature matrix
  4. train_dollar_alpha.py equivalent  → OOF AUC, net bps/trade estimate

Key parameters auto-calibrated from theta:
  bars_per_day:   288 × (2_000_000 / theta)
  daily_window:   bars_per_day       (≡ compute daily vol over 1 day of bars)
  vertical_bars:  hold_days × bars_per_day   (default hold_days=3)
  cusum_span:     4 × bars_per_day   (EWMA volatility span for CUSUM events)
  pt, sl:         wider at coarser resolution (more room for signal to develop)

Usage:
  python scripts/run_coarse_bar_battery.py --thetas 2000000 20000000 50000000
  python scripts/run_coarse_bar_battery.py --thetas 20000000 --skip-build  # if bars exist
  python scripts/run_coarse_bar_battery.py --thetas 20000000 --skip-build --skip-features
"""
import argparse
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

_HERE = Path(__file__).resolve().parent.parent
sys.path.append(str(_HERE))

BASELINE_THETA = 2_000_000
BASELINE_BARS_PER_DAY = 288

COST_RATE = 0.0007      # 7bps one-way (5bps fee + 2bps slippage)
COST_ROUNDTRIP = 0.0014  # 14bps round-trip


def bars_per_day(theta: float) -> float:
    return BASELINE_BARS_PER_DAY * (BASELINE_THETA / theta)


def calibrate(theta: float, hold_days: int = 3, pt_sl: float = 2.0) -> dict:
    bpd = bars_per_day(theta)
    return {
        "daily_window":   max(10, int(round(bpd))),
        "vertical_bars":  max(5, int(round(hold_days * bpd))),
        "cusum_span":     max(20, int(round(4 * bpd))),
        "pt":             pt_sl,
        "sl":             pt_sl,
        "bars_per_day":   bpd,
        "hold_days":      hold_days,
    }


def run(cmd: list, label: str) -> int:
    print(f"\n  [{label}] $ {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=str(_HERE))
    elapsed = time.time() - t0
    status = "OK" if ret.returncode == 0 else f"ERROR (code {ret.returncode})"
    print(f"  [{label}] {status}  ({elapsed:.0f}s)")
    return ret.returncode


def evaluate_theta(theta: float, cfg: dict, features: list) -> dict:
    """Load labeled features feather and run OOF AUC + net bps estimate."""
    import lightgbm as lgb
    import heapq
    from sklearn.metrics import roc_auc_score
    from scripts.train_dollar_alpha import (
        compute_uniqueness_weights, compute_recency_weights, purged_walk_forward
    )

    feat_path = Path(f"cache/dollar_bars_btc_{int(theta)}_minimal_features.feather")
    if not feat_path.exists():
        print(f"  [eval] {feat_path} not found — skipping evaluation.")
        return {}

    # ─── 0. HYPERPARAMETERS ───
    # Small LightGBM for better generalization on sparse coarse bars
    if theta >= 50_000_000:
        LGB_PARAMS = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
            "n_estimators": 200, "learning_rate": 0.02, "max_depth": 2,
            "num_leaves": 4, "min_child_samples": 50, "subsample": 0.7,
            "subsample_freq": 1, "colsample_bytree": 0.8,
            "reg_alpha": 10.0, "reg_lambda": 10.0, "n_jobs": -1,
            "random_state": 42, "verbose": -1,
        }
    else:
        LGB_PARAMS = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
            "n_estimators": 500, "learning_rate": 0.015, "max_depth": 3,
            "num_leaves": 8, "min_child_samples": 100, "subsample": 0.7,
            "subsample_freq": 1, "colsample_bytree": 0.6,
            "reg_alpha": 5.0, "reg_lambda": 20.0, "n_jobs": -1,
            "random_state": 42, "verbose": -1,
        }

    df_all = pd.read_feather(feat_path)
    df_all["date"] = pd.to_datetime(df_all["date"])
    
    # Filter for labeled samples for training
    df_labeled = df_all[df_all["label"] != 0].copy()
    df_labeled["binary_target"] = (df_labeled["label"] == 1).astype(int)
    df_labeled = df_labeled.sort_values("date").reset_index(drop=True)

    available = [f for f in features if f in df_labeled.columns]
    if len(available) < 5:
        print(f"  [eval] Too few features ({len(available)}), skipping.")
        return {}

    # 1. SPLIT LOGIC (Corrected)
    research_df = df_labeled[df_labeled["date"] < "2024-01-01"].copy()
    validation_df = df_labeled[(df_labeled["date"] >= "2024-01-01") & (df_labeled["date"] < "2025-01-01")].copy()
    holdout_df = df_labeled[df_labeled["date"] >= "2025-01-01"].copy()
    
    eval_df = pd.concat([research_df, validation_df])
    if len(eval_df) < 300:
        print(f"  [eval] Insufficient data for Research/Validation ({len(eval_df)} samples).")
        return {}

    sw = compute_uniqueness_weights(eval_df) * compute_recency_weights(eval_df)
    sw *= len(eval_df) / sw.sum()

    alpha_probs, avg_auc = purged_walk_forward(
        eval_df, available, folds=4, sample_weights=sw
    )

    # 2. SIMULATION (Vectorized Portfolio Engine)
    eval_df["alpha_prob_oof"] = alpha_probs
    val_oos = eval_df[eval_df["date"] >= "2024-01-01"].copy()
    
    if len(val_oos) < 50:
        return {"theta": theta, "oof_auc": round(avg_auc, 4), "status": "insufficient_val"}

    # Map signals back to full time series for proper accounting
    # Use full series to account for continuous exposure and Buy & Hold comparison
    full_val = df_all[(df_all["date"] >= "2024-01-01") & (df_all["date"] < "2025-01-01")].copy()
    full_val = full_val.sort_values("date").reset_index(drop=True)
    
    # Threshold signals
    thr = val_oos["alpha_prob_oof"].quantile(0.60)
    val_oos["is_signal"] = (val_oos["alpha_prob_oof"] > thr).astype(int)
    
    # Map signals to full_val by 'date'
    full_val = full_val.merge(val_oos[["date", "is_signal"]], on="date", how="left").fillna(0)
    
    def simulate_portfolio(price_series, signal_series, v_bars):
        n = len(price_series)
        rets = price_series.pct_change().fillna(0).values
        sigs = signal_series.values
        
        # Track active signal duration
        active_matrix = np.zeros((n, v_bars))
        for i in range(n):
            if sigs[i] == 1:
                # Signal i is active from i to min(i + v_bars, n)
                end = min(i + v_bars, n)
                active_matrix[i : end, 0] = 1 # Mark as active in the period
        
        # Sum active signals per bar
        active_counts = np.zeros(n)
        for i in np.where(sigs == 1)[0]:
            active_counts[i : min(i + v_bars, n)] += 1
            
        # CORRECTED: For a single-asset long-only strategy, exposure is 1.0 if any signal is active.
        target_exposure = np.where(active_counts > 0, 1.0, 0.0)
        
        # Portfolio returns
        p_rets = np.zeros(n)
        p_rets[1:] = target_exposure[:-1] * rets[1:]
        
        # Costs (7bps per side = 14bps roundtrip if full flip)
        # cost_t = abs(exp_t - exp_t-1) * 0.0007
        costs = np.abs(np.diff(target_exposure, prepend=0)) * COST_RATE
        
        net_rets = p_rets - costs
        equity = np.cumprod(1 + net_rets)
        
        # Metrics
        total_ret = equity[-1] - 1
        n_trades = np.sum(sigs)
        net_bps_per_trade = (total_ret * 10000 / n_trades) if n_trades > 0 else 0
        
        peak = np.maximum.accumulate(equity)
        dd = np.min(equity / peak - 1)
        
        tim = np.mean(active_counts > 0)
        turnover = np.sum(np.abs(np.diff(target_exposure, prepend=0)))
        
        return {
            "equity": equity, "roi": total_ret, "dd": dd, "tim": tim,
            "n_trades": n_trades, "net_bps": net_bps_per_trade, "to": turnover
        }

    v = cfg["vertical_bars"]
    model_perf = simulate_portfolio(full_val["close"], full_val["is_signal"], v)
    
    # Benchmarks
    bh_ret = full_val["close"].iloc[-1] / full_val["close"].iloc[0] - 1
    
    # Always Long (exposure = 1.0 constantly)
    always_long_rets = full_val["close"].pct_change().fillna(0)
    # Entry cost only (exit not in loop)
    al_net_rets = always_long_rets.copy()
    al_net_rets.iloc[0] -= COST_RATE # Entry
    al_equity = np.cumprod(1 + al_net_rets)
    al_roi = al_equity.iloc[-1] - 1

    # Random Benchmarks (Matched Time-in-Market)
    n_sims = 200
    random_rois = []
    target_sigs = int(val_oos["is_signal"].sum())
    
    for _ in range(n_sims):
        # Place random signals
        rand_sigs = np.zeros(len(full_val))
        indices = np.random.choice(len(full_val), target_sigs, replace=False)
        rand_sigs[indices] = 1
        sim = simulate_portfolio(full_val["close"], pd.Series(rand_sigs), v)
        random_rois.append(sim["roi"])
    
    p50_rand = np.percentile(random_rois, 50)
    p95_rand = np.percentile(random_rois, 95)
    
    tag = "PASS" if model_perf["roi"] > p95_rand else "FAIL"

    result = {
        "theta": theta,
        "bars_per_day": round(cfg["bars_per_day"], 1),
        "n_train": len(research_df),
        "n_val": len(validation_df),
        "oof_auc": round(avg_auc, 4),
        "roi_net": round(model_perf["roi"], 4),
        "max_dd": round(model_perf["dd"], 4),
        "net_bps": round(model_perf["net_bps"], 2),
        "to_monthly": round(model_perf["to"] / (len(full_val) / cfg["bars_per_day"] / 30.4), 2),
        "tim": round(model_perf["tim"], 4),
        "roi_bh": round(bh_ret, 4),
        "roi_al": round(al_roi, 4),
        "rand_p50": round(p50_rand, 4),
        "rand_p95": round(p95_rand, 4),
        "alpha_tag": tag
    }
    return result


def process_theta(theta, args, features):
    """Full pipeline for a single theta resolution, redirected to its own log."""
    theta_name = f"{int(theta/1e6)}M"
    log_path = Path(f"logs/battery_theta_{theta_name}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    cfg = calibrate(theta, hold_days=args.hold_days, pt_sl=args.pt_sl)
    bars_path = f"cache/dollar_bars_btc_{int(theta)}.feather"
    labeled_path = f"cache/dollar_bars_btc_{int(theta)}_labeled.feather"
    features_path = f"cache/dollar_bars_btc_{int(theta)}_features.feather"

    # Redirection helper
    with open(log_path, "w", buffering=1) as f:
        def log(msg):
            print(msg, file=f, flush=True)

        log(f"\n{'='*60}")
        log(f"  theta={theta:,.0f}  ({cfg['bars_per_day']:.0f} bars/day)")
        log(f"  daily_window={cfg['daily_window']}  vertical_bars={cfg['vertical_bars']}"
              f"  ({cfg['hold_days']}d hold)  pt=sl={cfg['pt']}")
        log(f"{'='*60}")

        if args.eval_only:
            return evaluate_theta(theta, cfg, features)

        # Step 1: Build dollar bars
        if not args.skip_build:
            if not Path(args.trades).exists():
                log(f"  [ERROR] trades file not found: {args.trades}")
                return {}
            
            log(f"  [build_bars theta={int(theta):,}] Starting...")
            cmd = [sys.executable, "scripts/build_dollar_bars.py", "--trades", args.trades, "--theta", str(theta)]
            ret = subprocess.run(cmd, cwd=str(_HERE), stdout=f, stderr=f)
            if ret.returncode != 0:
                log(f"  Build failed (code {ret.returncode}).")
                return {}
        else:
            if not Path(bars_path).exists():
                log(f"  [SKIP] {bars_path} not found.")
                return {}

        # Step 2: Triple Barrier
        if not args.skip_tb:
            log(f"  [triple_barrier theta={int(theta):,}] Starting...")
            cmd = [sys.executable, "scripts/compute_triple_barrier.py",
                   "--data", bars_path, "--span", str(cfg["cusum_span"]),
                   "--daily-window", str(cfg["daily_window"]), "--v-bars", str(cfg["vertical_bars"]),
                   "--pt", str(cfg["pt"]), "--sl", str(cfg["sl"]), "--min-ret", str(args.min_ret),
                   "--pt-floor", "0.0060", "--sl-floor", "0.0040"]
            ret = subprocess.run(cmd, cwd=str(_HERE), stdout=f, stderr=f)
            if ret.returncode != 0:
                log(f"  Triple barrier failed.")
                return {}

        # Step 3: Feature engineering
        if not args.skip_features:
            log(f"  [build_features theta={int(theta):,}] Starting...")
            cmd = [sys.executable, "scripts/build_features_minimal.py",
                   "--data", bars_path, "--labels", labeled_path]
            ret = subprocess.run(cmd, cwd=str(_HERE), stdout=f, stderr=f)
            if ret.returncode != 0:
                log(f"  Feature build failed.")
                return {}

        # Step 4: Evaluate
        log(f"  [evaluate theta={int(theta):,}] Starting OOF validation...")
        return evaluate_theta(theta, cfg, features)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thetas", type=float, nargs="+",
                    default=[2_000_000, 20_000_000, 50_000_000, 100_000_000, 200_000_000, 500_000_000])
    ap.add_argument("--trades", type=str,
                    default="/home/nosferatu/freqtrade/user_data/data/binance/BTC_USDT-trades.feather")
    ap.add_argument("--hold-days", type=int, default=3,
                    help="Target holding horizon in real trading days for vertical barrier")
    ap.add_argument("--pt-sl", type=float, default=2.0,
                    help="PT/SL multiplier on daily volatility")
    ap.add_argument("--min-ret", type=float, default=0.003,
                    help="Minimum barrier width (fraction, must cover roundtrip cost)")
    ap.add_argument("--skip-build", action="store_true",
                    help="Skip build_dollar_bars step (use existing feathers)")
    ap.add_argument("--skip-tb", action="store_true",
                    help="Skip compute_triple_barrier step")
    ap.add_argument("--skip-features", action="store_true",
                    help="Skip build_features_dollar step")
    ap.add_argument("--eval-only", action="store_true",
                    help="Only evaluate existing feature feathers, no building")
    args = ap.parse_args()

    # Minimal Feature Set for Institutional Audit
    features = [
        "return_3_bars_feature", "return_5_bars_feature", "vol_10_feature",
        "cvd_slope_feature", "aggressor_imbalance_feature", "hma_dist_feature",
        "wvf_zscore_feature"
    ]

    print(f"Launching battery for {len(args.thetas)} thetas in parallel...")
    print(f"Individual logs in logs/battery_theta_XM.log")
    
    results = []
    with ProcessPoolExecutor(max_workers=min(len(args.thetas), 10)) as executor:
        future_to_theta = {executor.submit(process_theta, t, args, features): t for t in args.thetas}
        
        for future in as_completed(future_to_theta):
            theta = future_to_theta[future]
            try:
                r = future.result()
                if r:
                    print(f"  [DONE] theta=${int(theta/1e6)}M | AUC: {r.get('oof_auc','?')}")
                    results.append(r)
                else:
                    print(f"  [FAIL] theta=${int(theta/1e6)}M")
            except Exception as exc:
                print(f"  [ERROR] theta=${int(theta/1e6)}M generated an exception: {exc}")

    # ─── Summary table ───────────────────────────────────────────────────────
    if results:
        print(f"\n{'='*100}")
        print("INSTITUTIONAL AUDIT SUMMARY (Validation: 2024)")
        print(f"{'='*100}")
        cols = ["theta", "oof_auc", "roi_net", "roi_bh", "roi_al", "rand_p95", 
                "net_bps", "to_monthly", "tim", "max_dd", "alpha_tag"]
        df_res = pd.DataFrame(results)[cols]
        df_res["theta"] = df_res["theta"].apply(lambda x: f"${x/1e6:.0f}M")
        print(df_res.to_string(index=False))

        print(f"\nBenchmark Key:")
        print(f"  roi_bh: Buy & Hold (Full 2024)")
        print(f"  roi_al: Always Long (100% Exposure with turnover costs)")
        print(f"  rand_p95: 95th percentile of 200 random matched simulations")
        print(f"  alpha_tag: PASS if roi_net > rand_p95")
    else:
        print("\nNo results to summarize.")


if __name__ == "__main__":
    main()
