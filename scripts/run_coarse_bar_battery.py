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

    feat_path = Path(f"cache/dollar_bars_btc_{int(theta)}_features.feather")
    if not feat_path.exists():
        print(f"  [eval] {feat_path} not found — skipping evaluation.")
        return {}

    LGB_PARAMS = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "n_estimators": 1000, "learning_rate": 0.015, "max_depth": 6,
        "num_leaves": 31, "min_child_samples": 200, "subsample": 0.7,
        "subsample_freq": 1, "colsample_bytree": 0.6,
        "reg_alpha": 5.0, "reg_lambda": 20.0, "n_jobs": -1,
        "random_state": 42, "verbose": -1,
    }

    df = pd.read_feather(feat_path)
    df = df[df["label"] != 0].copy()
    df["binary_target"] = (df["label"] == 1).astype(int)
    df = df.sort_values("date").reset_index(drop=True)

    available = [f for f in features if f in df.columns]
    missing = set(features) - set(available)
    if missing:
        print(f"  [eval] {len(missing)} features missing: {sorted(missing)[:5]}")

    if len(available) < 5:
        print(f"  [eval] Too few features ({len(available)}), skipping.")
        return {}

    sw = compute_uniqueness_weights(df) * compute_recency_weights(df)
    sw *= len(df) / sw.sum()

    oof_preds, avg_auc = purged_walk_forward(
        df, available, folds=4, sample_weights=sw
    )

    # Net bps estimate: on OOS bars where model predicts UP (prob > 0.55),
    # check forward_ret over vertical_bars horizon vs cost.
    v = cfg["vertical_bars"]
    df["forward_ret"] = df["close"].shift(-v) / df["close"] - 1
    df["oof_prob"] = np.where(np.isnan(oof_preds), 0.5, oof_preds)
    df_oos = df.iloc[len(df) // 4:].copy()  # last 75% as approximate OOS
    df_oos = df_oos.dropna(subset=["forward_ret"])

    # Selected trades: oof_prob > p60 in oos
    thr = df_oos["oof_prob"].quantile(0.60)
    selected = df_oos[df_oos["oof_prob"] > thr]
    gross_bps = selected["forward_ret"].mean() * 10000
    net_bps = gross_bps - COST_ROUNDTRIP * 10000
    coverage = len(selected) / max(len(df_oos), 1)

    n_selected = len(selected)
    n_total = len(df)
    bpd = cfg["bars_per_day"]
    trading_days = n_total / bpd
    trades_per_month = n_selected / (trading_days / 21)

    result = {
        "theta": theta,
        "n_bars": len(df),
        "bars_per_day": round(bpd, 1),
        "n_events_cusum": n_total,
        "oof_auc": round(avg_auc, 4),
        "gross_bps_per_trade": round(gross_bps, 2),
        "net_bps_per_trade": round(net_bps, 2),
        "coverage": round(coverage, 4),
        "est_trades_per_month": round(trades_per_month, 1),
        "vertical_bars": cfg["vertical_bars"],
        "hold_days": cfg["hold_days"],
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thetas", type=float, nargs="+",
                    default=[2_000_000, 20_000_000, 50_000_000])
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

    # Feature list for evaluation (adapt if features change)
    features = [
        "cvd_4h_sum_trade_feature", "aggressor_ratio_4h_mean_trade_feature",
        "whale_trades_4h_sum_trade_feature", "large_trades_4h_sum_trade_feature",
        "max_trade_usd_4h_max_trade_feature", "vwap_skew_4h_mean_trade_feature",
        "whale_intensity_4h_mean_trade_feature", "l2_imbalance_feature",
        "liq_vola_feature", "cross_exchange_premium_feature",
        "tv_cvd_zscore_feature", "tv_cvd_slope_feature",
        "tv_aggr_delta_feature", "tv_buy_sell_imbalance_feature",
        "tv_wvf_panic_feature", "tv_wvf_val_feature",
        "pr_exhaust_ob_feature", "pr_exhaust_os_feature",
        "pr_exhaust_ob_reversal_feature", "pr_exhaust_os_reversal_feature",
        "pr_spread_feature",
    ]

    results = []
    for theta in args.thetas:
        cfg = calibrate(theta, hold_days=args.hold_days, pt_sl=args.pt_sl)
        bars_path = f"cache/dollar_bars_btc_{int(theta)}.feather"
        labeled_path = f"cache/dollar_bars_btc_{int(theta)}_labeled.feather"
        features_path = f"cache/dollar_bars_btc_{int(theta)}_features.feather"

        print(f"\n{'='*60}")
        print(f"  theta={theta:,.0f}  ({cfg['bars_per_day']:.0f} bars/day)")
        print(f"  daily_window={cfg['daily_window']}  vertical_bars={cfg['vertical_bars']}"
              f"  ({cfg['hold_days']}d hold)  pt=sl={cfg['pt']}")
        print(f"{'='*60}")

        if args.eval_only:
            r = evaluate_theta(theta, cfg, features)
            if r:
                results.append(r)
            continue

        # Step 1: Build dollar bars
        if not args.skip_build:
            if not Path(args.trades).exists():
                print(f"  [ERROR] trades file not found: {args.trades}")
                sys.exit(1)
            rc = run(
                [sys.executable, "scripts/build_dollar_bars.py",
                 "--trades", args.trades, "--theta", str(theta)],
                f"build_bars theta={int(theta):,}"
            )
            if rc != 0:
                print("  Build failed. Aborting this theta.")
                continue
        else:
            if not Path(bars_path).exists():
                print(f"  [SKIP] {bars_path} not found — cannot skip build.")
                continue
            print(f"  [SKIP] build_bars — using existing {bars_path}")

        # Step 2: Triple Barrier with auto-calibrated params
        if not args.skip_tb:
            rc = run(
                [sys.executable, "scripts/compute_triple_barrier.py",
                 "--data", bars_path,
                 "--span", str(cfg["cusum_span"]),
                 "--daily-window", str(cfg["daily_window"]),
                 "--v-bars", str(cfg["vertical_bars"]),
                 "--pt", str(cfg["pt"]), "--sl", str(cfg["sl"]),
                 "--min-ret", str(args.min_ret)],
                f"triple_barrier theta={int(theta):,}"
            )
            if rc != 0:
                print("  Triple barrier failed. Aborting this theta.")
                continue
        else:
            if not Path(labeled_path).exists():
                print(f"  [SKIP] {labeled_path} not found — cannot skip TB.")
                continue
            print(f"  [SKIP] triple_barrier — using existing {labeled_path}")

        # Step 3: Feature engineering
        if not args.skip_features:
            rc = run(
                [sys.executable, "scripts/build_features_dollar.py",
                 "--bars", bars_path, "--labels", labeled_path],
                f"build_features theta={int(theta):,}"
            )
            if rc != 0:
                print("  Feature build failed. Aborting this theta.")
                continue
        else:
            if not Path(features_path).exists():
                print(f"  [SKIP] {features_path} not found — cannot skip features.")
                continue
            print(f"  [SKIP] build_features — using existing {features_path}")

        # Step 4: Evaluate
        r = evaluate_theta(theta, cfg, features)
        if r:
            results.append(r)

    # ─── Summary table ───────────────────────────────────────────────────────
    if results:
        print(f"\n{'='*80}")
        print("BATTERY SUMMARY")
        print(f"{'='*80}")
        cols = ["theta", "bars_per_day", "n_events_cusum", "oof_auc",
                "gross_bps_per_trade", "net_bps_per_trade",
                "coverage", "est_trades_per_month", "vertical_bars"]
        df_res = pd.DataFrame(results)[cols]
        df_res["theta"] = df_res["theta"].apply(lambda x: f"${x/1e6:.0f}M")
        print(df_res.to_string(index=False))

        # Key decision: is net_bps > 0 and trades_per_month manageable?
        print(f"\nDecision threshold: net_bps > 0 AND trades_per_month < 50")
        for r in results:
            ok = r.get("net_bps_per_trade", -999) > 0 and \
                 r.get("est_trades_per_month", 999) < 50
            tag = "PASS" if ok else "fail"
            print(f"  ${r['theta']/1e6:.0f}M: net={r.get('net_bps_per_trade','?'):+.1f}bps  "
                  f"TO={r.get('est_trades_per_month','?'):.1f}/mo  [{tag}]")
    else:
        print("\nNo results to summarize.")


if __name__ == "__main__":
    main()
