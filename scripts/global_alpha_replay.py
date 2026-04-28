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
# Feature engineering — MUST match scripts/generate_metamodel_data.py exactly,
# otherwise the meta-model receives an out-of-distribution feature vector.
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


def _build_alpha_derived_features(df: pd.DataFrame, alpha_col: str = "alpha_signal") -> None:
    s = df[alpha_col]
    df["alpha_prob"] = s
    df["alpha_prob_smooth"] = s.ewm(span=ALPHA_SMOOTH_SPAN).mean()
    df["alpha_prob_zscore"] = (
        (s - s.rolling(ALPHA_ZSCORE_WIN).mean()) / s.rolling(ALPHA_ZSCORE_WIN).std()
    )
    df["alpha_prob_percentile"] = s.rolling(ALPHA_PCTILE_WIN).rank(pct=True)
    df["alpha_signal_persistence"] = (s > 0.55).astype(int).rolling(ALPHA_PERSIST_WIN).sum()


def _walk_forward_hmm(df: pd.DataFrame, hmm_feats: list) -> np.ndarray:
    """Walk-forward HMM mirroring generate_metamodel_data.py to avoid look-ahead."""
    n = len(df)
    states = np.full(n, -1, dtype=int)
    eng = HMMRegimeModel(n_components=3, verbose=False)

    print(f"  HMM initial fit on first {HMM_INITIAL_FIT} bars...")
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


def _simulate(target_pos: np.ndarray, bar_return: np.ndarray, total_cost_rate: float,
              rebal_threshold: float = 0.1):
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


def _summarize(name: str, cum: np.ndarray, pnl: np.ndarray, trade_diff: np.ndarray,
               total_months: float, bars_per_year: float = 365 * 24):
    total_ret = cum[-1] - 1
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(bars_per_year) if pnl.std() > 0 else 0.0
    cum_max = np.maximum.accumulate(cum)
    dd = (cum / cum_max) - 1
    n_trades = int((trade_diff > 0).sum() / 2)  # round-trips
    monthly_to = trade_diff.sum() / total_months if total_months > 0 else 0.0
    pos_active = (np.cumsum(trade_diff > 0) % 2 == 1).mean()
    print(f"  {name:32s}  ROI {total_ret:+7.2%}  Sharpe {sharpe:+5.2f}  "
          f"DD {dd.min():+7.2%}  Trades {n_trades:5d}  TimeInMkt {pos_active:5.1%}  TO/m {monthly_to:5.2f}x")
    return {"name": name, "roi": total_ret, "sharpe": sharpe, "dd": dd.min(),
            "trades": n_trades, "time_in_market": pos_active, "monthly_turnover": monthly_to}


def run_replay(model_path: str, data_path: str, meta_model_path: Optional[str] = None,
               meta_threshold: float = 0.60, min_entry: float = 0.55,
               exit_threshold: float = 0.50, min_hold_bars: int = 50,
               use_dynamic_threshold: bool = True):
    print(f"Loading data from {data_path}...")
    df = pd.read_feather(data_path)

    print(f"Loading Alpha model from {model_path}...")
    alpha_model = lgb.Booster(model_file=model_path)

    meta_model = None
    if meta_model_path and os.path.exists(meta_model_path):
        print(f"Loading Meta-model from {meta_model_path}...")
        meta_model = lgb.Booster(model_file=meta_model_path)

    n_features = len(alpha_model.feature_name())
    if n_features == 21:
        features, model_name = SIGNAL_FEAT_COLS_V2, "Elite v2.1"
    elif n_features == 14:
        features, model_name = FEATURE_SET_INSTITUTIONAL, "Institutional Base"
    else:
        features = [c for c in df.columns
                    if c.endswith("_feature") and c in alpha_model.feature_name()]
        model_name = "Unknown"
    print(f"Model: {model_name} ({n_features} features) on {len(df)} samples")

    alpha_preds = alpha_model.predict(df[features])
    df["alpha_signal"] = alpha_preds
    _build_alpha_derived_features(df, "alpha_signal")

    if use_dynamic_threshold:
        dyn = pd.Series(alpha_preds).rolling(window=1000, min_periods=100) \
            .quantile(0.90).fillna(0.60).values
    else:
        dyn = np.full(len(df), min_entry)

    print("Computing context features (turbulence + walk-forward HMM)...")
    turb_engine = MahalanobisTurbulence(window=TURB_WINDOW)
    risk_feats = ["log_return_feature", "volatility_24_feature", "intraday_range_feature"]
    available_risk = [c for c in risk_feats if c in df.columns]
    df["turbulence_score"] = turb_engine.compute(df, available_risk).bfill().fillna(5.0)
    df["turbulence_percentile"] = df["turbulence_score"].rolling(TURB_PCTILE_WIN).rank(pct=True)

    hmm_feats_all = ["log_return_feature", "volatility_24_feature", "aggressor_ratio",
                     "intraday_range_feature"]
    available_hmm = [c for c in hmm_feats_all if c in df.columns]
    df["hmm_state"] = _walk_forward_hmm(df, available_hmm)

    df["meta_pass"] = 1
    df["meta_prob"] = 1.0
    if meta_model is not None:
        print("Running Meta-Gatekeeper inference...")
        df["spread_bps"] = df["pr_spread_feature"] * 10000 if "pr_spread_feature" in df.columns else 0.0
        df["expected_cost_bps"] = 5 + df["spread_bps"] + 2

        meta_feats_df = pd.DataFrame({
            "alpha_prob":               df["alpha_prob"],
            "alpha_prob_smooth":        df["alpha_prob_smooth"],
            "alpha_prob_zscore":        df["alpha_prob_zscore"],
            "alpha_prob_percentile":    df["alpha_prob_percentile"],
            "alpha_signal_persistence": df["alpha_signal_persistence"],
            "turbulence_score":         df["turbulence_score"],
            "turbulence_percentile":    df["turbulence_percentile"],
            "hmm_state":                df["hmm_state"].astype(int),
            "volatility_24_feature":    df["volatility_24_feature"],
            "aggressor_ratio":          df.get("aggressor_ratio", 0.0),
            "l2_imbalance_feature":     df.get("l2_imbalance_feature", 0.0),
            "spread_bps":               df["spread_bps"],
            "expected_cost_bps":        df["expected_cost_bps"],
        }).fillna(0)

        df["meta_prob"] = meta_model.predict(meta_feats_df)
        df["meta_pass"] = (df["meta_prob"] >= meta_threshold).astype(int)

    # ─── Gate diagnostics ────────────────────────────────────────────────────
    print("\n=== Gate Diagnostics ===")
    n = len(df)
    smooth = df["alpha_prob_smooth"].values
    meta_ok = df["meta_pass"].values
    g_alpha_min = smooth > min_entry
    g_alpha_dyn = smooth > np.maximum(min_entry, dyn)
    g_meta = meta_ok == 1
    print(f"  bars total                          : {n}")
    print(f"  pass alpha > {min_entry:.2f}                  : "
          f"{g_alpha_min.sum()} ({g_alpha_min.mean():.1%})")
    print(f"  pass alpha > max({min_entry:.2f}, p90 dyn)    : "
          f"{g_alpha_dyn.sum()} ({g_alpha_dyn.mean():.1%})")
    print(f"  pass meta >= {meta_threshold:.2f}                   : "
          f"{g_meta.sum()} ({g_meta.mean():.1%})")
    print(f"  pass alpha_dyn AND meta             : "
          f"{(g_alpha_dyn & g_meta).sum()} ({(g_alpha_dyn & g_meta).mean():.1%})")
    if meta_model is not None:
        # how often meta vetoes a candidate the alpha passes
        veto_rate = (g_alpha_dyn & ~g_meta).sum() / max(g_alpha_dyn.sum(), 1)
        print(f"  meta veto rate on alpha passes      : {veto_rate:.1%}")

    # ─── Strategy simulation: Alpha + Meta + dynamic threshold ──────────────
    target_pos = np.zeros(n)
    cur_t = 0.0
    bars_since_entry = 0
    for i in range(n):
        thr = max(min_entry, dyn[i]) if use_dynamic_threshold else min_entry
        if cur_t == 0.0:
            if smooth[i] > thr and meta_ok[i] == 1:
                cur_t = 1.0
                bars_since_entry = 0
        else:
            bars_since_entry += 1
            if smooth[i] < exit_threshold and bars_since_entry >= min_hold_bars:
                cur_t = 0.0
        target_pos[i] = cur_t

    bar_return = df["close"].pct_change().fillna(0).values
    total_cost_rate = 0.0005 + 0.0002

    if "date" in df.columns:
        total_days = (df["date"].max() - df["date"].min()).total_seconds() / (24 * 3600)
    else:
        total_days = n / 17.6 / 24
    total_months = total_days / 30.44

    print("\n=== Strategy & Benchmarks ===")
    actual_pos, cum_strat, pnl_strat, td_strat = _simulate(target_pos, bar_return, total_cost_rate)
    summary = []
    summary.append(_summarize(f"Strategy ({model_name}+Meta)", cum_strat, pnl_strat, td_strat, total_months))

    # Buy & Hold
    bh_pos = np.ones(n)
    bh_pos[0] = 0
    _, cum_bh, pnl_bh, td_bh = _simulate(bh_pos, bar_return, total_cost_rate)
    summary.append(_summarize("Buy & Hold", cum_bh, pnl_bh, td_bh, total_months))

    # EMA200 long-only trend follower
    ema_fast = pd.Series(df["close"].values).ewm(span=50, adjust=False).mean().values
    ema_slow = pd.Series(df["close"].values).ewm(span=200, adjust=False).mean().values
    trend_long = (ema_fast > ema_slow).astype(float)
    _, cum_ema, pnl_ema, td_ema = _simulate(trend_long, bar_return, total_cost_rate)
    summary.append(_summarize("EMA50>EMA200 (trend long-only)", cum_ema, pnl_ema, td_ema, total_months))

    # Hybrid: trend base + meta gatekeeper veto in non-bull regimes
    if meta_model is not None:
        hybrid = trend_long.copy()
        # When trend is up but meta says no AND turbulence is high → reduce to 0
        veto = (df["turbulence_percentile"].fillna(0).values > 0.95) & (df["meta_pass"].values == 0)
        hybrid[veto] = 0.0
        _, cum_hyb, pnl_hyb, td_hyb = _simulate(hybrid, bar_return, total_cost_rate)
        summary.append(_summarize("Hybrid trend + meta veto (turb>p95)",
                                  cum_hyb, pnl_hyb, td_hyb, total_months))
    else:
        cum_hyb = None

    # ─── Plot ────────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    plt.figure(figsize=(13, 7))
    x = np.arange(n)
    plt.plot(x, cum_bh, label=f"Buy&Hold ({summary[1]['roi']:+.1%})",
             color="gray", linestyle="--", alpha=0.6)
    plt.plot(x, cum_ema, label=f"EMA50>EMA200 ({summary[2]['roi']:+.1%})",
             color="orange", alpha=0.8)
    if cum_hyb is not None:
        plt.plot(x, cum_hyb, label=f"Hybrid trend+meta ({summary[3]['roi']:+.1%})",
                 color="green", alpha=0.8)
    plt.plot(x, cum_strat, label=f"Strategy ({summary[0]['roi']:+.1%})",
             color="blue", linewidth=2)
    cum_max = np.maximum.accumulate(cum_strat)
    plt.fill_between(x, cum_strat, cum_max, color="red", alpha=0.15, label="Strat DD")

    plt.title(f"Global Alpha Replay: {Path(model_path).name}\n"
              f"Strategy ROI {summary[0]['roi']:+.2%} | Trades {summary[0]['trades']} | "
              f"Time-in-market {summary[0]['time_in_market']:.1%}")
    plt.xlabel("Dollar Bars")
    plt.ylabel("Cumulative Return")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.2)
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    plot_file = reports_dir / f"equity_{Path(model_path).stem}.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"\nEquity curves saved to {plot_file}")

    df["position"] = actual_pos
    df["cum_return"] = cum_strat
    df["cum_market"] = cum_bh
    output_path = f"cache/replay_{Path(model_path).stem}.feather"
    Path("cache").mkdir(exist_ok=True)
    df[["alpha_signal", "alpha_prob_smooth", "meta_prob", "meta_pass",
        "hmm_state", "turbulence_score", "position",
        "cum_return", "cum_market"]].reset_index().to_feather(output_path)
    print(f"Replay results saved to {output_path}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/dollar_alpha_v1/model_a_institutional.txt")
    parser.add_argument("--meta", type=str, default="models/meta_model_v1/gatekeeper.txt")
    parser.add_argument("--meta-threshold", type=float, default=0.60)
    parser.add_argument("--min-entry", type=float, default=0.55)
    parser.add_argument("--exit-threshold", type=float, default=0.50)
    parser.add_argument("--min-hold", type=int, default=50)
    parser.add_argument("--no-dynamic", action="store_true",
                        help="Disable rolling p90 dynamic threshold")
    parser.add_argument("--no-meta", action="store_true", help="Disable meta gatekeeper")
    args = parser.parse_args()

    data_path = "cache/dollar_bars_btc_2000000_features.feather"
    meta_path = None if args.no_meta else (args.meta if os.path.exists(args.meta) else None)

    if os.path.exists(args.model) and os.path.exists(data_path):
        run_replay(args.model, data_path,
                   meta_model_path=meta_path,
                   meta_threshold=args.meta_threshold,
                   min_entry=args.min_entry,
                   exit_threshold=args.exit_threshold,
                   min_hold_bars=args.min_hold,
                   use_dynamic_threshold=not args.no_dynamic)
    else:
        print(f"Error: Required files not found.\nModel: {args.model}\nData: {data_path}")
