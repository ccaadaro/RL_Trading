#!/usr/bin/env python3
"""
Phase 9 evaluation: LightGBM on 1h BTC dataset with two feature modes.

Two modes:
  --feature-set trend_only   : Jan 2021 → Mar 2026, trend features only
  --feature-set hybrid_micro : Jun 2023 → Mar 2026, trend + microstructure

Temporal splits (fixed — 2025 is "observed test", not clean holdout):
  trend_only : train=[2021-01,2024-01), val=[2024-01,2025-01), obs=[2025-01,2026-04)
  hybrid     : train=[2023-06,2024-01), val=[2024-01,2025-01), obs=[2025-01,2026-04)

Benchmarks per period: Cash, B&H/AlwaysLong, EMA-cross, HMA-trend,
                       Random-matched-TiM, Random-matched-count

Metrics: ROI, gross ROI, DD, Calmar, Sharpe, capture_ratio, TiM, turnover/month,
         trades/month, net/gross bps/trade, cost_drag, profit_factor, win_rate,
         avg_hold_h, rand_tim_p50, rand_tim_p95, rand_count_p95

Pass criteria:
  obs_roi > rand_tim_p95
  obs_calmar > bh_calmar
  obs_max_dd > bh_max_dd  (less negative)
  obs_capture_ratio in [0.3, 2.0]

Usage:
    python scripts/evaluate_1h_dataset.py --data cache/btc_1h_phase9.feather \
        --feature-set trend_only --output-dir reports/phase9
    python scripts/evaluate_1h_dataset.py --data cache/btc_1h_phase9.feather \
        --feature-set hybrid_micro --output-dir reports/phase9
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ─────────────────────────────────────────────────────────────────

COST_PER_SIDE_BPS = 7.0
N_RANDOM_TRIALS   = 300

MICRO_COLS = {
    "cvd_1h_feature", "cvd_4h_feature", "cvd_8h_feature",
    "cvd_slope_4h_feature", "cvd_zscore_24h_feature",
    "aggressor_ratio_1h_feature", "aggressor_ratio_4h_feature",
    "trade_count_1h_feature", "buy_ratio_1h_feature",
    "notional_1h_feature", "notional_zscore_24h_feature",
}

DATE_SPLITS = {
    "trend_only":   ("2021-01-01", "2024-01-01", "2025-01-01"),
    "hybrid_micro": ("2023-06-01", "2024-01-01", "2025-01-01"),
}

THRESHOLD_POLICIES = [0.60, 0.65, 0.70, "train_p80", "train_p90"]

ALL_TARGETS = [
    "target_trend_24h",
    "target_trend_48h",
    "target_trend_72h",
    "target_barrier_1.5tp_0.8sl_24h",
    "target_barrier_2.5tp_1.2sl_48h",
    "target_barrier_3.5tp_1.5sl_72h",
    "target_regime_48h",
    "target_regime_72h",
]


# ── LightGBM ──────────────────────────────────────────────────────────────────

def lgb_params() -> dict:
    return {
        "objective":        "binary",
        "metric":           "auc",
        "learning_rate":    0.03,
        "n_estimators":     300,
        "max_depth":        3,
        "num_leaves":       7,
        "min_child_samples": 100,
        "feature_fraction": 0.7,
        "subsample":        0.8,
        "subsample_freq":   1,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "random_state":     42,
        "verbose":          -1,
        "n_jobs":           -1,
    }


# ── HMA helper ────────────────────────────────────────────────────────────────

def _wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def _hma(s: pd.Series, n: int) -> pd.Series:
    return _wma(2.0 * _wma(s, max(n // 2, 1)) - _wma(s, n), max(int(round(n**0.5)), 1))


# ── Position generation ───────────────────────────────────────────────────────

def generate_position(
    probs: np.ndarray,
    entry_thr: float,
    exit_thr:  float,
    min_hold:  int,
) -> np.ndarray:
    """Hysteresis with minimum hold constraint."""
    n = len(probs)
    pos = np.zeros(n)
    in_pos, hold_cnt = False, 0
    for i in range(n):
        if not in_pos and probs[i] >= entry_thr:
            in_pos, hold_cnt = True, 0
        elif in_pos:
            hold_cnt += 1
            if hold_cnt >= min_hold and probs[i] < exit_thr:
                in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    return pos


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_equity(
    bar_returns:     np.ndarray,
    position:        np.ndarray,
    cost_per_side:   float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (equity, gross_equity, net_bar_ret).
    position[t] is set at bar-t close, effective at bar t+1.
    Costs charged on actual exposure changes.
    """
    n = len(bar_returns)
    gross = np.zeros(n)
    gross[1:] = position[:-1] * bar_returns[1:]
    delta = np.abs(np.diff(position, prepend=0.0))
    costs = delta * (cost_per_side / 10_000.0)
    net    = gross - costs
    return np.cumprod(1.0 + net), np.cumprod(1.0 + gross), net


# ── Per-trade statistics ──────────────────────────────────────────────────────

def trade_stats(position: np.ndarray, net_bar_ret: np.ndarray) -> dict:
    pos_diff = np.diff(position, prepend=0.0)
    entries  = np.where(pos_diff > 0)[0]
    exits    = np.where(pos_diff < 0)[0]

    if len(entries) == 0:
        return {"n_trades": 0, "win_rate": np.nan, "avg_holding_h": np.nan,
                "profit_factor": np.nan, "net_bps_per_trade": np.nan,
                "gross_bps_per_trade": np.nan}

    n = len(position)
    rets, durations = [], []
    for ep in entries:
        cands = exits[exits > ep]
        xp = cands[0] if len(cands) else n
        tr = float(np.prod(1.0 + net_bar_ret[ep:xp]) - 1.0)
        rets.append(tr)
        durations.append(float(xp - ep))

    rets = np.array(rets)
    durations = np.array(durations)
    pos_r = rets[rets > 0]; neg_r = rets[rets <= 0]
    pf = (float(np.sum(pos_r)) / float(np.sum(np.abs(neg_r)))) if len(neg_r) and np.sum(np.abs(neg_r)) > 0 else np.inf

    return {
        "n_trades":           len(entries),
        "win_rate":           float((rets > 0).mean()),
        "avg_holding_h":      float(durations.mean()),
        "profit_factor":      pf,
        "net_bps_per_trade":  float(np.mean(rets) * 10_000.0),
        "gross_bps_per_trade": float(np.mean(rets) * 10_000.0),  # refined below by caller
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def compute_metrics(
    equity:         np.ndarray,
    gross_equity:   np.ndarray,
    net_bar_ret:    np.ndarray,
    position:       np.ndarray,
    periods_per_year: float,
    bh_roi:         float,
    bh_calmar:      float,
    bh_max_dd:      float,
    label:          str = "",
) -> dict:
    n = len(equity)
    net_roi   = equity[-1] - 1.0
    gross_roi = gross_equity[-1] - 1.0
    max_dd    = _max_drawdown(equity)

    span_yr   = n / max(periods_per_year, 1.0)
    ann_ret   = (1.0 + net_roi) ** (1.0 / max(span_yr, 1e-6)) - 1.0
    calmar    = ann_ret / abs(max_dd) if abs(max_dd) > 1e-9 else (np.inf if ann_ret > 0 else -np.inf)

    std_net   = np.std(net_bar_ret)
    sharpe    = float(np.mean(net_bar_ret) / std_net * np.sqrt(periods_per_year)) if std_net > 1e-12 else 0.0

    capture   = net_roi / bh_roi if abs(bh_roi) > 1e-9 else np.nan
    tim       = float(position.mean())
    n_months  = n / max(periods_per_year / 12.0, 1.0)
    turnover  = float(np.sum(np.abs(np.diff(position, prepend=0.0))))
    turnover_monthly = turnover / max(n_months, 1.0)

    ts = trade_stats(position, net_bar_ret)

    # Refine gross bps/trade using actual gross equity
    if ts["n_trades"] > 0:
        ts["gross_bps_per_trade"] = gross_roi / ts["n_trades"] * 10_000.0

    trades_per_month = ts["n_trades"] / max(n_months, 1.0)
    cost_drag        = (gross_roi - net_roi) * 100.0

    return {
        "label":               label,
        "roi_pct":             round(net_roi * 100, 3),
        "gross_roi_pct":       round(gross_roi * 100, 3),
        "max_dd_pct":          round(max_dd * 100, 3),
        "calmar":              round(calmar, 4),
        "sharpe":              round(sharpe, 4),
        "capture_ratio":       round(capture, 4) if not np.isnan(capture) else np.nan,
        "tim_pct":             round(tim * 100, 2),
        "turnover_monthly":    round(turnover_monthly, 3),
        "trades_per_month":    round(trades_per_month, 3),
        "net_bps_per_trade":   round(ts["net_bps_per_trade"], 1) if not np.isnan(ts["net_bps_per_trade"]) else np.nan,
        "gross_bps_per_trade": round(ts["gross_bps_per_trade"], 1) if not np.isnan(ts["gross_bps_per_trade"]) else np.nan,
        "cost_drag_pct":       round(cost_drag, 3),
        "profit_factor":       round(ts["profit_factor"], 3) if not np.isinf(ts["profit_factor"]) else 999.0,
        "win_rate":            round(ts["win_rate"], 4) if not np.isnan(ts["win_rate"]) else np.nan,
        "avg_holding_h":       round(ts["avg_holding_h"], 1) if not np.isnan(ts["avg_holding_h"]) else np.nan,
        "n_trades":            ts["n_trades"],
        # Filled later by caller
        "rand_tim_p50":        np.nan,
        "rand_tim_p95":        np.nan,
        "rand_count_p95":      np.nan,
    }


# ── Random benchmarks ─────────────────────────────────────────────────────────

def random_benchmark_tim(
    bar_returns:   np.ndarray,
    tim:           float,
    cost_per_side: float,
    rng:           np.random.Generator,
    n_trials:      int = N_RANDOM_TRIALS,
) -> tuple[float, float]:
    """Vectorized TiM-matched random. Returns (P50, P95) of net ROI."""
    n = len(bar_returns)
    c = cost_per_side / 10_000.0
    pos = (rng.random((n_trials, n)) < tim).astype(float)
    pos_prev = np.hstack([np.zeros((n_trials, 1)), pos[:, :-1]])
    gross = pos_prev * bar_returns[np.newaxis, :]
    delta = np.abs(np.diff(pos, prepend=np.zeros((n_trials, 1)), axis=1))
    net   = gross - delta * c
    roi   = np.cumprod(1.0 + net, axis=1)[:, -1] - 1.0
    return float(np.percentile(roi, 50)), float(np.percentile(roi, 95))


def random_benchmark_count(
    bar_returns:   np.ndarray,
    n_trades:      int,
    avg_hold:      float,
    cost_per_side: float,
    rng:           np.random.Generator,
    n_trials:      int = N_RANDOM_TRIALS,
) -> tuple[float, float]:
    """Trade-count-matched random benchmark. Returns (P50, P95) of net ROI."""
    n = len(bar_returns)
    if n_trades == 0:
        return 0.0, 0.0
    c    = cost_per_side / 10_000.0
    hold = max(int(round(avg_hold)), 1)
    roi_list = []
    for _ in range(n_trials):
        pos = np.zeros(n)
        # sample entry points uniformly, eliminating overlaps greedily
        candidates = rng.permutation(max(n - hold, 1))
        prev_end = -1
        placed = 0
        for ep in candidates:
            if placed >= n_trades:
                break
            if ep > prev_end:
                end = min(ep + hold, n)
                pos[ep:end] = 1.0
                prev_end = end
                placed += 1
        pos_prev = np.roll(pos, 1); pos_prev[0] = 0.0
        gross = pos_prev * bar_returns
        delta = np.abs(np.diff(pos, prepend=0.0))
        net   = gross - delta * c
        roi_list.append(float(np.cumprod(1.0 + net)[-1] - 1.0))
    arr = np.array(roi_list)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


# ── Benchmark signals ─────────────────────────────────────────────────────────

def ema_cross_position(close: pd.Series, fast: int = 12, slow: int = 48) -> np.ndarray:
    """Long when EMA(fast) > EMA(slow)."""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    return (ema_f > ema_s).astype(float).to_numpy()


def hma_trend_position(close: pd.Series, n: int = 48) -> np.ndarray:
    """Long when HMA(n) is rising (positive 6-bar slope)."""
    hma = _hma(close, n)
    slope = hma.diff(6)
    return (slope > 0).astype(float).to_numpy()


# ── Core per-target evaluation ────────────────────────────────────────────────

def infer_horizon_h(target: str) -> int:
    for part in reversed(target.split("_")):
        if part.endswith("h"):
            try:
                return int(part[:-1])
            except ValueError:
                continue
    return 24


def evaluate_target(
    df:              pd.DataFrame,
    target:          str,
    feature_cols:    list[str],
    feature_set:     str,
    cost_bps:        float,
    periods_per_year: float,
    out_dir:         Path,
    rng:             np.random.Generator,
) -> list[dict]:
    """
    Full evaluation for one target. Returns list of result dicts,
    one per threshold policy (best_val flag marks the selected one).
    """
    horizon_h = infer_horizon_h(target)
    min_hold  = max(horizon_h // 3, 1)

    train_start, val_start, obs_start = DATE_SPLITS[feature_set]
    train_start = pd.Timestamp(train_start)
    val_start   = pd.Timestamp(val_start)
    obs_start   = pd.Timestamp(obs_start)

    # ── Split ──────────────────────────────────────────────────────────────
    df_tr  = df[(df["date"] >= train_start) & (df["date"] < val_start)].dropna(subset=feature_cols + [target]).reset_index(drop=True)
    df_val = df[(df["date"] >= val_start)   & (df["date"] < obs_start)].dropna(subset=feature_cols + [target]).reset_index(drop=True)
    df_obs = df[(df["date"] >= obs_start)].dropna(subset=feature_cols + [target]).reset_index(drop=True)

    if len(df_tr) < 200 or len(df_val) < 100 or len(df_obs) < 100:
        print(f"    [SKIP] {target}: insufficient data (tr={len(df_tr)}, val={len(df_val)}, obs={len(df_obs)})")
        return []

    print(f"  {target}  tr={len(df_tr):,} val={len(df_val):,} obs={len(df_obs):,}  horizon={horizon_h}h  min_hold={min_hold}")

    # ── B&H baseline for each period ──────────────────────────────────────
    def _bh(df_period: pd.DataFrame, cost: float) -> tuple[float, float, float]:
        ret = df_period["close"].pct_change().fillna(0.0).to_numpy()
        pos = np.ones(len(ret))
        eq, _, _ = simulate_equity(ret, pos, cost)
        roi = eq[-1] - 1.0
        dd  = _max_drawdown(eq)
        n   = len(ret)
        sy  = n / max(periods_per_year, 1.0)
        ann = (1.0 + roi) ** (1.0 / max(sy, 1e-6)) - 1.0
        cal = ann / abs(dd) if abs(dd) > 1e-9 else np.inf
        return roi, dd, cal

    bh_val_roi,  bh_val_dd,  bh_val_cal  = _bh(df_val, cost_bps)
    bh_obs_roi,  bh_obs_dd,  bh_obs_cal  = _bh(df_obs, cost_bps)

    # ── Train model ────────────────────────────────────────────────────────
    X_tr,  y_tr  = df_tr[feature_cols].values,  df_tr[target].values
    X_val, y_val = df_val[feature_cols].values, df_val[target].values
    X_obs, y_obs = df_obs[feature_cols].values, df_obs[target].values

    model = lgb.LGBMClassifier(**lgb_params())
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    val_probs = model.predict_proba(X_val)[:, 1]
    obs_probs = model.predict_proba(X_obs)[:, 1]

    try:
        val_auc = float(roc_auc_score(y_val, val_probs))
        obs_auc = float(roc_auc_score(y_obs, obs_probs))
        val_brier = float(brier_score_loss(y_val, val_probs))
        obs_brier = float(brier_score_loss(y_obs, obs_probs))
    except ValueError:
        val_auc = obs_auc = val_brier = obs_brier = np.nan

    # ── Feature importance ─────────────────────────────────────────────────
    fi_path = out_dir / "phase9_1h_feature_importance"
    fi_path.mkdir(parents=True, exist_ok=True)
    fi_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(fi_path / f"{feature_set}_{target}.csv", index=False)

    # ── Threshold policies ─────────────────────────────────────────────────
    train_quantiles = {
        "train_p80": float(np.quantile(model.predict_proba(X_tr)[:, 1], 0.80)),
        "train_p90": float(np.quantile(model.predict_proba(X_tr)[:, 1], 0.90)),
    }
    policies = []
    for tp in THRESHOLD_POLICIES:
        entry_thr = train_quantiles[tp] if isinstance(tp, str) else float(tp)
        exit_thr  = 0.50
        policies.append((str(tp), entry_thr, exit_thr))

    # ── Bar returns for each period ────────────────────────────────────────
    val_ret = df_val["close"].pct_change().fillna(0.0).to_numpy()
    obs_ret = df_obs["close"].pct_change().fillna(0.0).to_numpy()

    # ── Compute all benchmarks (once) ──────────────────────────────────────
    def _bench_row(pos: np.ndarray, bar_ret: np.ndarray, bh_roi: float,
                   bh_calmar: float, bh_max_dd: float, name: str) -> dict:
        eq, geq, net = simulate_equity(bar_ret, pos, cost_bps)
        m = compute_metrics(eq, geq, net, pos, periods_per_year,
                            bh_roi, bh_calmar, bh_max_dd, label=name)
        return m

    val_close = df_val["close"]
    obs_close = df_obs["close"]

    # EMA and HMA signals: computed on the full df to avoid warmup, then sliced
    ema_pos_full = ema_cross_position(df["close"], 12, 48)
    hma_pos_full = hma_trend_position(df["close"], 48)

    val_idx = df[(df["date"] >= val_start) & (df["date"] < obs_start)].index
    obs_idx = df[df["date"] >= obs_start].index

    def _align(pos_full: np.ndarray, idx) -> np.ndarray:
        # idx are positions in the full df; align to len of the period df
        period_len = len(idx)
        full_positions = pos_full[idx - df.index[0]]
        return full_positions[:period_len]

    ema_val = _align(ema_pos_full, val_idx)
    ema_obs = _align(ema_pos_full, obs_idx)
    hma_val = _align(hma_pos_full, val_idx)
    hma_obs = _align(hma_pos_full, obs_idx)

    bh_pos_val = np.ones(len(val_ret))
    bh_pos_obs = np.ones(len(obs_ret))

    benchmarks_val = {
        "cash":     _bench_row(np.zeros(len(val_ret)), val_ret, bh_val_roi, bh_val_cal, bh_val_dd, "cash"),
        "bh":       _bench_row(bh_pos_val,             val_ret, bh_val_roi, bh_val_cal, bh_val_dd, "buy_hold"),
        "ema_cross": _bench_row(ema_val[:len(val_ret)], val_ret, bh_val_roi, bh_val_cal, bh_val_dd, "ema_cross"),
        "hma_trend": _bench_row(hma_val[:len(val_ret)], val_ret, bh_val_roi, bh_val_cal, bh_val_dd, "hma_trend"),
    }
    benchmarks_obs = {
        "cash":      _bench_row(np.zeros(len(obs_ret)), obs_ret, bh_obs_roi, bh_obs_cal, bh_obs_dd, "cash"),
        "bh":        _bench_row(bh_pos_obs,             obs_ret, bh_obs_roi, bh_obs_cal, bh_obs_dd, "buy_hold"),
        "ema_cross": _bench_row(ema_obs[:len(obs_ret)], obs_ret, bh_obs_roi, bh_obs_cal, bh_obs_dd, "ema_cross"),
        "hma_trend": _bench_row(hma_obs[:len(obs_ret)], obs_ret, bh_obs_roi, bh_obs_cal, bh_obs_dd, "hma_trend"),
    }

    # ── Evaluate each threshold policy ─────────────────────────────────────
    ec_dir = out_dir / "phase9_1h_equity_curves"
    ec_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for policy_name, entry_thr, exit_thr in policies:
        val_pos = generate_position(val_probs, entry_thr, exit_thr, min_hold)
        obs_pos = generate_position(obs_probs, entry_thr, exit_thr, min_hold)

        val_eq, val_geq, val_net = simulate_equity(val_ret, val_pos, cost_bps)
        obs_eq, obs_geq, obs_net = simulate_equity(obs_ret, obs_pos, cost_bps)

        vm = compute_metrics(val_eq, val_geq, val_net, val_pos, periods_per_year,
                             bh_val_roi, bh_val_cal, bh_val_dd, label="model")
        om = compute_metrics(obs_eq, obs_geq, obs_net, obs_pos, periods_per_year,
                             bh_obs_roi, bh_obs_cal, bh_obs_dd, label="model")

        all_rows.append({
            "feature_set":      feature_set,
            "target":           target,
            "threshold_policy": policy_name,
            "entry_thr":        round(entry_thr, 4),
            "exit_thr":         round(exit_thr, 4),
            "min_hold_h":       min_hold,
            # Val
            "val_roi_pct":          vm["roi_pct"],
            "val_gross_roi_pct":    vm["gross_roi_pct"],
            "val_max_dd_pct":       vm["max_dd_pct"],
            "val_calmar":           vm["calmar"],
            "val_sharpe":           vm["sharpe"],
            "val_capture_ratio":    vm["capture_ratio"],
            "val_tim_pct":          vm["tim_pct"],
            "val_turnover_monthly": vm["turnover_monthly"],
            "val_trades_per_month": vm["trades_per_month"],
            "val_net_bps_per_trade":   vm["net_bps_per_trade"],
            "val_gross_bps_per_trade": vm["gross_bps_per_trade"],
            "val_cost_drag_pct":    vm["cost_drag_pct"],
            "val_profit_factor":    vm["profit_factor"],
            "val_win_rate":         vm["win_rate"],
            "val_avg_holding_h":    vm["avg_holding_h"],
            "val_n_trades":         vm["n_trades"],
            "val_rand_tim_p50":     np.nan,
            "val_rand_tim_p95":     np.nan,
            "val_rand_count_p95":   np.nan,
            # Obs
            "obs_roi_pct":          om["roi_pct"],
            "obs_gross_roi_pct":    om["gross_roi_pct"],
            "obs_max_dd_pct":       om["max_dd_pct"],
            "obs_calmar":           om["calmar"],
            "obs_sharpe":           om["sharpe"],
            "obs_capture_ratio":    om["capture_ratio"],
            "obs_tim_pct":          om["tim_pct"],
            "obs_turnover_monthly": om["turnover_monthly"],
            "obs_trades_per_month": om["trades_per_month"],
            "obs_net_bps_per_trade":   om["net_bps_per_trade"],
            "obs_gross_bps_per_trade": om["gross_bps_per_trade"],
            "obs_cost_drag_pct":    om["cost_drag_pct"],
            "obs_profit_factor":    om["profit_factor"],
            "obs_win_rate":         om["win_rate"],
            "obs_avg_holding_h":    om["avg_holding_h"],
            "obs_n_trades":         om["n_trades"],
            "obs_rand_tim_p50":     np.nan,
            "obs_rand_tim_p95":     np.nan,
            "obs_rand_count_p95":   np.nan,
            # Model diagnostics
            "val_auc":       round(val_auc, 5),
            "obs_auc":       round(obs_auc, 5),
            "val_brier":     round(val_brier, 5),
            "obs_brier":     round(obs_brier, 5),
            "val_pred_mean": round(float(val_probs.mean()), 5),
            "val_pred_p90":  round(float(np.quantile(val_probs, 0.90)), 5),
            "obs_pred_mean": round(float(obs_probs.mean()), 5),
            "obs_pred_p90":  round(float(np.quantile(obs_probs, 0.90)), 5),
            "val_pos_rate":  round(float(y_val.mean()), 5),
            "obs_pos_rate":  round(float(y_obs.mean()), 5),
            "is_best_val":   False,
            "obs_pass":      "",
            # B&H reference
            "bh_val_roi_pct":  round(bh_val_roi * 100, 3),
            "bh_val_dd_pct":   round(bh_val_dd * 100, 3),
            "bh_val_calmar":   round(bh_val_cal, 4),
            "bh_obs_roi_pct":  round(bh_obs_roi * 100, 3),
            "bh_obs_dd_pct":   round(bh_obs_dd * 100, 3),
            "bh_obs_calmar":   round(bh_obs_cal, 4),
            # Equity curves saved below
            "_val_eq": val_eq, "_val_pos": val_pos, "_val_dates": df_val["date"].values,
            "_obs_eq": obs_eq, "_obs_pos": obs_pos, "_obs_dates": df_obs["date"].values,
        })

    # ── Select best val config (max val_calmar, tie-break val_roi) ─────────
    val_metrics = [(r["val_calmar"] if r["val_n_trades"] > 0 else -np.inf,
                    r["val_roi_pct"], i)
                   for i, r in enumerate(all_rows)]
    best_idx = max(val_metrics, key=lambda x: (x[0], x[1]))[2]
    all_rows[best_idx]["is_best_val"] = True
    best_row = all_rows[best_idx]

    print(f"    best_val={best_row['threshold_policy']}  "
          f"val_cal={best_row['val_calmar']:.3f} val_roi={best_row['val_roi_pct']:+.1f}%  "
          f"obs_roi={best_row['obs_roi_pct']:+.1f}%  obs_dd={best_row['obs_max_dd_pct']:.1f}%  "
          f"auc_obs={best_row['obs_auc']:.4f}")

    # ── Random benchmarks for best val config ──────────────────────────────
    for i, row in enumerate(all_rows):
        for period, bar_ret, probs_arr in [("val", val_ret, val_probs), ("obs", obs_ret, obs_probs)]:
            pos = generate_position(probs_arr, row["entry_thr"], row["exit_thr"], min_hold)
            tim = pos.mean()
            ts  = trade_stats(pos, simulate_equity(bar_ret, pos, cost_bps)[2])

            rt50, rt95 = random_benchmark_tim(bar_ret, tim, cost_bps, rng)
            rc_p50, rc_p95 = random_benchmark_count(
                bar_ret, ts["n_trades"],
                ts["avg_holding_h"] if not np.isnan(ts["avg_holding_h"]) else 24.0,
                cost_bps, rng
            )
            row[f"{period}_rand_tim_p50"]   = round(rt50 * 100, 3)
            row[f"{period}_rand_tim_p95"]   = round(rt95 * 100, 3)
            row[f"{period}_rand_count_p95"] = round(rc_p95 * 100, 3)

    # ── Pass/fail for each row (obs period) ───────────────────────────────
    for row in all_rows:
        flags = []
        if row["obs_roi_pct"] > row["obs_rand_tim_p95"]:   flags.append("ROI>randP95")
        if row["obs_calmar"]  > row["bh_obs_calmar"]:       flags.append("Calmar>BH")
        if row["obs_max_dd_pct"] > row["bh_obs_dd_pct"]:   flags.append("DD<BH")
        capture = row["obs_capture_ratio"]
        if not np.isnan(capture) and 0.3 <= capture <= 2.0: flags.append("capture_ok")
        row["obs_pass"] = "+".join(flags) if flags else "FAIL"
        # Clean internal equity curve data from row dict before saving
        # (kept temporarily for equity curve export below)

    # ── Save equity curves (best val config only) ──────────────────────────
    br = best_row
    for period, dates, eq, pos in [
        ("val", br["_val_dates"], br["_val_eq"], br["_val_pos"]),
        ("obs", br["_obs_dates"], br["_obs_eq"], br["_obs_pos"]),
    ]:
        # Also compute B&H equity curve for comparison
        bar_ret  = val_ret if period == "val" else obs_ret
        bh_eq, _, _ = simulate_equity(bar_ret, np.ones(len(bar_ret)), cost_bps)
        ec_df = pd.DataFrame({
            "date":     dates,
            "equity":   eq,
            "bh_equity": bh_eq,
            "position": pos,
        })
        ec_df.to_csv(ec_dir / f"{feature_set}_{target}_{period}.csv", index=False)

    # ── Save benchmark table ───────────────────────────────────────────────
    bench_rows = []
    for bname, bm_val in benchmarks_val.items():
        bm_obs = benchmarks_obs[bname]
        bench_rows.append({
            "feature_set": feature_set, "target": target, "benchmark": bname,
            "val_roi_pct":  bm_val["roi_pct"],  "val_max_dd_pct": bm_val["max_dd_pct"],
            "val_calmar":   bm_val["calmar"],    "val_sharpe":     bm_val["sharpe"],
            "val_tim_pct":  bm_val["tim_pct"],   "val_turnover_monthly": bm_val["turnover_monthly"],
            "obs_roi_pct":  bm_obs["roi_pct"],   "obs_max_dd_pct": bm_obs["max_dd_pct"],
            "obs_calmar":   bm_obs["calmar"],     "obs_sharpe":     bm_obs["sharpe"],
            "obs_tim_pct":  bm_obs["tim_pct"],   "obs_turnover_monthly": bm_obs["turnover_monthly"],
        })

    # ── Strip private keys before returning ───────────────────────────────
    for row in all_rows:
        for k in ["_val_eq", "_val_pos", "_val_dates", "_obs_eq", "_obs_pos", "_obs_dates"]:
            row.pop(k, None)

    return all_rows, bench_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        required=True, help="Path to btc_1h_phase9.feather")
    parser.add_argument("--feature-set", required=True, choices=["trend_only", "hybrid_micro"])
    parser.add_argument("--output-dir",  default="reports/phase9")
    parser.add_argument("--targets",     nargs="+", default=ALL_TARGETS)
    parser.add_argument("--cost-bps",    type=float, default=COST_PER_SIDE_BPS)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    df = feather.read_feather(args.data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df):,} rows  {df['date'].iloc[0]} → {df['date'].iloc[-1]}")

    # ── Feature set ───────────────────────────────────────────────────────
    all_feat_cols = sorted(c for c in df.columns if c.endswith("_feature"))
    if args.feature_set == "trend_only":
        feature_cols = [c for c in all_feat_cols if c not in MICRO_COLS]
        # Use full date range (start filter already baked into dataset)
    else:
        feature_cols = all_feat_cols
        # Trim dataset to micro-available period
        df = df[df["date"] >= pd.Timestamp(DATE_SPLITS["hybrid_micro"][0])].reset_index(drop=True)
        print(f"hybrid_micro: trimmed to {len(df):,} rows from {df['date'].iloc[0]}")

    print(f"Feature set: {args.feature_set}  ({len(feature_cols)} features)")

    # ── Periods per year ──────────────────────────────────────────────────
    span_days = (df["date"].iloc[-1] - df["date"].iloc[0]).total_seconds() / 86400.0
    periods_per_year = len(df) / max(span_days, 1.0) * 365.25
    print(f"periods_per_year ≈ {periods_per_year:.0f}")

    rng = np.random.default_rng(42)

    all_model_rows  = []
    all_bench_rows  = []

    for target in args.targets:
        if target not in df.columns:
            print(f"  [SKIP] {target} not in dataset")
            continue
        print(f"\n── {target} ──")
        result = evaluate_target(
            df, target, feature_cols, args.feature_set,
            args.cost_bps, periods_per_year, out_dir, rng
        )
        if result:
            model_rows, bench_rows = result
            all_model_rows.extend(model_rows)
            all_bench_rows.extend(bench_rows)

    if not all_model_rows:
        print("No results produced.")
        return

    # ── Save outputs ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_model_rows)
    bench_df   = pd.DataFrame(all_bench_rows)

    results_path = out_dir / f"phase9_1h_evaluation_{args.feature_set}.csv"
    bench_path   = out_dir / f"phase9_1h_benchmarks_{args.feature_set}.csv"
    results_df.to_csv(results_path, index=False)
    bench_df.to_csv(bench_path, index=False)

    # Summary: best val config per target
    best_df = results_df[results_df["is_best_val"]].copy()
    summary_cols = [
        "feature_set", "target", "threshold_policy",
        "val_roi_pct",  "val_max_dd_pct",  "val_calmar",  "val_rand_tim_p95",  "val_n_trades",
        "obs_roi_pct",  "obs_max_dd_pct",  "obs_calmar",  "obs_rand_tim_p95",  "obs_n_trades",
        "obs_capture_ratio", "bh_obs_roi_pct", "bh_obs_calmar",
        "val_auc", "obs_auc", "obs_pass",
    ]
    summary_df = best_df[[c for c in summary_cols if c in best_df.columns]]
    summary_path = out_dir / f"phase9_1h_summary_{args.feature_set}.csv"
    summary_df.to_csv(summary_path, index=False)

    # ── Print summary table ────────────────────────────────────────────────
    print(f"\n{'═'*120}")
    print(f"PHASE 9 SUMMARY — {args.feature_set.upper()}")
    print(f"{'═'*120}")
    fmt = "{:<38s} {:>6s} {:>7s} {:>7s} {:>7s} {:>7s}   {:>7s} {:>7s} {:>7s} {:>7s}   {:>7s} {:>5s} {}"
    print(fmt.format(
        "target", "thr", "v_roi%","v_dd%","v_cal","v_rnd95",
        "o_roi%","o_dd%","o_cal","o_rnd95",
        "capture","auc","pass"
    ))
    print("─" * 120)
    for _, r in summary_df.iterrows():
        print(fmt.format(
            r["target"],
            str(r["threshold_policy"]),
            f"{r['val_roi_pct']:+.1f}", f"{r['val_max_dd_pct']:.1f}",
            f"{r['val_calmar']:.2f}", f"{r['val_rand_tim_p95']:+.1f}",
            f"{r['obs_roi_pct']:+.1f}", f"{r['obs_max_dd_pct']:.1f}",
            f"{r['obs_calmar']:.2f}", f"{r['obs_rand_tim_p95']:+.1f}",
            f"{r['obs_capture_ratio']:.2f}" if not pd.isna(r['obs_capture_ratio']) else "  nan",
            f"{r['obs_auc']:.4f}",
            str(r["obs_pass"]),
        ))

    print(f"\nSaved: {results_path}")
    print(f"Saved: {bench_path}")
    print(f"Saved: {summary_path}")
    print(f"Equity curves → {out_dir}/phase9_1h_equity_curves/")
    print(f"Feature importance → {out_dir}/phase9_1h_feature_importance/")


if __name__ == "__main__":
    main()
