#!/usr/bin/env python3
"""
Evaluate BTC v2 economic targets.

This is intentionally a research gate, not a deployment trainer. It trains
LightGBM on target_tradeable_* labels and scores the resulting long-only policy
on net bar returns, drawdown, exposure, turnover, and fold stability.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from sklearn.metrics import accuracy_score, roc_auc_score


@dataclass
class EvalResult:
    dataset: str
    target: str
    horizon_bars: int
    fold: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    train_rows: int
    val_rows: int
    test_rows: int
    threshold: float
    auc: float
    accuracy: float
    base_rate: float
    val_net_return_pct: float
    test_net_return_pct: float
    buy_hold_pct: float
    excess_vs_bh_pct: float
    sharpe: float
    max_drawdown_pct: float
    entries: int
    time_in_market_pct: float
    turnover: float
    shuffle_net_return_pct: float
    signal_lift_pct: float


def lgb_params() -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 600,
        "learning_rate": 0.01,
        "max_depth": 5,
        "num_leaves": 24,
        "min_child_samples": 50,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / np.maximum(peak, 1e-12) - 1.0).min())


def simulate(
    bar_returns: pd.Series,
    costs_bps: pd.Series,
    probs: np.ndarray,
    threshold: float,
    periods_per_year: float,
) -> dict:
    position = (probs >= threshold).astype(float)
    gross = np.zeros(len(position))
    r = bar_returns.fillna(0.0).to_numpy()
    gross[1:] = position[:-1] * r[1:]
    turnover = np.abs(np.diff(position, prepend=0.0))
    costs = turnover * (costs_bps.fillna(10.0).to_numpy() / 10_000.0)
    net = gross - costs
    equity = np.cumprod(1.0 + net)
    sharpe = 0.0
    if np.std(net) > 1e-12:
        sharpe = float(np.mean(net) / np.std(net) * np.sqrt(periods_per_year))
    return {
        "net_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity) if len(equity) else 0.0,
        "entries": int(np.sum((position == 1.0) & (np.roll(position, 1) == 0.0))),
        "time_in_market": float(position.mean()) if len(position) else 0.0,
        "turnover": float(turnover.sum()),
    }


def simulate_hysteresis(
    bar_returns: pd.Series,
    costs_bps: pd.Series,
    probs: np.ndarray,
    entry_thr: float,
    exit_thr: float,
    periods_per_year: float,
) -> dict:
    position = np.zeros(len(probs))
    in_pos = False
    for i, p in enumerate(probs):
        if not in_pos and p >= entry_thr:
            in_pos = True
        elif in_pos and p < exit_thr:
            in_pos = False
        position[i] = 1.0 if in_pos else 0.0
    gross = np.zeros(len(position))
    r = bar_returns.fillna(0.0).to_numpy()
    gross[1:] = position[:-1] * r[1:]
    turnover = np.abs(np.diff(position, prepend=0.0))
    costs = turnover * (costs_bps.fillna(10.0).to_numpy() / 10_000.0)
    net = gross - costs
    equity = np.cumprod(1.0 + net)
    sharpe = 0.0
    if np.std(net) > 1e-12:
        sharpe = float(np.mean(net) / np.std(net) * np.sqrt(periods_per_year))
    return {
        "net_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity) if len(equity) else 0.0,
        "entries": int(np.sum((position == 1.0) & (np.roll(position, 1) == 0.0))),
        "time_in_market": float(position.mean()) if len(position) else 0.0,
        "turnover": float(turnover.sum()),
    }


def infer_horizon(target: str) -> int:
    return int(target.removeprefix("target_tradeable_").removesuffix("b"))


def choose_threshold(
    val_returns: pd.Series,
    val_costs_bps: pd.Series,
    probs: np.ndarray,
    periods_per_year: float,
) -> tuple[float, dict]:
    thresholds = np.arange(0.50, 0.65, 0.025)
    scored = [
        (simulate(val_returns, val_costs_bps, probs, float(t), periods_per_year)["net_return"], float(t))
        for t in thresholds
    ]
    _, best = max(scored, key=lambda row: row[0])
    return best, simulate(val_returns, val_costs_bps, probs, best, periods_per_year)


def choose_hysteresis(
    val_returns: pd.Series,
    val_costs_bps: pd.Series,
    probs: np.ndarray,
    periods_per_year: float,
) -> tuple[float, float, dict]:
    entry_thrs = np.arange(0.50, 0.65, 0.025)
    exit_thrs  = np.arange(0.35, 0.55, 0.025)
    best_net, best_entry, best_exit = -np.inf, 0.55, 0.45
    for entry in entry_thrs:
        for exit_ in exit_thrs:
            if exit_ >= entry:
                continue
            r = simulate_hysteresis(val_returns, val_costs_bps, probs, entry, exit_, periods_per_year)["net_return"]
            if r > best_net:
                best_net, best_entry, best_exit = r, float(entry), float(exit_)
    return best_entry, best_exit, simulate_hysteresis(
        val_returns, val_costs_bps, probs, best_entry, best_exit, periods_per_year
    )


def time_folds(df: pd.DataFrame, folds: int) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    n = len(df)
    fold_size = n // (folds + 2)
    out = []
    for i in range(folds):
        train_end = fold_size * (i + 2)
        val_end = train_end + fold_size
        test_end = val_end + fold_size
        if test_end > n:
            test_end = n
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:test_end]
        if min(len(train), len(val), len(test)) > 500:
            out.append((train, val, test))
    return out


def evaluate_target(df: pd.DataFrame, dataset: str, target: str, folds: int) -> list[EvalResult]:
    horizon = infer_horizon(target)
    ret_col = "bar_net_return"
    if ret_col not in df.columns:
        # One-bar return net of proportional position changes is already
        # accounted for in labels; policy simulation uses bar price return.
        df = df.copy()
        df[ret_col] = df["close"].pct_change().fillna(0.0)

    blocked_prefixes = ("target_tradeable_", "future_return_", "net_return_", "net_risk_adj_")
    feature_cols = [
        c for c in df.columns
        if c.endswith("_feature") and not c.startswith(blocked_prefixes)
    ]
    feature_cols += [
        c for c in [
            "cvd_4h_sum_trade_feature", "aggressor_ratio_4h_mean_trade_feature",
            "whale_trades_4h_sum_trade_feature", "large_trades_4h_sum_trade_feature",
            "max_trade_usd_4h_max_trade_feature", "vwap_skew_4h_mean_trade_feature",
            "estimated_oneway_cost_bps", "estimated_roundtrip_cost_bps",
        ]
        if c in df.columns and c not in feature_cols
    ]
    feature_cols = sorted(set(feature_cols))

    span_days = max((df["date"].iloc[-1] - df["date"].iloc[0]).total_seconds() / 86400.0, 1.0)
    periods_per_year = len(df) / span_days * 365.25
    results = []
    for fold_idx, (train, val, test) in enumerate(time_folds(df, folds), start=1):
        model = lgb.LGBMClassifier(**lgb_params())
        model.fit(
            train[feature_cols],
            train[target],
            eval_set=[(val[feature_cols], val[target])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        val_probs = model.predict_proba(val[feature_cols])[:, 1]
        test_probs = model.predict_proba(test[feature_cols])[:, 1]
        threshold, val_sim = choose_threshold(
            val[ret_col], val["estimated_oneway_cost_bps"], val_probs, periods_per_year
        )
        test_sim = simulate(
            test[ret_col], test["estimated_oneway_cost_bps"], test_probs, threshold, periods_per_year
        )
        # Hysteresis: entry/exit thresholds tuned on val, applied to test
        hys_entry, hys_exit, _ = choose_hysteresis(
            val[ret_col], val["estimated_oneway_cost_bps"], val_probs, periods_per_year
        )
        hys_sim = simulate_hysteresis(
            test[ret_col], test["estimated_oneway_cost_bps"], test_probs,
            hys_entry, hys_exit, periods_per_year
        )
        buy_hold = test["close"].iloc[-1] / test["close"].iloc[0] - 1.0
        try:
            auc = float(roc_auc_score(test[target], test_probs))
        except ValueError:
            auc = 0.5

        # Shuffle control: same hysteresis params, random probs — isolates whether
        # outperformance is from model signal or just from the holding duration
        rng = np.random.default_rng(42)
        shuffled_probs = rng.permutation(test_probs)
        shuffle_sim = simulate_hysteresis(
            test[ret_col], test["estimated_oneway_cost_bps"], shuffled_probs,
            hys_entry, hys_exit, periods_per_year
        )

        n_feats_used = int(np.sum(model.feature_importances_ > 0))
        print(
            f"  fold {fold_idx}: AUC={auc:.3f} feats={n_feats_used}/{len(feature_cols)} "
            f"naive={test_sim['net_return']:+.1%} "
            f"hys(e={hys_entry:.2f}/x={hys_exit:.2f})={hys_sim['net_return']:+.1%} "
            f"shuffle={shuffle_sim['net_return']:+.1%} "
            f"B&H={buy_hold:+.1%}"
        )

        results.append(EvalResult(
            dataset=dataset,
            target=target,
            horizon_bars=horizon,
            fold=fold_idx,
            train_start=str(train["date"].iloc[0]),
            train_end=str(train["date"].iloc[-1]),
            val_start=str(val["date"].iloc[0]),
            val_end=str(val["date"].iloc[-1]),
            test_start=str(test["date"].iloc[0]),
            test_end=str(test["date"].iloc[-1]),
            train_rows=len(train),
            val_rows=len(val),
            test_rows=len(test),
            threshold=hys_entry,  # record entry threshold
            auc=auc,
            accuracy=float(accuracy_score(test[target], (test_probs >= 0.5).astype(int))),
            base_rate=float(test[target].mean()),
            val_net_return_pct=val_sim["net_return"] * 100.0,
            test_net_return_pct=hys_sim["net_return"] * 100.0,  # use hysteresis result
            buy_hold_pct=buy_hold * 100.0,
            excess_vs_bh_pct=(hys_sim["net_return"] - buy_hold) * 100.0,
            sharpe=hys_sim["sharpe"],
            max_drawdown_pct=hys_sim["max_drawdown"] * 100.0,
            entries=hys_sim["entries"],
            time_in_market_pct=hys_sim["time_in_market"] * 100.0,
            turnover=hys_sim["turnover"],
            shuffle_net_return_pct=shuffle_sim["net_return"] * 100.0,
            signal_lift_pct=(hys_sim["net_return"] - shuffle_sim["net_return"]) * 100.0,
        ))
    return results


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, target), g in results.groupby(["dataset", "target"]):
        rows.append({
            "dataset": dataset,
            "target": target,
            "folds": len(g),
            "mean_auc": g["auc"].mean(),
            "mean_net_pct": g["test_net_return_pct"].mean(),
            "median_net_pct": g["test_net_return_pct"].median(),
            "mean_bh_pct": g["buy_hold_pct"].mean(),
            "mean_excess_pct": g["excess_vs_bh_pct"].mean(),
            "positive_folds_pct": (g["test_net_return_pct"] > 0).mean() * 100.0,
            "positive_excess_folds_pct": (g["excess_vs_bh_pct"] > 0).mean() * 100.0,
            "mean_sharpe": g["sharpe"].mean(),
            "total_entries": int(g["entries"].sum()),
            "mean_time_in_market_pct": g["time_in_market_pct"].mean(),
            "worst_drawdown_pct": g["max_drawdown_pct"].min(),
            "mean_shuffle_net_pct": g["shuffle_net_return_pct"].mean(),
            "mean_signal_lift_pct": g["signal_lift_pct"].mean(),
        })
    return pd.DataFrame(rows).sort_values("mean_net_pct", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--output", default="reports/btc_v2_eval.csv")
    args = parser.parse_args()

    all_results: list[EvalResult] = []
    for path_str in args.data:
        path = Path(path_str)
        df = feather.read_feather(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        targets = sorted([c for c in df.columns if c.startswith("target_tradeable_")])
        print(f"\n=== {path.name}: rows={len(df):,} targets={len(targets)} ===")
        for target in targets:
            rows = evaluate_target(df, path.stem, target, args.folds)
            all_results.extend(rows)
            if rows:
                g = pd.DataFrame([asdict(r) for r in rows])
                print(
                    f"  {target}: auc={g['auc'].mean():.4f} "
                    f"net={g['test_net_return_pct'].mean():+.2f}% "
                    f"excess={g['excess_vs_bh_pct'].mean():+.2f}% "
                    f"pos={((g['test_net_return_pct'] > 0).mean() * 100):.1f}% "
                    f"entries={int(g['entries'].sum())}"
                )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame([asdict(r) for r in all_results])
    result_df.to_csv(out_path, index=False)
    out_path.with_suffix(".json").write_text(json.dumps([asdict(r) for r in all_results], indent=2), encoding="utf-8")
    summary = summarize(result_df) if not result_df.empty else pd.DataFrame()
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    summary_path.with_suffix(".json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    print("\nSummary:")
    print(summary.to_string(index=False) if not summary.empty else "No results.")
    print(f"\nSaved: {out_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
