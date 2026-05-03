#!/usr/bin/env python3
"""
Strict walk-forward validation for ETH/USDT 1h.

Protocol:
  - Features are computed once from local OHLCV only.
  - For each test year Y:
      train = all data before Y-1
      validation = year Y-1
      test = year Y
  - Threshold is selected only on the validation year.
  - Final metrics are reported on the unseen test year.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_timeframe_battery import add_features, load_ohlcv, simulate  # noqa: E402


DATA_PATH = Path("/home/nosferatu/freqtrade/user_data/data/binance/ETH_USDT-1h.feather")
PERIODS_PER_YEAR = 365.25 * 24


@dataclass
class FoldResult:
    horizon_bars: int
    horizon_label: str
    cost_bps_one_way: float
    test_year: int
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
    val_net_return_pct: float
    test_net_return_pct: float
    buy_hold_pct: float
    excess_vs_bh_pct: float
    auc: float
    accuracy: float
    base_rate: float
    sharpe: float
    max_drawdown_pct: float
    trades: int
    time_in_market_pct: float
    turnover: float


def horizon_label(horizon: int) -> str:
    return f"{horizon}h" if horizon < 24 else f"{horizon / 24:.1f}d"


def model_params() -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 250,
        "learning_rate": 0.03,
        "max_depth": 3,
        "num_leaves": 8,
        "min_child_samples": 200,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 5.0,
        "reg_lambda": 15.0,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }


def prepare_data(horizon: int) -> pd.DataFrame:
    raw = load_ohlcv(DATA_PATH)
    df = add_features(raw)
    future_return = df["close"].shift(-horizon) / df["close"] - 1.0
    df["target"] = (future_return > 0.0).astype(int)
    df["future_return"] = future_return
    return df.iloc[:-horizon].dropna().copy()


def feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {"open", "high", "low", "close", "volume", "target", "future_return"}
    return [c for c in df.columns if c not in blocked]


def choose_threshold(val_returns: pd.Series, val_probs: np.ndarray, cost_bps: float) -> tuple[float, dict]:
    thresholds = np.arange(0.50, 0.625, 0.025)
    scored = []
    for threshold in thresholds:
        sim = simulate(val_returns, val_probs, float(threshold), cost_bps, PERIODS_PER_YEAR)
        scored.append((sim["net_return"], float(threshold), sim))
    _, threshold, sim = max(scored, key=lambda row: row[0])
    return threshold, sim


def run_fold(df: pd.DataFrame, horizon: int, costs_bps: list[float], test_year: int) -> list[FoldResult]:
    val_year = test_year - 1
    train = df[df.index.year < val_year]
    val = df[df.index.year == val_year]
    test = df[df.index.year == test_year]
    if min(len(train), len(val), len(test)) < 1000:
        return []

    cols = feature_columns(df)
    model = lgb.LGBMClassifier(**model_params())
    model.fit(
        train[cols],
        train["target"],
        eval_set=[(val[cols], val["target"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    val_probs = model.predict_proba(val[cols])[:, 1]
    test_probs = model.predict_proba(test[cols])[:, 1]
    val_returns = val["close"].pct_change().fillna(0.0)
    test_returns = test["close"].pct_change().fillna(0.0)

    buy_hold = test["close"].iloc[-1] / test["close"].iloc[0] - 1.0

    try:
        auc = float(roc_auc_score(test["target"], test_probs))
    except ValueError:
        auc = 0.5

    results = []
    for cost_bps in costs_bps:
        threshold, val_sim = choose_threshold(val_returns, val_probs, cost_bps)
        test_sim = simulate(test_returns, test_probs, threshold, cost_bps, PERIODS_PER_YEAR)
        results.append(FoldResult(
            horizon_bars=horizon,
            horizon_label=horizon_label(horizon),
            cost_bps_one_way=cost_bps,
            test_year=test_year,
            train_start=str(train.index[0]),
            train_end=str(train.index[-1]),
            val_start=str(val.index[0]),
            val_end=str(val.index[-1]),
            test_start=str(test.index[0]),
            test_end=str(test.index[-1]),
            train_rows=len(train),
            val_rows=len(val),
            test_rows=len(test),
            threshold=threshold,
            val_net_return_pct=val_sim["net_return"] * 100.0,
            test_net_return_pct=test_sim["net_return"] * 100.0,
            buy_hold_pct=buy_hold * 100.0,
            excess_vs_bh_pct=(test_sim["net_return"] - buy_hold) * 100.0,
            auc=auc,
            accuracy=float(accuracy_score(test["target"], (test_probs >= 0.5).astype(int))),
            base_rate=float(test["target"].mean()),
            sharpe=test_sim["sharpe"],
            max_drawdown_pct=test_sim["max_drawdown"] * 100.0,
            trades=test_sim["trades"],
            time_in_market_pct=test_sim["time_in_market"] * 100.0,
            turnover=test_sim["turnover"],
        ))
    return results


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    grouped = []
    for (horizon, cost), g in results.groupby(["horizon_label", "cost_bps_one_way"]):
        grouped.append({
            "horizon_label": horizon,
            "cost_bps_one_way": cost,
            "folds": len(g),
            "mean_net_pct": g["test_net_return_pct"].mean(),
            "median_net_pct": g["test_net_return_pct"].median(),
            "sum_net_pct": g["test_net_return_pct"].sum(),
            "mean_bh_pct": g["buy_hold_pct"].mean(),
            "mean_excess_pct": g["excess_vs_bh_pct"].mean(),
            "mean_auc": g["auc"].mean(),
            "positive_folds_pct": (g["test_net_return_pct"] > 0).mean() * 100.0,
            "positive_excess_folds_pct": (g["excess_vs_bh_pct"] > 0).mean() * 100.0,
            "mean_sharpe": g["sharpe"].mean(),
            "total_trades": int(g["trades"].sum()),
            "mean_time_in_market_pct": g["time_in_market_pct"].mean(),
            "worst_drawdown_pct": g["max_drawdown_pct"].min(),
        })
    return pd.DataFrame(grouped).sort_values(
        ["cost_bps_one_way", "mean_net_pct"],
        ascending=[True, False],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--costs", type=float, nargs="+", default=[7.0, 10.0, 2.0])
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--output", default=str(ROOT / "reports" / "eth_1h_walkforward.csv"))
    args = parser.parse_args()

    all_results: list[FoldResult] = []
    for horizon in args.horizons:
        df = prepare_data(horizon)
        print(f"\n=== ETH_USDT 1h horizon={horizon_label(horizon)} rows={len(df):,} ===")
        for year in range(args.start_year, args.end_year + 1):
            fold_results = run_fold(df, horizon, args.costs, year)
            if not fold_results:
                print(f"  {year}: skipped")
                continue
            all_results.extend(fold_results)
            for result in fold_results:
                print(
                    "  {year} cost={cost:>4.1f}: auc={auc:.4f} net={net:+7.2f}% bh={bh:+7.2f}% "
                    "excess={excess:+7.2f}% sharpe={sharpe:+5.2f} trades={trades:>4d} "
                    "tim={tim:>5.1f}% thr={thr:.3f}".format(
                        year=year,
                        cost=result.cost_bps_one_way,
                        auc=result.auc,
                        net=result.test_net_return_pct,
                        bh=result.buy_hold_pct,
                        excess=result.excess_vs_bh_pct,
                        sharpe=result.sharpe,
                        trades=result.trades,
                        tim=result.time_in_market_pct,
                        thr=result.threshold,
                    )
                )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame([asdict(r) for r in all_results])
    results_df.to_csv(out_path, index=False)
    out_path.with_suffix(".json").write_text(
        json.dumps([asdict(r) for r in all_results], indent=2),
        encoding="utf-8",
    )

    summary_df = summarize(results_df)
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    summary_path.with_suffix(".json").write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    print("\nSummary:")
    if summary_df.empty:
        print("No valid folds.")
    else:
        print(summary_df.to_string(index=False))
    print(f"\nSaved: {out_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
