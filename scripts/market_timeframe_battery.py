#!/usr/bin/env python3
"""
Market/timeframe battery for quick alpha sanity checks.

This intentionally does not reuse the live dollar-bar pipeline. It answers a
coarser question: do local OHLCV features produce a net-positive, stable signal
on another market/timeframe after realistic trading costs?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/home/nosferatu/freqtrade/user_data/data/binance")


@dataclass
class Result:
    pair: str
    timeframe: str
    source: str
    horizon_bars: int
    horizon_label: str
    rows: int
    start: str
    end: str
    train_rows: int
    val_rows: int
    test_rows: int
    threshold: float
    val_net_return_pct: float
    test_net_return_pct: float
    buy_hold_pct: float
    excess_vs_bh_pct: float
    test_auc: float
    test_accuracy: float
    base_rate: float
    sharpe: float
    max_drawdown_pct: float
    trades: int
    time_in_market_pct: float
    turnover: float


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_feather(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return out.dropna(subset=["open", "high", "low", "close"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]
    log_close = np.log(close)
    log_ret = log_close.diff()

    out["ret_1"] = log_ret
    for n in (2, 3, 6, 12, 24, 48, 96, 168):
        out[f"ret_{n}"] = log_close.diff(n)
        out[f"vol_{n}"] = log_ret.rolling(n).std()
        out[f"range_{n}"] = ((high - low) / close).rolling(n).mean()
        out[f"volume_z_{n}"] = (volume - volume.rolling(n).mean()) / volume.rolling(n).std()

    for fast, slow in ((6, 24), (12, 48), (24, 96)):
        ma_fast = close.ewm(span=fast, adjust=False).mean()
        ma_slow = close.ewm(span=slow, adjust=False).mean()
        out[f"ema_spread_{fast}_{slow}"] = ma_fast / ma_slow - 1.0

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    out["atr_14_pct"] = tr.ewm(span=14, adjust=False).mean() / close
    out["hl_pct"] = (high - low) / close
    out["oc_pct"] = (close - out["open"]) / out["open"]
    out["dollar_volume_log"] = np.log1p(close * volume)

    feature_cols = [c for c in out.columns if c not in {"open", "high", "low", "close", "volume"}]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)
    return out.dropna()


def split_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())


def simulate(
    returns: pd.Series,
    probs: np.ndarray,
    threshold: float,
    cost_bps_one_way: float,
    periods_per_year: float,
) -> dict:
    position = (probs >= threshold).astype(float)
    bar_ret = returns.fillna(0.0).to_numpy()
    gross = np.zeros(len(bar_ret))
    gross[1:] = position[:-1] * bar_ret[1:]
    turnover = np.abs(np.diff(position, prepend=0.0))
    costs = turnover * (cost_bps_one_way / 10_000.0)
    net = gross - costs
    equity = np.cumprod(1.0 + net)
    sharpe = 0.0
    if np.std(net) > 1e-12:
        sharpe = float(np.mean(net) / np.std(net) * np.sqrt(periods_per_year))
    return {
        "net_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity) if len(equity) else 0.0,
        "trades": int(np.sum((position == 1.0) & (np.roll(position, 1) == 0.0))),
        "time_in_market": float(position.mean()) if len(position) else 0.0,
        "turnover": float(turnover.sum()),
    }


def horizon_label(timeframe: str, horizon: int) -> str:
    minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}.get(timeframe, 60) * horizon
    if minutes < 60:
        return f"{minutes}m"
    if minutes < 1440:
        return f"{minutes // 60}h"
    return f"{minutes / 1440:.1f}d"


def evaluate_case(
    pair: str,
    timeframe: str,
    source_path: Path,
    df_raw: pd.DataFrame,
    horizon: int,
    cost_bps_one_way: float,
) -> Result | None:
    df = add_features(df_raw)
    future_return = df["close"].shift(-horizon) / df["close"] - 1.0
    df["target"] = (future_return > 0.0).astype(int)
    df["future_return"] = future_return
    df = df.iloc[:-horizon].dropna().copy()
    if len(df) < 1000:
        return None

    train, val, test = split_df(df)
    if min(len(train), len(val), len(test)) < 200:
        return None

    feature_cols = [c for c in df.columns if c not in {"open", "high", "low", "close", "volume", "target", "future_return"}]
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 400,
        "learning_rate": 0.025,
        "max_depth": 3,
        "num_leaves": 8,
        "min_child_samples": 80,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 3.0,
        "reg_lambda": 10.0,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        train[feature_cols],
        train["target"],
        eval_set=[(val[feature_cols], val["target"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    val_probs = model.predict_proba(val[feature_cols])[:, 1]
    test_probs = model.predict_proba(test[feature_cols])[:, 1]

    periods_per_year = {
        "1m": 365.25 * 24 * 60,
        "5m": 365.25 * 24 * 12,
        "15m": 365.25 * 24 * 4,
        "1h": 365.25 * 24,
    }.get(timeframe, 365.25 * 24)

    thresholds = np.arange(0.50, 0.625, 0.025)
    val_bar_returns = val["close"].pct_change().fillna(0.0)
    test_bar_returns = test["close"].pct_change().fillna(0.0)

    val_scores = [
        (simulate(val_bar_returns, val_probs, float(t), cost_bps_one_way, periods_per_year)["net_return"], float(t))
        for t in thresholds
    ]
    _, threshold = max(val_scores, key=lambda x: x[0])
    val_sim = simulate(val_bar_returns, val_probs, threshold, cost_bps_one_way, periods_per_year)
    test_sim = simulate(test_bar_returns, test_probs, threshold, cost_bps_one_way, periods_per_year)

    buy_hold = test["close"].iloc[-1] / test["close"].iloc[0] - 1.0
    try:
        auc = float(roc_auc_score(test["target"], test_probs))
    except ValueError:
        auc = 0.5
    acc = float(accuracy_score(test["target"], (test_probs >= 0.5).astype(int)))

    return Result(
        pair=pair,
        timeframe=timeframe,
        source=str(source_path),
        horizon_bars=horizon,
        horizon_label=horizon_label(timeframe, horizon),
        rows=len(df),
        start=str(df.index[0]),
        end=str(df.index[-1]),
        train_rows=len(train),
        val_rows=len(val),
        test_rows=len(test),
        threshold=threshold,
        val_net_return_pct=val_sim["net_return"] * 100.0,
        test_net_return_pct=test_sim["net_return"] * 100.0,
        buy_hold_pct=buy_hold * 100.0,
        excess_vs_bh_pct=(test_sim["net_return"] - buy_hold) * 100.0,
        test_auc=auc,
        test_accuracy=acc,
        base_rate=float(test["target"].mean()),
        sharpe=test_sim["sharpe"],
        max_drawdown_pct=test_sim["max_drawdown"] * 100.0,
        trades=test_sim["trades"],
        time_in_market_pct=test_sim["time_in_market"] * 100.0,
        turnover=test_sim["turnover"],
    )


def build_cases() -> list[tuple[str, str, Path, pd.DataFrame, list[int]]]:
    cases = []
    for pair in ("BTC_USDT", "ETH_USDT"):
        path = DATA_DIR / f"{pair}-1h.feather"
        if path.exists():
            cases.append((pair, "1h", path, load_ohlcv(path), [1, 4, 24]))

        path_1m = DATA_DIR / f"{pair}-1m.feather"
        if path_1m.exists():
            df_1m = load_ohlcv(path_1m)
            cases.append((pair, "15m", path_1m, resample_ohlcv(df_1m, "15min"), [4, 16, 96]))

    path_5m = DATA_DIR / "BTC_USDT-5m.feather"
    if path_5m.exists():
        cases.append(("BTC_USDT", "5m", path_5m, load_ohlcv(path_5m), [12, 48, 288]))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-bps-one-way", type=float, default=7.0)
    parser.add_argument("--output", default=str(ROOT / "reports" / "market_timeframe_battery.csv"))
    args = parser.parse_args()

    results: list[Result] = []
    for pair, timeframe, source, df, horizons in build_cases():
        print(f"\n=== {pair} {timeframe} rows={len(df):,} source={source.name} ===")
        for horizon in horizons:
            result = evaluate_case(pair, timeframe, source, df, horizon, args.cost_bps_one_way)
            if result is None:
                print(f"  h={horizon}: skipped")
                continue
            results.append(result)
            print(
                "  h={:<3d} {:>5s} auc={:.4f} net={:+7.2f}% bh={:+7.2f}% "
                "sharpe={:+5.2f} trades={:>4d} tim={:>5.1f}% thr={:.3f}".format(
                    horizon,
                    result.horizon_label,
                    result.test_auc,
                    result.test_net_return_pct,
                    result.buy_hold_pct,
                    result.sharpe,
                    result.trades,
                    result.time_in_market_pct,
                    result.threshold,
                )
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame([asdict(r) for r in results])
    df_out.to_csv(out_path, index=False)
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")

    if not df_out.empty:
        print("\nTop by test net return:")
        cols = [
            "pair", "timeframe", "horizon_label", "test_auc", "test_net_return_pct",
            "buy_hold_pct", "excess_vs_bh_pct", "sharpe", "trades", "time_in_market_pct",
        ]
        print(df_out.sort_values("test_net_return_pct", ascending=False)[cols].head(12).to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
