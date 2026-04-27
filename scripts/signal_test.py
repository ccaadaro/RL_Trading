#!/usr/bin/env python3
"""
signal_test.py — LightGBM Direction Prediction Signal Test
═══════════════════════════════════════════════════════════

The fundamental question: do your 107 features contain alpha?

If LightGBM can't predict next-bar direction at >51%, then no RL
architecture will either. This test takes 5 minutes and answers
the question that 64 RL runs couldn't.

Targets:
  1. sign(next_bar_close_return)  — binary classification
  2. sign(next_N_bar_return)      — multi-horizon

Evaluation:
  - Accuracy, AUC, precision/recall per class
  - Cumulative return of a naive strategy: go LONG if P(up) > 0.5, SHORT otherwise
  - Feature importance ranking
  - Regime-stratified accuracy (bull/bear/sideways)
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    precision_recall_fscore_support, log_loss,
)
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH = "cache/data_v1_430340a861af5f8f9bcbd7a4ca16ba95.parquet"

# Same splits as the RL training
TRAIN_END   = "2024-06-30"
VAL_START   = "2024-07-01"
VAL_END     = "2025-06-30"
TEST_START  = "2025-07-01"

# Prediction horizons (in bars)
HORIZONS = [1, 3, 6, 12, 24]

# LightGBM params — conservative to avoid overfitting
LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


    # Features to exclude due to data quality / leakage issues:
    # - cross_section_market_index_feature: (BTC+ETH)/2 ≈ close price proxy (corr 0.9996)
    # - futures_mark_basis_feature: constant/fictitious values pre-Q3 2019
    # - futures_mark_basis_ma_feature: same source
EXCLUDE_FEATURES = {
    "cross_section_market_index_feature",
    "futures_mark_basis_feature",
    "futures_mark_basis_ma_feature",
}


def load_data():
    """Load data and prepare features + targets."""
    df = pd.read_parquet(DATA_PATH)
    feat_cols = sorted([c for c in df.columns if "feature" in c and c not in EXCLUDE_FEATURES])

    # Target: sign of future return at various horizons
    for h in HORIZONS:
        future_ret = np.log(df["close"].shift(-h) / df["close"])
        df[f"target_{h}bar"] = (future_ret > 0).astype(int)
        df[f"return_{h}bar"] = future_ret

    # Drop rows where target is NaN (last N rows)
    max_h = max(HORIZONS)
    df = df.iloc[:-max_h].copy()

    return df, feat_cols


def split_data(df, feat_cols):
    """Split into train/val/test using the same dates as RL training."""
    train = df[:TRAIN_END]
    val   = df[VAL_START:VAL_END]
    test  = df[TEST_START:]

    print(f"  Train : {train.index[0].date()} → {train.index[-1].date()}  ({len(train)} bars)")
    print(f"  Val   : {val.index[0].date()} → {val.index[-1].date()}  ({len(val)} bars)")
    print(f"  Test  : {test.index[0].date()} → {test.index[-1].date()}  ({len(test)} bars)")
    print()

    return train, val, test


def regime_label(df):
    """Simple regime labeling using 180-bar trend return."""
    col = "trend_return_180_feature"
    if col not in df.columns:
        return pd.Series("unknown", index=df.index)
    labels = pd.Series("sideways", index=df.index)
    labels[df[col] > df[col].quantile(0.67)] = "bull"
    labels[df[col] < df[col].quantile(0.33)] = "bear"
    return labels


def train_and_evaluate(train, val, test, feat_cols, horizon):
    """Train LightGBM for a single horizon, evaluate on val and test."""
    target_col = f"target_{horizon}bar"
    return_col = f"return_{horizon}bar"

    X_train, y_train = train[feat_cols], train[target_col]
    X_val,   y_val   = val[feat_cols],   val[target_col]
    X_test,  y_test  = test[feat_cols],  test[target_col]

    # Train with early stopping on validation
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    results = {}
    for name, X, y, ret_series in [
        ("val",  X_val,  y_val,  val[return_col]),
        ("test", X_test, y_test, test[return_col]),
    ]:
        probs = model.predict_proba(X)[:, 1]  # P(up)
        preds = (probs > 0.5).astype(int)

        acc  = accuracy_score(y, preds)
        auc  = roc_auc_score(y, probs)
        loss = log_loss(y, probs)

        # Naive trading return: LONG if P(up) > 0.5, SHORT otherwise
        position = np.where(probs > 0.5, 1.0, -1.0)
        per_bar_ret = position * ret_series.values
        cum_ret = np.expm1(per_bar_ret.sum()) * 100  # total %
        sharpe = per_bar_ret.mean() / (per_bar_ret.std() + 1e-10) * np.sqrt(365 * 24)  # annualized for 1h bars

        # Buy & hold benchmark
        bh_ret = np.expm1(ret_series.sum()) * 100

        # Regime-stratified accuracy
        regimes = regime_label(val if name == "val" else test)
        regime_acc = {}
        for regime in ["bull", "bear", "sideways"]:
            mask = regimes == regime
            if mask.sum() > 0:
                regime_acc[regime] = accuracy_score(y[mask], preds[mask])

        results[name] = {
            "accuracy": acc,
            "auc": auc,
            "logloss": loss,
            "cum_return": cum_ret,
            "bh_return": bh_ret,
            "alpha": cum_ret - bh_ret,
            "sharpe": sharpe,
            "n_samples": len(y),
            "base_rate": y.mean(),  # % of up-bars (random baseline)
            "regime_acc": regime_acc,
            "n_trees": model.best_iteration_ if model.best_iteration_ else LGB_PARAMS["n_estimators"],
        }

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)

    return results, imp


def print_results(horizon, results, importance):
    """Pretty-print results for one horizon."""
    print(f"\n{'═' * 70}")
    print(f"  HORIZON: {horizon}-bar ahead  ({'~' + str(horizon) + 'h' if horizon <= 24 else str(horizon // 24) + 'd'})")
    print(f"{'═' * 70}")

    for split in ["val", "test"]:
        r = results[split]
        base = r["base_rate"]
        acc  = r["accuracy"]
        edge = acc - base  # accuracy above random

        print(f"\n  [{split.upper()}]  ({r['n_samples']} bars)")
        print(f"    Base rate (% up)  : {base:.1%}")
        print(f"    Accuracy          : {acc:.1%}  (edge: {edge:+.1%})")
        print(f"    AUC               : {r['auc']:.4f}")
        print(f"    Log-loss          : {r['logloss']:.4f}")
        print(f"    Trees used        : {r['n_trees']}")
        print(f"    ─── Trading Simulation ───")
        print(f"    Strategy return   : {r['cum_return']:+.2f}%")
        print(f"    Buy & Hold        : {r['bh_return']:+.2f}%")
        print(f"    Alpha             : {r['alpha']:+.2f}%")
        print(f"    Sharpe (ann.)     : {r['sharpe']:+.3f}")

        if r["regime_acc"]:
            print(f"    ─── By Regime ───")
            for regime, racc in r["regime_acc"].items():
                print(f"    {regime:>10s} : {racc:.1%}")

    print(f"\n  Top 15 Features:")
    for i, (feat, score) in enumerate(importance.head(15).items()):
        print(f"    {i+1:2d}. {feat:45s} {score:5.0f}")


def confidence_threshold_test(train, val, test, feat_cols, horizon=1):
    """Test if higher confidence predictions have higher accuracy."""
    target_col = f"target_{horizon}bar"
    return_col = f"return_{horizon}bar"

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        train[feat_cols], train[target_col],
        eval_set=[(val[feat_cols], val[target_col])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    print(f"\n{'═' * 70}")
    print(f"  CONFIDENCE THRESHOLD ANALYSIS (horizon={horizon})")
    print(f"{'═' * 70}")

    for name, X, y, rets in [
        ("val",  val[feat_cols],  val[target_col],  val[return_col]),
        ("test", test[feat_cols], test[target_col], test[return_col]),
    ]:
        probs = model.predict_proba(X)[:, 1]
        print(f"\n  [{name.upper()}]")
        print(f"  {'Threshold':>12s} {'Acc':>7s} {'Trades':>7s} {'Coverage':>9s} {'Return':>9s} {'Sharpe':>8s}")
        print(f"  {'─' * 55}")

        for thresh in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]:
            # Only trade when confident
            long_mask  = probs > (0.5 + (thresh - 0.5))
            short_mask = probs < (1.0 - thresh)
            trade_mask = long_mask | short_mask

            if trade_mask.sum() == 0:
                print(f"  {thresh:>12.2f} {'---':>7s} {0:>7d} {0:>8.1%} {'---':>9s} {'---':>8s}")
                continue

            position = np.where(long_mask, 1.0, np.where(short_mask, -1.0, 0.0))
            preds_filtered = np.where(long_mask, 1, 0)[trade_mask]
            acc = accuracy_score(y.values[trade_mask], preds_filtered)

            trade_rets = (position * rets.values)[trade_mask]
            cum = np.expm1(trade_rets.sum()) * 100
            sharpe = trade_rets.mean() / (trade_rets.std() + 1e-10) * np.sqrt(365 * 24)
            coverage = trade_mask.sum() / len(trade_mask)

            print(f"  {thresh:>12.2f} {acc:>6.1%} {trade_mask.sum():>7d} {coverage:>8.1%} {cum:>+8.2f}% {sharpe:>+7.3f}")


def walk_forward_test(df, feat_cols, horizon=1, window_months=6, step_months=3):
    """Walk-forward validation to check for temporal stability."""
    target_col = f"target_{horizon}bar"
    return_col = f"return_{horizon}bar"

    print(f"\n{'═' * 70}")
    print(f"  WALK-FORWARD TEST (horizon={horizon}, window={window_months}mo, step={step_months}mo)")
    print(f"{'═' * 70}")
    print(f"  {'Period':>25s} {'Acc':>7s} {'AUC':>7s} {'Edge':>7s} {'Return':>9s} {'BH':>9s} {'Sharpe':>8s}")
    print(f"  {'─' * 75}")

    # Generate test windows
    all_dates = df.index
    start = pd.Timestamp("2021-01-01")  # Start walk-forward after 2 years of training data
    results = []

    while start < all_dates[-1] - pd.DateOffset(months=window_months):
        end = start + pd.DateOffset(months=window_months)
        train_end = start - pd.DateOffset(days=1)

        train_data = df[:train_end]
        test_data = df[start:end]

        if len(train_data) < 5000 or len(test_data) < 100:
            start += pd.DateOffset(months=step_months)
            continue

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        # Use last 20% of train as validation for early stopping
        split_idx = int(len(train_data) * 0.8)
        t_train = train_data.iloc[:split_idx]
        t_val = train_data.iloc[split_idx:]

        model.fit(
            t_train[feat_cols], t_train[target_col],
            eval_set=[(t_val[feat_cols], t_val[target_col])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        probs = model.predict_proba(test_data[feat_cols])[:, 1]
        preds = (probs > 0.5).astype(int)
        y = test_data[target_col]

        acc = accuracy_score(y, preds)
        try:
            auc = roc_auc_score(y, probs)
        except:
            auc = 0.5
        base = y.mean()
        edge = acc - base

        position = np.where(probs > 0.5, 1.0, -1.0)
        bar_rets = position * test_data[return_col].values
        cum_ret = np.expm1(bar_rets.sum()) * 100
        bh_ret = np.expm1(test_data[return_col].sum()) * 100
        sharpe = bar_rets.mean() / (bar_rets.std() + 1e-10) * np.sqrt(365 * 24)

        period_str = f"{start.date()} → {end.date()}"
        print(f"  {period_str:>25s} {acc:>6.1%} {auc:>6.3f} {edge:>+6.1%} {cum_ret:>+8.2f}% {bh_ret:>+8.2f}% {sharpe:>+7.3f}")

        results.append({
            "start": start, "end": end, "acc": acc, "auc": auc, "edge": edge,
            "return": cum_ret, "bh": bh_ret, "sharpe": sharpe,
        })
        start += pd.DateOffset(months=step_months)

    if results:
        rdf = pd.DataFrame(results)
        print(f"\n  {'AGGREGATE':>25s} {rdf['acc'].mean():>6.1%} {rdf['auc'].mean():>6.3f} {rdf['edge'].mean():>+6.1%} {rdf['return'].sum():>+8.2f}% {rdf['bh'].sum():>+8.2f}% {rdf['sharpe'].mean():>+7.3f}")
        print(f"  {'Win rate (periods)':>25s} {(rdf['return'] > 0).mean():>6.1%}")
        print(f"  {'Positive edge periods':>25s} {(rdf['edge'] > 0).mean():>6.1%}")


def main():
    print("=" * 70)
    print("  LIGHTGBM SIGNAL TEST — Do your features contain alpha?")
    print("=" * 70)
    print()

    df, feat_cols = load_data()
    print(f"  Data: {len(df)} bars, {len(feat_cols)} features")
    train, val, test = split_data(df, feat_cols)

    # ── Test each horizon ─────────────────────────────────────────────────
    all_importance = None
    for h in HORIZONS:
        results, imp = train_and_evaluate(train, val, test, feat_cols, h)
        print_results(h, results, imp)
        if h == 1:
            all_importance = imp

    # ── Confidence threshold analysis ─────────────────────────────────────
    confidence_threshold_test(train, val, test, feat_cols, horizon=1)

    # ── Walk-forward test ─────────────────────────────────────────────────
    walk_forward_test(df, feat_cols, horizon=1)

    # ── Final verdict ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  VERDICT")
    print(f"{'=' * 70}")
    print("""
  Look at these numbers:
  1. If accuracy > 51% consistently on TEST set → signal exists
  2. If AUC > 0.52 on TEST → features have ranking power
  3. If walk-forward edge is positive in >60% of periods → robust signal
  4. If confidence threshold helps (higher thresh = higher acc) → real signal
  5. If all of the above fail → features don't contain tradeable alpha
     and no RL architecture will fix that.
""")


if __name__ == "__main__":
    main()
