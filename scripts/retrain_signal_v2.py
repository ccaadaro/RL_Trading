#!/usr/bin/env python3
"""
retrain_signal_v2.py — Retrain LightGBM signal model on computable features only
═══════════════════════════════════════════════════════════════════════════════════

v1 model used 104 features, many requiring taker-volume, cross-section networks,
or external APIs unavailable at Freqtrade inference time. v2 uses 58 features
that can be reproduced exactly by the strategy's populate_indicators.

The key improvement: training features EXACTLY MATCH inference features.
Expected performance: ~85–90% of v1 signal quality (covers 80%+ of importance).

Walk-forward settings (same as v1):
  WF_WARMUP_BARS = 4 380   (~6 months of 1h bars)
  WF_STEP_BARS   = 1 095   (~45 days, 7 windows over training period)

Output:
  models/signal_lgbm_v2.pkl   — model + feat_cols
  Parquet updated              — signal_prob_v2_feature column added

Usage:
  cd user_data/strategies/RL_Trading
  python scripts/retrain_signal_v2.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

# ── project path ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from utils.signal_features import (
    SIGNAL_FEAT_COLS_V2,
    build_feature_matrix,
    load_funding_series,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH    = "cache/data_v1_430340a861af5f8f9bcbd7a4ca16ba95.parquet"
ETH_PATH     = "../../data/binance/ETH_USDT-1h.feather"
FUNDING_PATH = "../../data/binance/futures/BTC_USDT_USDT-8h-funding_rate.feather"
MODEL_PATH   = "models/signal_lgbm_v2.pkl"

TRAIN_END  = "2024-06-30"
VAL_START  = "2024-07-01"
VAL_END    = "2025-06-30"
TEST_START = "2025-07-01"

HORIZON = 1   # bars ahead to predict

WF_WARMUP_BARS = 4_380   # ~6 months
WF_STEP_BARS   = 1_095   # ~45 days → 7 windows

LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "boosting_type":     "gbdt",
    "n_estimators":      600,
    "learning_rate":     0.03,
    "max_depth":         5,
    "num_leaves":        31,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.6,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "random_state":      42,
    "verbose":           -1,
    "n_jobs":            -1,
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    print(f"  Loading parquet: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    # Build target: sign of 1-bar-ahead return
    future_ret   = np.log(df["close"].shift(-HORIZON) / df["close"])
    df["target"] = (future_ret > 0).astype(int)
    df           = df.iloc[:-HORIZON].copy()

    # Load ETH data
    eth_df = pd.DataFrame()
    eth_p  = Path(ETH_PATH)
    if eth_p.exists():
        eth_df = pd.read_feather(str(eth_p))
        if "date" in eth_df.columns:
            eth_df = eth_df.set_index("date")
        eth_df.index = pd.to_datetime(eth_df.index, utc=True)
        eth_df.index = eth_df.index.tz_localize(None)
        print(f"  ETH data: {len(eth_df)} bars  "
              f"({eth_df.index[0].date()} → {eth_df.index[-1].date()})")
    else:
        print(f"  [WARN] ETH data not found at {eth_p} — ETH features set to 0")

    # Load funding rate
    funding = load_funding_series(FUNDING_PATH)
    if funding is not None:
        funding.index = funding.index.tz_localize(None)
        print(f"  Funding rate: {len(funding)} 1h bars  "
              f"({funding.index[0].date()} → {funding.index[-1].date()})")
    else:
        print(f"  [WARN] Funding rate not found at {FUNDING_PATH}")

    # Build full feature matrix
    print(f"  Computing {len(SIGNAL_FEAT_COLS_V2)} features …")
    X_full = build_feature_matrix(df, eth_df=eth_df, funding_series=funding)
    df[SIGNAL_FEAT_COLS_V2] = X_full[SIGNAL_FEAT_COLS_V2].values

    print(f"  Data  : {len(df)} bars  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    print(f"  Target: {df['target'].mean():.3f} mean (up fraction)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# LightGBM helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_lgbm(X_train, y_train, X_val, y_val):
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def evaluate(model, X, y, label):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)
    acc   = accuracy_score(y, preds)
    auc   = roc_auc_score(y, probs)
    ll    = log_loss(y, probs)
    edge  = acc - y.mean()
    print(f"  [{label}]  acc={acc:.3f} (edge={edge:+.3f})  AUC={auc:.4f}  logloss={ll:.4f}")
    return probs


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD (training set, zero lookahead)
# ══════════════════════════════════════════════════════════════════════════════

def generate_walkforward_predictions(df):
    train_df = df[:TRAIN_END].copy()
    n        = len(train_df)

    signal_probs = np.full(n, np.nan)
    windows      = []
    i = WF_WARMUP_BARS
    while i < n:
        windows.append((i, min(i + WF_STEP_BARS, n)))
        i += WF_STEP_BARS

    print(f"\n  Walk-forward: {len(windows)} windows  "
          f"(warmup={WF_WARMUP_BARS}, step={WF_STEP_BARS})")
    print(f"  {'Window':>35s}  {'Train':>6s}  {'Acc':>6s}  {'AUC':>6s}  {'Edge':>6s}")
    print(f"  {'─'*65}")

    for pred_start_i, pred_end_i in windows:
        es_split = int(pred_start_i * 0.8)

        X_t = train_df[SIGNAL_FEAT_COLS_V2].iloc[:es_split]
        y_t = train_df["target"].iloc[:es_split]
        X_v = train_df[SIGNAL_FEAT_COLS_V2].iloc[es_split:pred_start_i]
        y_v = train_df["target"].iloc[es_split:pred_start_i]
        X_p = train_df[SIGNAL_FEAT_COLS_V2].iloc[pred_start_i:pred_end_i]
        y_p = train_df["target"].iloc[pred_start_i:pred_end_i]

        model = train_lgbm(X_t, y_t, X_v, y_v)
        probs = model.predict_proba(X_p)[:, 1]
        signal_probs[pred_start_i:pred_end_i] = probs

        acc  = accuracy_score(y_p, (probs > 0.5).astype(int))
        auc  = roc_auc_score(y_p, probs) if y_p.nunique() > 1 else 0.5
        edge = acc - y_p.mean()
        period = (f"{train_df.index[pred_start_i].date()} → "
                  f"{train_df.index[pred_end_i - 1].date()}")
        print(f"  {period:>35s}  {len(X_t):>6d}  {acc:>5.1%}  {auc:>5.3f}  {edge:>+5.1%}")

    n_nan        = np.isnan(signal_probs).sum()
    signal_probs = np.where(np.isnan(signal_probs), 0.5, signal_probs)
    print(f"\n  Pre-warmup bars filled with 0.5: {n_nan}")
    return signal_probs


# ══════════════════════════════════════════════════════════════════════════════
# FINAL MODEL
# ══════════════════════════════════════════════════════════════════════════════

def train_final_model(df):
    train = df[:TRAIN_END]
    val   = df[VAL_START:VAL_END]
    test  = df[TEST_START:]

    print(f"\n{'═'*65}")
    print(f"  FINAL MODEL — trained on full training set")
    print(f"{'═'*65}")
    print(f"  Train : {len(train)} bars")
    print(f"  Val   : {len(val)} bars")
    print(f"  Test  : {len(test)} bars")
    print(f"  Features: {len(SIGNAL_FEAT_COLS_V2)}")

    model = train_lgbm(
        train[SIGNAL_FEAT_COLS_V2], train["target"],
        val[SIGNAL_FEAT_COLS_V2],   val["target"],
    )
    print(f"\n  Trees used: {model.best_iteration_}")

    val_probs  = evaluate(model, val[SIGNAL_FEAT_COLS_V2],  val["target"],  "VAL ")
    test_probs = evaluate(model, test[SIGNAL_FEAT_COLS_V2], test["target"], "TEST")

    # Top 20 feature importances
    imp_pairs = sorted(
        zip(SIGNAL_FEAT_COLS_V2, model.feature_importances_),
        key=lambda x: -x[1],
    )
    print(f"\n  Top 15 features by importance:")
    for feat, imp in imp_pairs[:15]:
        bar = "█" * int(imp / max(p[1] for p in imp_pairs) * 30)
        print(f"    {imp:>5d}  {bar:<30s}  {feat}")

    # Threshold summary on test
    print(f"\n  Confidence threshold on TEST:")
    print(f"  {'Thresh':>8s}  {'Acc':>6s}  {'Trades':>7s}  {'Coverage':>9s}")
    for thresh in [0.50, 0.54, 0.56, 0.58, 0.60, 0.62]:
        long_mask = test_probs > thresh
        if long_mask.sum() == 0:
            continue
        acc = accuracy_score(test["target"].values[long_mask],
                             np.ones(long_mask.sum(), dtype=int))
        cov = long_mask.sum() / len(long_mask)
        print(f"  {thresh:>8.2f}  {acc:>5.1%}  {long_mask.sum():>7d}  {cov:>8.1%}")

    return model, val_probs, test_probs


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def quick_backtest(signal, close, threshold, label, fee=7e-4):
    n   = len(signal)
    pos = np.where(signal > threshold, 1.0, 0.0)
    portfolio = 1.0
    position  = 0.0
    peak      = 1.0
    equity    = []
    trades    = 0

    for i in range(n):
        target = pos[i - 1] if i > 0 else 0.0
        if target != position:
            portfolio -= abs(target - position) * fee * portfolio
            if target != 0:
                trades += 1
            position = target
        if i > 0 and close[i - 1] > 0:
            portfolio *= np.exp(position * np.log(close[i] / close[i - 1]))
        peak = max(peak, portfolio)
        equity.append(portfolio)

    eq     = np.array(equity)
    bh     = (close[-1] / close[0] - 1) * 100
    ret    = (eq[-1] - 1) * 100
    alpha  = ret - bh
    max_dd = max((peak - v) / peak for v in eq) * 100

    pos_arr = np.array([pos[max(0, i - 1)] for i in range(n)])
    in_mkt  = pos_arr[1:] > 0
    mr      = np.diff(np.log(np.maximum(close, 1e-10)))
    acc     = np.mean(mr[in_mkt] > 0) * 100 if in_mkt.sum() > 0 else 50.0

    log_rets = np.diff(np.log(np.maximum(eq, 1e-10)))
    sharpe   = (log_rets.mean() / (log_rets.std() + 1e-12)) * np.sqrt(8760)

    print(f"  {label:8s}  ret={ret:>+7.1f}%  bh={bh:>+7.1f}%  alpha={alpha:>+7.1f}%  "
          f"dd={max_dd:>5.1f}%  sharpe={sharpe:>+5.2f}  trades={trades:>4d}  acc={acc:>5.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  SIGNAL MODEL v2 TRAINING")
    print(f"  Features: {len(SIGNAL_FEAT_COLS_V2)} (computable from OHLCV + ETH + funding)")
    print("=" * 65)

    df = load_data()

    # Walk-forward predictions for training set
    print(f"\n{'═'*65}")
    print(f"  STEP 1 — Walk-forward predictions (training set)")
    print(f"{'═'*65}")
    train_probs = generate_walkforward_predictions(df)

    # Final model
    final_model, val_probs, test_probs = train_final_model(df)

    # Assemble full signal column
    print(f"\n{'═'*65}")
    print(f"  STEP 3 — Assembling signal_prob_v2_feature")
    print(f"{'═'*65}")
    train_idx = df[:TRAIN_END].index
    val_idx   = df[VAL_START:VAL_END].index
    test_idx  = df[TEST_START:].index

    signal = pd.Series(np.nan, index=df.index, name="signal_prob_v2_feature")
    signal.loc[train_idx] = train_probs
    signal.loc[val_idx]   = val_probs
    signal.loc[test_idx]  = test_probs

    if signal.isna().sum() > 0:
        print(f"  [WARN] {signal.isna().sum()} NaN values — filling with 0.5")
        signal = signal.fillna(0.5)

    print(f"  mean={signal.mean():.4f}  std={signal.std():.4f}  "
          f"min={signal.min():.4f}  max={signal.max():.4f}")
    print(f"  P(signal > 0.60): {(signal > 0.60).mean():.1%}")
    print(f"  P(signal > 0.58): {(signal > 0.58).mean():.1%}")

    # Backtest validation
    print(f"\n{'═'*65}")
    print(f"  STEP 4 — Backtest validation (threshold=0.60)")
    print(f"{'═'*65}")
    splits = {
        "TRAIN": (df[:TRAIN_END], train_probs),
        "VAL":   (df[VAL_START:VAL_END], val_probs),
        "TEST":  (df[TEST_START:], test_probs),
    }
    for name, (split_df, probs) in splits.items():
        for thresh in ([0.58, 0.60, 0.62] if name == "TEST" else [0.60]):
            quick_backtest(probs, split_df["close"].values,
                           thresh, f"{name}@{thresh:.2f}")

    # Save model
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model":         final_model,
            "feat_cols":     SIGNAL_FEAT_COLS_V2,
            "horizon":       HORIZON,
            "train_end":     TRAIN_END,
            "lgb_params":    LGB_PARAMS,
            "version":       2,
            "n_features":    len(SIGNAL_FEAT_COLS_V2),
        }, f)
    print(f"\n  Model saved → {MODEL_PATH}")

    # Write signal back to parquet
    df_out = pd.read_parquet(DATA_PATH)
    df_out = df_out.iloc[:-HORIZON]
    df_out["signal_prob_v2_feature"] = signal.values
    df_out.to_parquet(DATA_PATH)
    print(f"  Parquet updated → signal_prob_v2_feature added")

    print(f"\n{'='*65}")
    print(f"  DONE — v2 model ready")
    print(f"  Update strategy: set _MODEL_PATH to models/signal_lgbm_v2.pkl")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
