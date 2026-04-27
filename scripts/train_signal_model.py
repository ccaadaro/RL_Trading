#!/usr/bin/env python3
"""
train_signal_model.py — [DEPRECATED v1] — kept for reproducing historical runs only.

Use scripts/retrain_signal_v2.py instead. The v1 model uses 104 features
(taker volume, cross-section network, external-API-only columns) that are
NOT reproducible at Freqtrade inference time, so SignalLGBMStrategy cannot
serve this model in production. v2 uses 58 features derivable from
OHLCV + ETH/USDT + funding rate — identical in train and live.

Reading this script for any reason other than historical reference?
Stop and use retrain_signal_v2.py. It:
  - Writes signal_prob_v2_feature (not signal_prob_1h_feature)
  - Saves to models/signal_lgbm_v2.pkl
  - Matches SignalLGBMStrategy.py feature set exactly

Running this script today will regenerate the legacy signal_prob_1h_feature
column. Downstream code (signal_env, train_signal_rl, backtest_threshold)
accepts either column but prefers v2 when both are present.
══════════════════════════════════════════════════════════════════════════════════════════

Layer 1 of the two-layer RL architecture:
  LightGBM → P(up) → RL agent (position sizing only)

This script:
  1. Generates walk-forward out-of-fold predictions for the training set
     (minimum 1-year warmup, 6-month steps — zero lookahead)
  2. Trains a final model on the full training set → predicts val + test
  3. Saves model to models/signal_lgbm.pkl
  4. Saves predictions back into the parquet as signal_prob_1h_feature
"""

import sys as _sys
print(
    "\n[DEPRECATED] train_signal_model.py — this is the v1 pipeline and its "
    "features are not reproducible in Freqtrade. Use retrain_signal_v2.py.\n"
    "Set RLTRADING_ALLOW_V1=1 to proceed anyway.\n",
    file=_sys.stderr,
)
import os as _os
if _os.environ.get("RLTRADING_ALLOW_V1") != "1":
    _sys.exit(2)

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH  = "cache/data_v1_430340a861af5f8f9bcbd7a4ca16ba95.parquet"
MODEL_PATH = "models/signal_lgbm.pkl"

TRAIN_END  = "2024-06-30"
VAL_START  = "2024-07-01"
VAL_END    = "2025-06-30"
TEST_START = "2025-07-01"

HORIZON = 1  # bars ahead to predict

# Walk-forward params
WF_WARMUP_BARS = 4_380   # 6 months minimum training window before first prediction
WF_STEP_BARS   = 1_095   # retrain every ~45 days (~7 windows over training set)

# Features contaminated by leakage or bad historical data
EXCLUDE_FEATURES = {
    "cross_section_market_index_feature",   # (BTC+ETH)/2 ≈ close proxy, corr 0.9996
    "futures_mark_basis_feature",           # constant/fictitious pre-Q3 2019
    "futures_mark_basis_ma_feature",        # same source
}

# LightGBM — same conservative params as signal_test.py
LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "boosting_type":     "gbdt",
    "n_estimators":      500,
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
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_parquet(DATA_PATH)
    feat_cols = sorted([c for c in df.columns if "feature" in c and c not in EXCLUDE_FEATURES])

    future_ret = np.log(df["close"].shift(-HORIZON) / df["close"])
    df["target"] = (future_ret > 0).astype(int)
    df["future_return"] = future_ret

    # Drop last HORIZON rows (NaN target)
    df = df.iloc[:-HORIZON].copy()

    print(f"  Data  : {len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})")
    print(f"  Features: {len(feat_cols)}")
    return df, feat_cols


def train_lgbm(X_train, y_train, X_val, y_val):
    """Train with early stopping on val set."""
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
    acc  = accuracy_score(y, preds)
    auc  = roc_auc_score(y, probs)
    loss = log_loss(y, probs)
    edge = acc - y.mean()
    print(f"  [{label}]  acc={acc:.3f} (edge={edge:+.3f})  AUC={auc:.4f}  logloss={loss:.4f}")
    return probs


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD PREDICTION (training set — zero lookahead)
# ══════════════════════════════════════════════════════════════════════════════

def generate_walkforward_predictions(df, feat_cols):
    """
    Generate out-of-fold predictions for every bar in training set.

    For bar at index i:
      - Model is trained on bars [0 .. i-1] (only past data)
      - Prediction for bar i uses only historical information
    Retrains every WF_STEP_BARS bars after initial WF_WARMUP_BARS warmup.
    """
    train_df = df[:TRAIN_END].copy()
    n = len(train_df)

    signal_probs = np.full(n, np.nan)

    # First prediction window starts after warmup
    pred_start = WF_WARMUP_BARS
    step = WF_STEP_BARS

    windows = []
    i = pred_start
    while i < n:
        windows.append((i, min(i + step, n)))
        i += step

    print(f"\n  Walk-forward: {len(windows)} windows "
          f"(warmup={WF_WARMUP_BARS} bars, step={WF_STEP_BARS} bars)")
    print(f"  {'Window':>35s}  {'Train':>6s}  {'Acc':>6s}  {'AUC':>6s}  {'Edge':>6s}")
    print(f"  {'─'*65}")

    for win_idx, (pred_start_i, pred_end_i) in enumerate(windows):
        # Train on everything before this window
        # Use last 20% of training data as early-stopping val
        train_end_i = pred_start_i
        es_split = int(train_end_i * 0.8)

        X_t = train_df[feat_cols].iloc[:es_split]
        y_t = train_df["target"].iloc[:es_split]
        X_v = train_df[feat_cols].iloc[es_split:train_end_i]
        y_v = train_df["target"].iloc[es_split:train_end_i]
        X_p = train_df[feat_cols].iloc[pred_start_i:pred_end_i]
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

    # Rows before warmup: fill with 0.5 (no signal)
    n_no_signal = np.isnan(signal_probs).sum()
    signal_probs = np.where(np.isnan(signal_probs), 0.5, signal_probs)
    print(f"\n  Pre-warmup rows filled with 0.5 (no signal): {n_no_signal}")

    return signal_probs


# ══════════════════════════════════════════════════════════════════════════════
# FINAL MODEL (train on full training set → predict val + test)
# ══════════════════════════════════════════════════════════════════════════════

def train_final_model(df, feat_cols):
    """
    Train on full training set (with val for early stopping).
    Returns model + predictions for val and test.
    """
    train = df[:TRAIN_END]
    val   = df[VAL_START:VAL_END]
    test  = df[TEST_START:]

    print(f"\n{'═'*65}")
    print(f"  FINAL MODEL — trained on full training set")
    print(f"{'═'*65}")
    print(f"  Train : {len(train)} bars")
    print(f"  Val   : {len(val)} bars")
    print(f"  Test  : {len(test)} bars")

    model = train_lgbm(
        train[feat_cols], train["target"],
        val[feat_cols],   val["target"],
    )

    print(f"\n  Trees: {model.best_iteration_}")
    val_probs  = evaluate(model, val[feat_cols],  val["target"],  "VAL ")
    test_probs = evaluate(model, test[feat_cols], test["target"], "TEST")

    # Confidence threshold summary on test
    print(f"\n  Confidence threshold on TEST:")
    print(f"  {'Thresh':>8s}  {'Acc':>6s}  {'Trades':>7s}  {'Coverage':>9s}")
    for thresh in [0.50, 0.54, 0.56, 0.58, 0.60]:
        long_mask  = test_probs > thresh
        short_mask = test_probs < (1 - thresh)
        mask = long_mask | short_mask
        if mask.sum() == 0:
            continue
        preds = np.where(long_mask, 1, 0)[mask]
        acc = accuracy_score(test["target"].values[mask], preds)
        cov = mask.sum() / len(mask)
        print(f"  {thresh:>8.2f}  {acc:>5.1%}  {mask.sum():>7d}  {cov:>8.1%}")

    return model, val_probs, test_probs


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  SIGNAL MODEL TRAINING")
    print("=" * 65)

    df, feat_cols = load_data()

    # ── Step 1: walk-forward predictions for training set ─────────────────
    print(f"\n{'═'*65}")
    print(f"  STEP 1 — Walk-forward predictions (training set)")
    print(f"{'═'*65}")
    train_probs = generate_walkforward_predictions(df, feat_cols)

    # ── Step 2: final model → val + test predictions ──────────────────────
    final_model, val_probs, test_probs = train_final_model(df, feat_cols)

    # ── Step 3: assemble full signal column ───────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  STEP 3 — Assembling signal column")
    print(f"{'═'*65}")

    train_idx = df[:TRAIN_END].index
    val_idx   = df[VAL_START:VAL_END].index
    test_idx  = df[TEST_START:].index

    signal = pd.Series(np.nan, index=df.index, name="signal_prob_1h_feature")
    signal.loc[train_idx] = train_probs
    signal.loc[val_idx]   = val_probs
    signal.loc[test_idx]  = test_probs

    n_nan = signal.isna().sum()
    if n_nan > 0:
        print(f"  [WARN] {n_nan} NaN values in signal — filling with 0.5")
        signal = signal.fillna(0.5)

    print(f"  signal_prob_1h_feature stats:")
    print(f"    mean={signal.mean():.4f}  std={signal.std():.4f}")
    print(f"    min={signal.min():.4f}   max={signal.max():.4f}")
    print(f"    P(signal > 0.60): {(signal > 0.60).mean():.1%} of bars")
    print(f"    P(signal < 0.40): {(signal < 0.40).mean():.1%} of bars")

    # ── Step 4: save model ────────────────────────────────────────────────
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model":        final_model,
            "feat_cols":    feat_cols,
            "exclude":      list(EXCLUDE_FEATURES),
            "horizon":      HORIZON,
            "train_end":    TRAIN_END,
            "lgb_params":   LGB_PARAMS,
        }, f)
    print(f"\n  Model saved → {MODEL_PATH}")

    # ── Step 5: write signal back to parquet ──────────────────────────────
    df_out = pd.read_parquet(DATA_PATH)
    df_out = df_out.iloc[:-HORIZON]   # same trim as df
    df_out["signal_prob_1h_feature"] = signal.values
    df_out.to_parquet(DATA_PATH)
    print(f"  Parquet updated → signal_prob_1h_feature added")
    print(f"  Total features now: {len([c for c in df_out.columns if 'feature' in c])}")

    print(f"\n{'='*65}")
    print(f"  DONE — signal model ready for RL position sizing")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
