#!/usr/bin/env python3
"""
scripts/train_signal_ensemble.py — Multi-horizon LGBM ensemble signal
══════════════════════════════════════════════════════════════════════

Why multiple horizons
─────────────────────
A single 1h-horizon model captures tactical edge but is dominated by
micro-noise (accuracy ~53%, AUC ~0.52 on TEST). Adding longer horizons
(4h, 24h) brings:
  - 4h  : short-trend edge, less noise, still reactive
  - 24h : daily regime — high signal-to-noise, captures the
          big-picture direction that overrides tactical whipsaws

Fused together the ensemble produces a signal that's:
  - agreeable across horizons → high confidence setups
  - split across horizons     → noisy setups, near 0.5 → RL sits out

Combination
───────────
We use logit-space weighted average (NOT arithmetic):
    p_ens = sigmoid( sum(w_h * logit(p_h)) / sum(w_h) )
Weights `w_h` come from OOF accuracy on TRAIN — horizons that
generalise better get more voice. Logit-space keeps additive the
"information content" of each prediction.

Output
──────
Writes `signal_prob_v2_ens_feature` to the parquet (raw ensemble
probability). Run `scripts/calibrate_signal_v2.py` afterwards to
produce `signal_prob_v2_ens_cal_feature` with stretched std.

Model artefact at `models/signal_lgbm_v2_ensemble.pkl` contains:
  - one LGBM per horizon
  - feat_cols, weights, horizons
  - metrics per horizon for provenance

Usage
─────
    python scripts/train_signal_ensemble.py
    python scripts/train_signal_ensemble.py --horizons 1,4,24 --no-walkforward
"""

import argparse
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from utils.signal_features import (
    SIGNAL_FEAT_COLS_V2,
    build_feature_matrix,
    load_funding_series,
)

# ── Config mirrors retrain_signal_v2.py for consistency ──────────────────────
DATA_PATH    = "cache/data_v1_430340a861af5f8f9bcbd7a4ca16ba95.parquet"
ETH_PATH     = "../../data/binance/ETH_USDT-1h.feather"
FUNDING_PATH = "../../data/binance/futures/BTC_USDT_USDT-8h-funding_rate.feather"
MODEL_PATH   = "models/signal_lgbm_v2_ensemble.pkl"

TRAIN_END  = "2024-06-30"
VAL_START  = "2024-07-01"
VAL_END    = "2025-06-30"
TEST_START = "2025-07-01"

WF_WARMUP_BARS = 4_380   # ~6 months
WF_STEP_BARS   = 1_095   # ~45 days

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

EPS = 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    print(f"  Loading parquet: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    eth_df = pd.DataFrame()
    eth_p  = Path(ETH_PATH)
    if eth_p.exists():
        eth_df = pd.read_feather(str(eth_p))
        if "date" in eth_df.columns:
            eth_df = eth_df.set_index("date")
        eth_df.index = pd.to_datetime(eth_df.index, utc=True).tz_localize(None)

    funding = load_funding_series(FUNDING_PATH)
    if funding is not None:
        funding.index = funding.index.tz_localize(None)

    print(f"  Computing {len(SIGNAL_FEAT_COLS_V2)} features …")
    X_full = build_feature_matrix(df, eth_df=eth_df, funding_series=funding)
    df[SIGNAL_FEAT_COLS_V2] = X_full[SIGNAL_FEAT_COLS_V2].values
    return df


def add_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Adds `target_h{H}` column — sign of log return H bars ahead."""
    fut = np.log(df["close"].shift(-horizon) / df["close"])
    df[f"target_h{horizon}"] = (fut > 0).astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# LGBM PER-HORIZON
# ══════════════════════════════════════════════════════════════════════════════

def train_lgbm(X_train, y_train, X_val, y_val, lgb_params=None):
    params = dict(lgb_params or LGB_PARAMS)
    model = lgb.LGBMClassifier(**params)
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
    auc   = roc_auc_score(y, probs) if y.nunique() > 1 else 0.5
    ll    = log_loss(y, probs)
    edge  = acc - y.mean()
    print(f"    [{label}]  acc={acc:.3f} (edge={edge:+.3f})  AUC={auc:.4f}  logloss={ll:.4f}")
    return probs, auc


def walkforward_oof(df: pd.DataFrame, horizon: int, feat_cols: list) -> tuple:
    """Out-of-fold predictions for TRAIN, zero lookahead.
    Also returns average OOF AUC — used to weight this horizon in the blend.
    """
    train_df   = df[:TRAIN_END].copy()
    target_col = f"target_h{horizon}"
    if target_col not in train_df.columns:
        raise RuntimeError(f"{target_col} missing; add_target(df, {horizon}) first")

    # Trim rows whose target uses future info past train boundary-safe region.
    # Simplest: drop the last `horizon` bars.
    train_df = train_df.iloc[:-horizon].copy()
    n        = len(train_df)

    preds  = np.full(n, np.nan)
    aucs   = []
    windows = []
    i = WF_WARMUP_BARS
    while i < n:
        windows.append((i, min(i + WF_STEP_BARS, n)))
        i += WF_STEP_BARS

    print(f"  [horizon={horizon}h] walk-forward: {len(windows)} windows")
    for pred_start, pred_end in windows:
        es_split = int(pred_start * 0.8)
        X_t = train_df[feat_cols].iloc[:es_split]
        y_t = train_df[target_col].iloc[:es_split]
        X_v = train_df[feat_cols].iloc[es_split:pred_start]
        y_v = train_df[target_col].iloc[es_split:pred_start]
        X_p = train_df[feat_cols].iloc[pred_start:pred_end]
        y_p = train_df[target_col].iloc[pred_start:pred_end]

        model = train_lgbm(X_t, y_t, X_v, y_v)
        probs = model.predict_proba(X_p)[:, 1]
        preds[pred_start:pred_end] = probs
        if y_p.nunique() > 1:
            aucs.append(roc_auc_score(y_p, probs))

    # Align full-training-set preds array back to full df index (NaN for val/test slots)
    full_len = len(df)
    padded   = np.full(full_len, np.nan)
    train_full_idx = df.index.get_indexer(train_df.index)
    padded[train_full_idx] = preds

    mean_auc = float(np.mean(aucs)) if aucs else 0.5
    print(f"  [horizon={horizon}h] WF mean AUC = {mean_auc:.4f}  "
          f"({sum(np.isnan(preds))} pre-warmup bars filled later)")
    return padded, mean_auc


def train_final_and_predict(df: pd.DataFrame, horizon: int, feat_cols: list):
    """Train final model on full TRAIN with VAL early-stopping → predict VAL+TEST."""
    target_col = f"target_h{horizon}"
    train = df[:TRAIN_END].iloc[:-horizon]
    val   = df[VAL_START:VAL_END].iloc[:-horizon]   # keep matching trim
    test  = df[TEST_START:].iloc[:-horizon]

    # If `horizon` trims too aggressively on small splits, fall back to no trim
    if len(val) < 100 or len(test) < 50:
        val  = df[VAL_START:VAL_END]
        test = df[TEST_START:]

    model = train_lgbm(train[feat_cols], train[target_col],
                       val[feat_cols],   val[target_col])

    print(f"  [horizon={horizon}h] final model (trees={model.best_iteration_})")
    val_probs,  val_auc  = evaluate(model, val[feat_cols],  val[target_col],  "VAL ")
    test_probs, test_auc = evaluate(model, test[feat_cols], test[target_col], "TEST")

    # Align val/test preds onto full df index
    full_len = len(df)
    padded   = np.full(full_len, np.nan)
    val_idx  = df.index.get_indexer(val.index)
    test_idx = df.index.get_indexer(test.index)
    padded[val_idx]  = val_probs
    padded[test_idx] = test_probs

    return model, padded, val_auc, test_auc


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE COMBINATION
# ══════════════════════════════════════════════════════════════════════════════

def blend_logits(horizon_probs: dict, weights: dict) -> np.ndarray:
    """Weighted logit-space average across horizons.
    horizon_probs: {h: np.ndarray of shape (N,)}
    weights:       {h: float (positive)}
    """
    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError("All weights are zero — aborting.")
    logit_blend = np.zeros(next(iter(horizon_probs.values())).shape)
    for h, probs in horizon_probs.items():
        p = np.clip(probs, EPS, 1 - EPS)
        logit_blend += (weights[h] / total_w) * np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-logit_blend))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=str, default="1,4,24",
                    help="Comma-separated horizons in bars (1h each). Default: 1,4,24")
    ap.add_argument("--no-walkforward", action="store_true",
                    help="Skip walk-forward OOF on TRAIN. Fills TRAIN with final-model "
                         "in-sample predictions — FOR DEBUG ONLY (leakage risk).")
    ap.add_argument("--parquet", type=str, default=DATA_PATH)
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    print("=" * 65)
    print(f"  Multi-horizon LGBM ensemble — horizons = {horizons}")
    print("=" * 65)

    df = load_data()
    for h in horizons:
        add_target(df, h)

    # Trim to the max horizon so targets are always available for all models.
    max_h = max(horizons)
    df_aligned = df.iloc[:-max_h].copy()   # only for train/eval bookkeeping

    feat_cols = SIGNAL_FEAT_COLS_V2
    print(f"  Features  : {len(feat_cols)}")
    print(f"  Total rows: {len(df)}  (aligned-to-max-h: {len(df_aligned)})")

    # ── Per-horizon: WF OOF (TRAIN) + final model (VAL/TEST) ─────────────
    oof_probs     = {}
    oof_aucs      = {}
    final_probs   = {}
    final_models  = {}
    metrics_store = {}

    for h in horizons:
        print(f"\n── Horizon {h}h ──────────────────────────────────────────")
        if args.no_walkforward:
            # Debug path — in-sample preds on TRAIN (LEAKAGE! only for quick tests)
            train_df = df[:TRAIN_END].iloc[:-h]
            oof_auc = 0.5
            full = np.full(len(df), np.nan)
            oof_probs[h] = full
        else:
            oof_full, oof_auc = walkforward_oof(df, h, feat_cols)
            oof_probs[h] = oof_full
        oof_aucs[h] = oof_auc

        model, split_full, val_auc, test_auc = train_final_and_predict(df, h, feat_cols)
        final_probs[h]  = split_full
        final_models[h] = model
        metrics_store[h] = {
            "oof_auc":  oof_auc,
            "val_auc":  val_auc,
            "test_auc": test_auc,
        }

    # ── Stitch TRAIN (OOF) + VAL+TEST (final) per horizon ─────────────────
    stitched = {}
    for h in horizons:
        s = oof_probs[h].copy()
        mask = ~np.isnan(final_probs[h])
        s[mask] = final_probs[h][mask]
        # Any bars with no prediction at all (pre-warmup + last max_h bars) → 0.5
        s[np.isnan(s)] = 0.5
        stitched[h] = s

    # ── Weights from OOF AUC (boosted 2× above 0.5) ───────────────────────
    # Heuristic: w = max(oof_auc - 0.5, 0.01)²
    raw_w = {h: max(oof_aucs[h] - 0.5, 0.01) ** 2 for h in horizons}
    total_w = sum(raw_w.values())
    weights = {h: raw_w[h] / total_w for h in horizons}
    print(f"\n  OOF AUCs   : {{ {', '.join(f'{h}h: {oof_aucs[h]:.4f}' for h in horizons)} }}")
    print(f"  Blend wts  : {{ {', '.join(f'{h}h: {weights[h]:.3f}' for h in horizons)} }}")

    # ── Blend ─────────────────────────────────────────────────────────────
    p_ens = blend_logits(stitched, weights)
    print(f"\n  Ensemble stats:")
    for lbl, idx in [("TRAIN", df.index <= TRAIN_END),
                     ("VAL",   (df.index >= VAL_START) & (df.index <= VAL_END)),
                     ("TEST",  df.index >= TEST_START)]:
        s = p_ens[idx]
        print(f"    {lbl:<5s}  mean={s.mean():.4f} std={s.std():.4f} "
              f"P(>0.60)={(s>0.60).mean():.1%} P(<0.40)={(s<0.40).mean():.1%}")

    # ── Persist to parquet ────────────────────────────────────────────────
    df_out = pd.read_parquet(args.parquet)
    # Align lengths — parquet on disk may or may not be the trimmed version
    n_write = min(len(df_out), len(p_ens))
    df_out = df_out.iloc[:n_write].copy()
    df_out["signal_prob_v2_ens_feature"] = p_ens[:n_write]
    df_out.to_parquet(args.parquet)
    print(f"\n  Parquet updated → signal_prob_v2_ens_feature")

    # ── Save models ───────────────────────────────────────────────────────
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "horizons":   horizons,
            "models":     final_models,
            "weights":    weights,
            "feat_cols":  feat_cols,
            "lgb_params": LGB_PARAMS,
            "metrics":    metrics_store,
            "train_end":  TRAIN_END,
            "version":    "v2_ensemble",
        }, f)
    print(f"  Models saved → {MODEL_PATH}")

    print(f"\n{'═' * 65}")
    print(f"  Next step: scripts/calibrate_signal_v2.py "
          f"--source-col signal_prob_v2_ens_feature "
          f"--target-col signal_prob_v2_ens_cal_feature")
    print(f"{'═' * 65}")


if __name__ == "__main__":
    main()
