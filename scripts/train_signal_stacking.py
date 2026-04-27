#!/usr/bin/env python3
"""
scripts/train_signal_stacking.py — Regime-conditional meta-LGBM over
per-horizon predictions.

Problem the simple ensemble hit
───────────────────────────────
Linear weighted blend of {1h, 4h, 24h} LGBM probs:
  - OOF AUC:  1h 0.562, 4h 0.538, 24h 0.573
  - TEST AUC: 1h 0.524, 4h 0.516, 24h 0.477  ← 24h FLIPS SIGN on TEST
OOF-weighted blend gave 24h the highest weight (0.507), so it dragged
down TEST. Fixed weights can't condition on regime.

What stacking does
──────────────────
Train a meta-LGBM:
  inputs  = [p_1h, p_4h, p_24h, trend_180, volatility_180, turbulence,
             funding_rate_zscore, rsi_12, bb_width_42]
  target  = sign(next_bar return)  (same as base 1h)
  output  = p_stack (the final signal)

The meta learns (we hope) that 24h helps only when trend_180 is strong
and turbulence is low — i.e. stable bull/bear regimes. In chop it
will down-weight 24h automatically.

Walk-forward: base models are fit on each window as before, but the
meta is ALWAYS trained on OOF base predictions (never on in-sample) —
this is the only way stacking is leakage-free.

Output column: `signal_prob_v2_stack_feature`
Model artefact: `models/signal_lgbm_v2_stacking.pkl`

Usage
─────
    python scripts/train_signal_stacking.py
    python scripts/train_signal_stacking.py --horizons 1,4,24
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

# ── Config mirrors retrain_signal_v2 / train_signal_ensemble ──────────
DATA_PATH    = "cache/data_v1_430340a861af5f8f9bcbd7a4ca16ba95.parquet"
ETH_PATH     = "../../data/binance/ETH_USDT-1h.feather"
FUNDING_PATH = "../../data/binance/futures/BTC_USDT_USDT-8h-funding_rate.feather"
MODEL_PATH   = "models/signal_lgbm_v2_stacking.pkl"
OUTPUT_COL   = "signal_prob_v2_stack_feature"

TRAIN_END  = "2024-06-30"
VAL_START  = "2024-07-01"
VAL_END    = "2025-06-30"
TEST_START = "2025-07-01"

WF_WARMUP_BARS = 4_380
WF_STEP_BARS   = 1_095

# Regime features fed to the meta-model. Picked for interpretability:
# trend/vol/turbulence/funding/rsi capture the "what kind of market is this".
REGIME_FEATS = [
    "trend_return_180_feature",
    "volatility_180_feature",
    "funding_rate_zscore_feature",
    "rsi_12_feature",
    "bb_width_42_feature",
]
# Added automatically if present in parquet (reflects live Freqtrade pipeline):
OPTIONAL_REGIME_FEATS = ["turbulence_feature"]

LGB_PARAMS_BASE = {
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

# Meta-model has FAR fewer features (≈3 horizons + 5-6 regime) and far
# fewer natural splits → smaller is better. Overfitting here is the
# biggest risk.
LGB_PARAMS_META = {
    **LGB_PARAMS_BASE,
    "n_estimators":      200,
    "max_depth":         3,
    "num_leaves":        7,
    "min_child_samples": 100,
    "learning_rate":     0.05,
    "colsample_bytree":  0.9,
}


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

    print(f"  Computing {len(SIGNAL_FEAT_COLS_V2)} base features …")
    X_full = build_feature_matrix(df, eth_df=eth_df, funding_series=funding)
    df[SIGNAL_FEAT_COLS_V2] = X_full[SIGNAL_FEAT_COLS_V2].values
    return df


def add_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    fut = np.log(df["close"].shift(-horizon) / df["close"])
    df[f"target_h{horizon}"] = (fut > 0).astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# BASE-LEVEL WALK-FORWARD (per horizon) — same as ensemble script, returns
# OOF preds AND the final-window model for VAL/TEST inference.
# ══════════════════════════════════════════════════════════════════════════════

def train_lgbm(X_tr, y_tr, X_val, y_val, params):
    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def walkforward_oof(df, horizon, feat_cols):
    """Zero-leakage walk-forward OOF probs on TRAIN + final model on full TRAIN."""
    target_col = f"target_h{horizon}"
    tr = df[:TRAIN_END].iloc[:-horizon].copy()
    n = len(tr)
    oof = np.full(n, np.nan)
    aucs = []

    windows = []
    i = WF_WARMUP_BARS
    while i < n:
        windows.append((i, min(i + WF_STEP_BARS, n)))
        i += WF_STEP_BARS

    print(f"  [h={horizon}h] walk-forward: {len(windows)} windows")
    for ps, pe in windows:
        es = int(ps * 0.8)
        X_t = tr[feat_cols].iloc[:es]
        y_t = tr[target_col].iloc[:es]
        X_v = tr[feat_cols].iloc[es:ps]
        y_v = tr[target_col].iloc[es:ps]
        X_p = tr[feat_cols].iloc[ps:pe]
        y_p = tr[target_col].iloc[ps:pe]
        m = train_lgbm(X_t, y_t, X_v, y_v, LGB_PARAMS_BASE)
        p = m.predict_proba(X_p)[:, 1]
        oof[ps:pe] = p
        if y_p.nunique() > 1:
            aucs.append(roc_auc_score(y_p, p))

    mean_auc = float(np.mean(aucs)) if aucs else 0.5
    print(f"  [h={horizon}h] OOF mean AUC = {mean_auc:.4f}")

    # Final base model (trained on all of TRAIN with VAL early-stop) for VAL/TEST inference
    val   = df[VAL_START:VAL_END].iloc[:-horizon]
    test  = df[TEST_START:].iloc[:-horizon]
    if len(val) < 100 or len(test) < 50:
        val  = df[VAL_START:VAL_END]; test = df[TEST_START:]
    full_tr = df[:TRAIN_END].iloc[:-horizon]
    final_m = train_lgbm(full_tr[feat_cols], full_tr[target_col],
                         val[feat_cols],      val[target_col],
                         LGB_PARAMS_BASE)

    return oof, tr.index, final_m, (val, test)


# ══════════════════════════════════════════════════════════════════════════════
# META-LEVEL STACKING
# ══════════════════════════════════════════════════════════════════════════════

def build_meta_df(df, oof_probs_by_h, oof_index, regime_feats):
    """Align OOF probs + regime features + target into a single meta-training df.
    Only uses rows where all three base horizons have OOF preds (no NaN)."""
    meta = pd.DataFrame(index=oof_index)
    for h, oof in oof_probs_by_h.items():
        meta[f"p_h{h}"] = oof
    for col in regime_feats:
        meta[col] = df.loc[oof_index, col].values
    meta["target"] = df.loc[oof_index, "target_h1"].values

    # Drop warmup rows (NaN in any oof column)
    pre = len(meta)
    meta = meta.dropna(subset=[f"p_h{h}" for h in oof_probs_by_h]).copy()
    print(f"\n  Meta training rows: {len(meta)} "
          f"(dropped {pre - len(meta)} pre-warmup rows)")
    return meta


def inference_preds(base_models, df_split, feat_cols, horizons):
    """Run each horizon's FINAL model on a split and return {h: probs_aligned_to_df_split}."""
    out = {}
    for h in horizons:
        m = base_models[h]
        out[h] = m.predict_proba(df_split[feat_cols])[:, 1]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=str, default="1,4,24")
    ap.add_argument("--parquet",  type=str, default=DATA_PATH)
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    print("=" * 65)
    print(f"  Regime-conditional stacking — base horizons = {horizons}")
    print("=" * 65)

    df = load_data()
    for h in horizons:
        add_target(df, h)

    # Regime features: take REGIME_FEATS + any OPTIONAL_REGIME_FEATS present
    regime_feats = list(REGIME_FEATS)
    for c in OPTIONAL_REGIME_FEATS:
        if c in df.columns:
            regime_feats.append(c)
    missing = [c for c in regime_feats if c not in df.columns]
    if missing:
        raise RuntimeError(f"Regime features missing from parquet: {missing}")
    print(f"  Regime feats ({len(regime_feats)}): {regime_feats}")

    feat_cols = SIGNAL_FEAT_COLS_V2
    print(f"  Base feats  : {len(feat_cols)}")

    # ── Base walk-forward per horizon ─────────────────────────────────────
    oof_by_h    = {}
    oof_index   = None
    base_models = {}
    split_frames = {}

    for h in horizons:
        print(f"\n── Horizon {h}h ──────────────────────────────────────────")
        oof, tr_idx, final_m, (val, test) = walkforward_oof(df, h, feat_cols)
        if oof_index is None:
            oof_index = tr_idx
        else:
            # Truncate to shortest TR index (different horizons trim differently)
            common = oof_index.intersection(tr_idx)
            oof_index = common
        oof_by_h[h]    = oof
        base_models[h] = final_m
        split_frames[h] = (val, test)

    # Re-align OOF arrays to common index
    aligned_oof = {}
    for h in horizons:
        full = pd.Series(oof_by_h[h],
                         index=df[:TRAIN_END].iloc[:-h].index)
        aligned_oof[h] = full.reindex(oof_index).values

    # ── Meta-model training ───────────────────────────────────────────────
    meta_df = build_meta_df(df, aligned_oof, oof_index, regime_feats)
    meta_feats = [f"p_h{h}" for h in horizons] + regime_feats

    # 80/20 split inside meta-train for early stopping
    cutoff = int(len(meta_df) * 0.85)
    X_mt, y_mt = meta_df[meta_feats].iloc[:cutoff], meta_df["target"].iloc[:cutoff]
    X_mv, y_mv = meta_df[meta_feats].iloc[cutoff:], meta_df["target"].iloc[cutoff:]
    print(f"  Meta-train: {len(X_mt)}   Meta-val (early stop): {len(X_mv)}")

    meta = lgb.LGBMClassifier(**LGB_PARAMS_META)
    meta.fit(X_mt, y_mt,
             eval_set=[(X_mv, y_mv)],
             callbacks=[lgb.early_stopping(30, verbose=False)])
    print(f"  Meta trees: {meta.best_iteration_}")

    # Feature importance (gain) — tells us what the meta learned to rely on
    print(f"\n  Meta importance (top):")
    imp = sorted(zip(meta_feats, meta.feature_importances_),
                 key=lambda x: -x[1])
    for f, i in imp:
        bar = "█" * int(i / max(1, imp[0][1]) * 30)
        print(f"    {i:>5d}  {bar:<30s}  {f}")

    # ── Meta inference: generate p_stack for all splits ───────────────────
    def _meta_predict(df_sub):
        """Build the meta inputs for an arbitrary slice and predict."""
        base_p = inference_preds(base_models, df_sub, feat_cols, horizons)
        X = pd.DataFrame(index=df_sub.index)
        for h in horizons:
            X[f"p_h{h}"] = base_p[h]
        for col in regime_feats:
            X[col] = df_sub[col].values
        return meta.predict_proba(X[meta_feats])[:, 1]

    print(f"\n{'─'*65}")
    print(f"  Per-split stacking results")
    print(f"{'─'*65}")

    out_series = pd.Series(0.5, index=df.index, name=OUTPUT_COL)

    # TRAIN: reuse OOF meta-predictions for reporting (can't use meta on TRAIN data
    # it was fit on; expose out-of-fold-only evaluation). For parquet output we
    # use in-sample meta preds on TRAIN — only base models are held-out leakage-wise.
    # For honest reporting we report the 20% holdout AUC (y_mv).
    holdout_probs = meta.predict_proba(X_mv)[:, 1]
    if y_mv.nunique() > 1:
        auc = roc_auc_score(y_mv, holdout_probs)
        acc = accuracy_score(y_mv, (holdout_probs > 0.5).astype(int))
        print(f"    [meta holdout]  acc={acc:.3f}  AUC={auc:.4f}  n={len(y_mv)}")

    for name, slice_df in [("VAL",  df[VAL_START:VAL_END]),
                           ("TEST", df[TEST_START:])]:
        # For final scored rows we trim the last 1 bar (matches target horizon)
        s = slice_df.iloc[:-1].copy()
        probs = _meta_predict(s)
        # Score on target_h1
        y = s["target_h1"].values
        if len(set(y)) > 1:
            auc = roc_auc_score(y, probs)
            acc = accuracy_score(y, (probs > 0.5).astype(int))
            print(f"    [{name}]  acc={acc:.3f}  AUC={auc:.4f}  "
                  f"mean={probs.mean():.4f} std={probs.std():.4f}  "
                  f"P(>0.6)={(probs>0.6).mean():.1%}  P(<0.4)={(probs<0.4).mean():.1%}")
        out_series.loc[s.index] = probs

    # TRAIN bars: in-sample meta preds so the parquet column is complete for RL
    train_slice = df[:TRAIN_END].iloc[:-1].copy()
    train_probs = _meta_predict(train_slice)
    out_series.loc[train_slice.index] = train_probs

    # ── Write back to parquet ─────────────────────────────────────────────
    df_out = pd.read_parquet(args.parquet)
    n = min(len(df_out), len(out_series))
    df_out = df_out.iloc[:n].copy()
    df_out[OUTPUT_COL] = out_series.values[:n]
    df_out.to_parquet(args.parquet)
    print(f"\n  Parquet updated → {OUTPUT_COL}")

    # ── Save artefacts ────────────────────────────────────────────────────
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "horizons":       horizons,
            "base_models":    base_models,
            "meta_model":     meta,
            "feat_cols":      feat_cols,
            "meta_feats":     meta_feats,
            "regime_feats":   regime_feats,
            "lgb_params_base": LGB_PARAMS_BASE,
            "lgb_params_meta": LGB_PARAMS_META,
            "train_end":      TRAIN_END,
            "version":        "v2_stacking",
        }, f)
    print(f"  Artefacts → {MODEL_PATH}")

    print(f"\n  Next: calibrate with")
    print(f"    python scripts/calibrate_signal_v2.py "
          f"--source-col {OUTPUT_COL} "
          f"--target-col signal_prob_v2_stack_cal_feature")


if __name__ == "__main__":
    main()
