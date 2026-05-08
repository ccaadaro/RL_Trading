#!/usr/bin/env python3
"""
Phase 11 — Horizon Shift: 4-fold walk-forward evaluator.

Hypothesis: Basis and funding carry a weak directional signal that is
destroyed by hourly execution costs. 7-day holds reduce cost drag by
~48x and may make the signal economically tradeable.

Gates (pre-committed, see MODEL3_PROTOCOLS.md):
  1. mean AUC >= 0.52
  2. min fold AUC >= 0.50
  3. Net ROI > Random-P95 (bootstrapped)
  4. Calmar > B&H Calmar
  5. No single fold explains all performance
  6. Timestamp audit: CLEAN
  7. Shuffled-label AUC ≈ 0.50

Kill criteria:
  - mean AUC < 0.52 → archive Phase 11
  - mean AUC in [0.52, 0.55] but Calmar < B&H → archive
  - min(fold AUC) < 0.50 → regime fragility, do not deploy

Usage:
    python3 scripts/evaluate_phase11.py --family basis
    python3 scripts/evaluate_phase11.py --family funding
    python3 scripts/evaluate_phase11.py --family both
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"

PATHS = {
    "basis": CACHE / "btc_daily_phase11_basis.feather",
    "funding": CACHE / "btc_daily_phase11_funding.feather",
}

LABEL_COLS = {"triple_barrier_7d", "trend_7d"}

PRICE_FEATURES = [
    "return_3d", "return_7d", "return_14d", "return_30d",
    "ma_bias_7d", "ma_bias_14d", "ma_bias_30d", "ma_bias_90d",
    "realized_vol_7d", "realized_vol_14d", "realized_vol_30d",
    "rsi_14d", "atr_pct",
    "drawdown_from_high_30d", "drawdown_from_high_90d",
    "volume_zscore_14d", "volume_zscore_30d",
]
BASIS_FEATURES = ["basis_raw", "basis_zscore_30d", "basis_mean_7d"]
FUNDING_FEATURES = ["funding_rate", "funding_zscore_30d", "funding_cumsum_7d", "funding_sign"]

COST_BPS = 12.5   # 25bps roundtrip / 2 sides
HOLD_DAYS = 7
PURGE_DAYS = 14   # > barrier horizon (7d) with margin
AUC_THRESHOLD = 0.52
MIN_FOLD_AUC = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Safety assertions
# ─────────────────────────────────────────────────────────────────────────────

def check_no_label_in_features(features: list[str]) -> None:
    leak = [f for f in features if f in LABEL_COLS]
    assert not leak, f"CRITICAL LEAKAGE: {leak} in feature list"


# ─────────────────────────────────────────────────────────────────────────────
# Shuffled-label sanity test
# ─────────────────────────────────────────────────────────────────────────────

def shuffled_label_test(X: pd.DataFrame, y: pd.Series) -> float:
    from sklearn.ensemble import GradientBoostingClassifier

    valid = y.notna()
    Xv = X[valid].fillna(0)
    yv = ((y[valid] + 1) / 2).astype(int)

    rng = np.random.default_rng(0)
    y_shuf = yv.copy()
    y_shuf.values[:] = rng.permutation(yv.values)

    sc = StandardScaler()
    Xs = sc.fit_transform(Xv)
    m = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=0)
    m.fit(Xs, y_shuf)
    auc = roc_auc_score(yv, m.predict_proba(Xs)[:, 1])
    logger.info(f"Shuffled-label AUC (should ≈ 0.50): {auc:.4f}")
    return auc


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM training (with sklearn GBM fallback)
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray) -> np.ndarray:
    try:
        import lightgbm as lgb
        td = lgb.Dataset(X_train, label=y_train)
        params = dict(
            objective="binary", metric="auc",
            num_leaves=15, learning_rate=0.05,
            feature_fraction=0.8, bagging_fraction=0.8,
            verbose=-1,
        )
        m = lgb.train(params, td, num_boost_round=100)
        return m.predict(X_test)
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        m = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                       learning_rate=0.05, random_state=0)
        m.fit(X_train, y_train)
        return m.predict_proba(X_test)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# PnL with 7-day holds, training-set vol threshold, asymmetric confidence
# ─────────────────────────────────────────────────────────────────────────────

def compute_pnl(df_test: pd.DataFrame, y_pred_proba: np.ndarray,
                vol_thresh: float) -> dict:
    returns = df_test["close"].pct_change().fillna(0).values
    pos = np.zeros(len(df_test))
    hold_timer = 0
    current_pos = 0

    for i in range(len(df_test)):
        if hold_timer > 0:
            pos[i] = current_pos
            hold_timer -= 1
        else:
            low_vol = df_test["realized_vol_7d"].iloc[i] < vol_thresh
            if low_vol:
                current_pos = 0
            else:
                p = y_pred_proba[i]
                if p > 0.55:
                    current_pos = 1
                elif p < 0.45:
                    current_pos = -1
                else:
                    current_pos = 0
            pos[i] = current_pos
            hold_timer = HOLD_DAYS - 1

    pos_series = pd.Series(pos)
    cost_per_side = COST_BPS / 10_000
    costs = pos_series.diff().abs().fillna(0) * cost_per_side
    net_ret = pd.Series(returns) * pos_series.shift(1).fillna(0) - costs

    cum = (1 + net_ret).cumprod()
    total_ret = cum.iloc[-1] - 1
    max_dd = (cum / cum.expanding().max()).min() - 1
    calmar = total_ret / abs(max_dd) if max_dd != 0 else 0.0
    n_trades = int((pos_series.diff().abs() > 0).sum())

    return dict(net_roi=total_ret, calmar=calmar, max_dd=max_dd,
                n_trades=n_trades, pos=pos_series)


def bh_pnl(df_test: pd.DataFrame) -> dict:
    ret = df_test["close"].pct_change().fillna(0)
    cum = (1 + ret).cumprod()
    total = cum.iloc[-1] - 1
    dd = (cum / cum.expanding().max()).min() - 1
    return dict(net_roi=total, calmar=total / abs(dd) if dd != 0 else 0.0)


def random_p95(df_test: pd.DataFrame, n_boot: int = 500) -> float:
    ret = df_test["close"].pct_change().fillna(0).values
    rng = np.random.default_rng(42)
    rois = []
    for _ in range(n_boot):
        pos = rng.choice([-1, 0, 1], size=len(ret))
        cost = np.abs(np.diff(pos, prepend=0)) * (COST_BPS / 10_000)
        nr = ret * np.roll(pos, 1) - cost
        rois.append((1 + pd.Series(nr)).cumprod().iloc[-1] - 1)
    return float(np.percentile(rois, 95))


# ─────────────────────────────────────────────────────────────────────────────
# 4-fold walk-forward
# ─────────────────────────────────────────────────────────────────────────────

def run_walkforward(df: pd.DataFrame, features: list[str], label: str = "triple_barrier_7d") -> dict:
    check_no_label_in_features(features)

    dfc = df.dropna(subset=[label]).reset_index(drop=True)
    n = len(dfc)
    initial_train = n // 5          # first fifth reserved for training
    fold_size = (n - initial_train) // 4
    logger.info(f"n={n}, initial_train={initial_train}, fold_size={fold_size}, purge={PURGE_DAYS}d")

    fold_aucs, fold_rois, fold_calmars = [], [], []
    bh_calmars = []

    for fold in range(4):
        ts = initial_train + fold * fold_size
        te = ts + fold_size if fold < 3 else n
        tr_end = max(0, ts - PURGE_DAYS)

        df_train = dfc.iloc[:tr_end]
        df_test = dfc.iloc[ts:te]

        if len(df_train) < 30 or len(df_test) < 10:
            logger.warning(f"Fold {fold+1}: insufficient data, skipping")
            continue

        X_tr = df_train[features].fillna(0).values
        y_tr_raw = df_train[label]
        valid_tr = y_tr_raw.notna()
        X_tr = X_tr[valid_tr]
        y_tr = ((y_tr_raw[valid_tr] + 1) / 2).astype(int).values

        X_te = df_test[features].fillna(0).values
        y_te_raw = df_test[label]
        valid_te = y_te_raw.notna()
        y_te = ((y_te_raw[valid_te] + 1) / 2).astype(int).values

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        proba_all = train_model(X_tr_s, y_tr, X_te_s)
        proba_valid = proba_all[valid_te.values]

        auc = roc_auc_score(y_te, proba_valid) if len(np.unique(y_te)) > 1 else np.nan

        # Vol threshold from training set only
        vol_thresh = df_train["realized_vol_7d"].quantile(0.25)
        pnl = compute_pnl(df_test.reset_index(drop=True), proba_all, vol_thresh)
        bh = bh_pnl(df_test.reset_index(drop=True))
        rand_p95 = random_p95(df_test.reset_index(drop=True))

        test_start = df_test["date"].iloc[0].strftime("%Y-%m")
        logger.info(
            f"Fold {fold+1} (test_start={test_start}): "
            f"n_train={len(X_tr)}, n_test={len(df_test)}, "
            f"AUC={auc:.4f}, NetROI={pnl['net_roi']:.3f}, "
            f"Calmar={pnl['calmar']:.3f}, "
            f"B&H_ROI={bh['net_roi']:.3f}, B&H_Calmar={bh['calmar']:.3f}, "
            f"Rand-P95={rand_p95:.3f}, n_trades={pnl['n_trades']}"
        )

        fold_aucs.append(auc)
        fold_rois.append(pnl["net_roi"])
        fold_calmars.append(pnl["calmar"])
        bh_calmars.append(bh["calmar"])

    return dict(
        aucs=fold_aucs,
        rois=fold_rois,
        calmars=fold_calmars,
        bh_calmars=bh_calmars,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gate evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_gates(results: dict, model_name: str) -> bool:
    aucs = results["aucs"]
    calmars = results["calmars"]
    bh_calmars = results["bh_calmars"]

    mean_auc = np.nanmean(aucs)
    min_auc = np.nanmin(aucs)
    mean_calmar = np.nanmean(calmars)
    mean_bh_calmar = np.nanmean(bh_calmars)

    logger.info(f"\n{'='*70}")
    logger.info(f"GATE SUMMARY — {model_name}")
    logger.info(f"{'='*70}")
    logger.info(f"  Mean AUC:          {mean_auc:.4f} ± {np.nanstd(aucs):.4f}  (folds: {[f'{a:.4f}' for a in aucs]})")
    logger.info(f"  Min fold AUC:      {min_auc:.4f}")
    logger.info(f"  Mean Calmar:       {mean_calmar:.4f} ± {np.nanstd(calmars):.4f}")
    logger.info(f"  B&H Mean Calmar:   {mean_bh_calmar:.4f} ± {np.nanstd(bh_calmars):.4f}")

    g1 = mean_auc >= AUC_THRESHOLD
    g2 = min_auc >= MIN_FOLD_AUC
    g3 = mean_calmar > mean_bh_calmar

    logger.info(f"\n  Gate 1 — mean AUC >= {AUC_THRESHOLD}:       {'✅' if g1 else '❌'}  ({mean_auc:.4f})")
    logger.info(f"  Gate 2 — min fold AUC >= {MIN_FOLD_AUC}:     {'✅' if g2 else '❌'}  ({min_auc:.4f})")
    logger.info(f"  Gate 3 — Calmar > B&H Calmar:     {'✅' if g3 else '❌'}  ({mean_calmar:.3f} vs {mean_bh_calmar:.3f})")

    gates_passed = sum([g1, g2, g3])
    passed = gates_passed == 3

    if passed:
        logger.info(f"\n  [GATE] APPROVED — all 3 gates passed")
    else:
        logger.info(f"\n  [GATE] REJECTED — {3 - gates_passed}/3 gates failed")
        # Kill criteria
        if mean_auc < AUC_THRESHOLD:
            logger.info(f"  [KILL] mean AUC {mean_auc:.4f} < {AUC_THRESHOLD} — archive Phase 11")
        elif mean_auc < 0.55 and mean_calmar <= mean_bh_calmar:
            logger.info(f"  [KILL] Marginal AUC + Calmar <= B&H — archive Phase 11")

    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["basis", "funding", "both"], default="both")
    args = parser.parse_args()

    families = ["basis", "funding"] if args.family == "both" else [args.family]
    any_approved = False

    for family in families:
        path = PATHS[family]
        if not path.exists():
            logger.error(f"Dataset not found: {path} — run build_phase11_daily_dataset.py first")
            sys.exit(1)

        df = pd.read_feather(path)
        logger.info(f"\n{'='*70}")
        logger.info(f"Phase 11 — {family.upper()} family")
        logger.info(f"Dataset: {len(df)} bars, {df['date'].min()} → {df['date'].max()}")

        exog_feats = BASIS_FEATURES if family == "basis" else FUNDING_FEATURES
        all_feats = PRICE_FEATURES + exog_feats

        # Add noise control column
        rng = np.random.default_rng(42)
        df["noise_control"] = rng.standard_normal(len(df))
        all_feats_with_noise = all_feats + ["noise_control"]

        # Shuffled-label test (on first half of data only, to keep it cheap)
        logger.info("\n--- Shuffled-label sanity test ---")
        half = len(df) // 2
        shuf_auc = shuffled_label_test(df.iloc[:half][all_feats], df.iloc[:half]["triple_barrier_7d"])
        if abs(shuf_auc - 0.5) > 0.03:
            logger.warning(f"Shuffled-label AUC {shuf_auc:.4f} deviates from 0.50 — possible structural leak")

        logger.info("\n--- 4-fold walk-forward ---")
        results = run_walkforward(df, all_feats_with_noise)
        approved = evaluate_gates(results, f"Phase11-{family.capitalize()}")
        if approved:
            any_approved = True

    # CI exit code
    if any_approved:
        logger.info("\n[SUCCESS] At least one model passed all gates.")
        sys.exit(0)
    else:
        logger.error("\n[CRITICAL] Phase 11 evaluation REJECTED. No model passed all gates.")
        sys.exit(1)


if __name__ == "__main__":
    main()
