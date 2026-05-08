#!/usr/bin/env python3
"""
Phase 10 — Model 3 Evaluator.

Supports --family {funding, basis} for isolated single-family evaluation.
Do NOT combine families until each passes standalone gates.

Pipeline:
  1. Single-feature AUC scan  (block if any > 0.85)
  2. Correlation with target   (warn if |corr| > 0.5)
  3. Shuffled-label test       (block if AUC > 0.55)
  4. Lag-all-features test     (AUC should degrade)
  5. 4-fold walk-forward with Random P95 / B&H / EMA benchmarks
  6. Feature importance stability

Pre-committed gates (MODEL3_PROTOCOLS.md):
  PASS if: mean_AUC >= 0.53, min_fold_AUC >= 0.51,
           Net_ROI > Random_P95, Calmar > BH_Calmar

Usage:
  python scripts/evaluate_model3.py --family funding
  python scripts/evaluate_model3.py --family basis

Exit 0 = PASS, Exit 1 = FAIL/BLOCKED, Exit 2 = LEAKAGE
"""
from __future__ import annotations
import logging, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"

# ── Feature / dataset registry — add new families here ────────────────────
FAMILY_REGISTRY: dict[str, dict] = {
    "funding": {
        "dataset": CACHE / "btc_1h_phase10_funding.feather",
        "features": [
            "funding_last",
            "funding_8h_zscore_30d",
            "funding_8h_zscore_90d",
            "funding_abs_zscore_30d",
            "funding_sign",
        ],
    },
    "basis": {
        "dataset": CACHE / "btc_1h_phase10_basis.feather",
        "features": [
            "basis_now",
            "basis_zscore_30d",
            "basis_mean_24h",
            "basis_change_24h",
            "basis_compression",
        ],
    },
}

LABEL_COLS = ["triple_barrier_48h", "trend_48h", "trend_72h"]
TARGET = "triple_barrier_48h"

# ── Model config (pre-committed — do NOT tune after seeing results) ─────────
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 3,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
}
N_ESTIMATORS = 300
PURGE = 72       # bars (3 days)
EMBARGO = 72     # bars

# ── Gates (pre-committed) ──────────────────────────────────────────────────
GATE_MEAN_AUC    = 0.53
GATE_MIN_FOLD    = 0.51
LEAK_AUC_WARN    = 0.65
LEAK_AUC_BLOCK   = 0.85
COST_BPS         = 7.0   # per side
HOLD_BARS        = 48


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def to_binary(y: pd.Series) -> pd.Series:
    """Convert {-1, 1} → {0, 1}."""
    return ((y + 1) / 2).astype(int)


def compute_pnl(close: pd.Series, positions: np.ndarray, cost_bps=COST_BPS):
    returns = close.pct_change().fillna(0).values
    pos = np.zeros_like(positions, dtype=float)
    pos[1:] = positions[:-1]           # 1-bar execution lag
    strat_ret = returns * pos
    cost = np.abs(np.diff(pos, prepend=pos[0])) * cost_bps / 10_000
    net = strat_ret - cost
    cum = np.cumprod(1 + net)
    total_ret = float(cum[-1] - 1) if len(cum) else 0.0
    dd = cum / np.maximum.accumulate(cum) - 1
    max_dd = float(dd.min()) if len(dd) else 0.0
    sharpe = float(net.mean() / net.std() * np.sqrt(252 * 24)) if net.std() > 0 else 0.0
    calmar = total_ret / abs(max_dd) if max_dd != 0 else 0.0
    n_trades = int(np.sum(np.diff(pos, prepend=pos[0]) != 0))
    return dict(ret=total_ret, max_dd=max_dd, sharpe=sharpe, calmar=calmar, n_trades=n_trades)


def train_lgb(X_tr, y_tr, X_te):
    valid = y_tr.notna()
    y_bin = to_binary(y_tr[valid])
    ds = lgb.Dataset(X_tr[valid], label=y_bin)
    model = lgb.train(LGB_PARAMS, ds, num_boost_round=N_ESTIMATORS)
    return model.predict(X_te), model


def random_p_benchmark(close: pd.Series, time_in_market: float,
                       n_sim=500, rng_seed=42) -> dict:
    """Random strategy matched on time-in-market."""
    rng = np.random.default_rng(rng_seed)
    rets = close.pct_change().fillna(0).values
    n = len(rets)
    hold = max(1, int(round(time_in_market * n)))
    sim_rets = []
    for _ in range(n_sim):
        pos = np.zeros(n)
        starts = rng.choice(n - HOLD_BARS, size=max(1, hold // HOLD_BARS), replace=False)
        for s in starts:
            pos[s:s + HOLD_BARS] = 1
        pos_lag = np.zeros_like(pos)
        pos_lag[1:] = pos[:-1]
        r = (rets * pos_lag).sum()
        sim_rets.append(r)
    arr = np.array(sim_rets)
    return dict(p50=float(np.percentile(arr, 50)), p95=float(np.percentile(arr, 95)))


# ─────────────────────────────────────────────────────────────────────────
# Sanity Tests
# ─────────────────────────────────────────────────────────────────────────

def sanity_single_feature_auc(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Single-feature AUC scan — any > LEAK_AUC_BLOCK triggers exit."""
    logger.info("── Sanity 1: Single-Feature AUC Scan ──")
    mask = df[TARGET].notna()
    y = to_binary(df.loc[mask, TARGET])
    rows = []
    blocked = False
    for feat in features:
        x = df.loc[mask, feat].fillna(0).values
        try:
            auc = roc_auc_score(y, x)
            auc = max(auc, 1 - auc)   # flip if below 0.5
        except Exception:
            auc = np.nan
        flag = ""
        if auc > LEAK_AUC_BLOCK:
            flag = "⚠ LEAKAGE — BLOCKED"
            blocked = True
        elif auc > LEAK_AUC_WARN:
            flag = "⚠ SUSPICIOUS"
        rows.append(dict(feature=feat, auc=round(auc, 4), note=flag))
        logger.info(f"  {feat:35s}  AUC={auc:.4f}  {flag}")
    if blocked:
        logger.error("BLOCKED BY LEAKAGE — single-feature AUC > 0.85. Fix before training.")
        sys.exit(2)
    return pd.DataFrame(rows)


def sanity_correlation(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Absolute correlation of each feature with target."""
    logger.info("── Sanity 2: Correlation with Target ──")
    mask = df[TARGET].notna()
    sub = df[mask].copy()
    rows = []
    for feat in features:
        c = sub[feat].fillna(0).corr(sub[TARGET].fillna(0))
        flag = "⚠ HIGH — INSPECT" if abs(c) > 0.5 else ""
        rows.append(dict(feature=feat, corr_with_target=round(c, 4), note=flag))
        logger.info(f"  {feat:35s}  corr={c:.4f}  {flag}")
    return pd.DataFrame(rows)


def sanity_shuffled_label(df_tr: pd.DataFrame, df_te: pd.DataFrame,
                          features: list[str], seed=42) -> float:
    """Shuffled-label test: AUC must collapse to ~0.50."""
    logger.info("── Sanity 3: Shuffled-Label Test ──")
    y_tr = df_tr[TARGET].copy()
    rng = np.random.default_rng(seed)
    y_shuffled = pd.Series(rng.permutation(y_tr.values), index=y_tr.index)
    X_tr = df_tr[features].fillna(0)
    X_te = df_te[features].fillna(0)
    mask_te = df_te[TARGET].notna()
    preds, _ = train_lgb(X_tr, y_shuffled, X_te)
    y_te = to_binary(df_te.loc[mask_te, TARGET])
    auc = roc_auc_score(y_te, preds[mask_te.values]) if len(y_te.unique()) > 1 else np.nan
    flag = "⚠ LEAKAGE SUSPECTED" if auc > 0.52 else "✅ OK"
    logger.info(f"  Shuffled-label AUC = {auc:.4f}  {flag}")
    if auc > 0.55:
        logger.error("BLOCKED — shuffled label AUC materially > 0.50. Structural leakage.")
        sys.exit(2)
    return auc


def sanity_lag_all_features(df_tr: pd.DataFrame, df_te: pd.DataFrame,
                             features: list[str]) -> float:
    """Shift all features +1 bar. AUC must degrade."""
    logger.info("── Sanity 4: Lag-All-Features Test ──")
    X_tr_lag = df_tr[features].shift(1).fillna(0)
    X_te_lag = df_te[features].shift(1).fillna(0)
    mask_te = df_te[TARGET].notna()
    preds, _ = train_lgb(X_tr_lag, df_tr[TARGET], X_te_lag)
    y_te = to_binary(df_te.loc[mask_te, TARGET])
    auc_lag = roc_auc_score(y_te, preds[mask_te.values]) if len(y_te.unique()) > 1 else np.nan
    logger.info(f"  Lag-all-features AUC = {auc_lag:.4f}  (should be <= normal AUC)")
    return auc_lag


# ─────────────────────────────────────────────────────────────────────────
# Walk-Forward Evaluation
# ─────────────────────────────────────────────────────────────────────────

def run_walkforward(df: pd.DataFrame, features: list[str]) -> dict:
    """4-fold strict chronological walk-forward."""
    logger.info("── 4-Fold Walk-Forward ──")

    # Add noise control feature
    rng = np.random.default_rng(0)
    df = df.copy()
    df["noise_control"] = rng.standard_normal(len(df))
    feat_with_noise = features + ["noise_control"]

    df_clean = df.dropna(subset=[TARGET]).reset_index(drop=True)
    n = len(df_clean)
    fold_size = n // 5   # 5 segments → 4 test folds with valid training

    fold_results, importances = [], {}

    for fold in range(4):
        test_start = (fold + 1) * fold_size
        test_end   = (fold + 2) * fold_size if fold < 3 else n
        train_end  = max(0, test_start - PURGE - EMBARGO)

        df_tr = df_clean.iloc[:train_end]
        df_te = df_clean.iloc[test_start:test_end].reset_index(drop=True)

        if len(df_tr) < 500 or len(df_te) < 100:
            logger.warning(f"Fold {fold+1}: insufficient data (tr={len(df_tr)}, te={len(df_te)}), skipping")
            continue

        logger.info(f"\nFold {fold+1}/4: train={len(df_tr)}, test={len(df_te)}, "
                    f"purge+embargo={PURGE+EMBARGO}")

        X_tr = df_tr[feat_with_noise].fillna(0)
        y_tr = df_tr[TARGET]
        X_te = df_te[feat_with_noise].fillna(0)

        # Sanity checks on fold 1 only (representative)
        if fold == 0:
            sanity_shuffled_label(df_tr, df_te, feat_with_noise)
            auc_lag = sanity_lag_all_features(df_tr, df_te, feat_with_noise)

        preds, model = train_lgb(X_tr, y_tr, X_te)

        # AUC
        mask_te = df_te[TARGET].notna()
        y_te_bin = to_binary(df_te.loc[mask_te, TARGET])
        auc = roc_auc_score(y_te_bin, preds[mask_te.values]) if len(y_te_bin.unique()) > 1 else np.nan

        # Feature importance
        fi = dict(zip(feat_with_noise, model.feature_importance("gain")))
        for k, v in fi.items():
            importances.setdefault(k, []).append(v)

        # Noise control check
        noise_rank = sorted(fi.items(), key=lambda x: -x[1])
        noise_pos = next((i for i, (k, _) in enumerate(noise_rank) if k == "noise_control"), None)
        if noise_pos is not None and noise_pos < 2:
            logger.warning(f"  ⚠ noise_control ranked #{noise_pos+1} — model may be unstable")

        # Positions (48h hold, confidence threshold 0.45/0.55)
        vol_thresh = float(df_tr["realized_vol_24"].quantile(0.25)) if "realized_vol_24" in df_tr.columns else 0.0
        positions = np.zeros(len(df_te))
        hold_timer = 0
        cur_pos = 0
        for i in range(len(df_te)):
            if hold_timer > 0:
                positions[i] = cur_pos
                hold_timer -= 1
            else:
                low_vol = ("realized_vol_24" in df_te.columns and
                           df_te["realized_vol_24"].iloc[i] < vol_thresh)
                if low_vol:
                    cur_pos = 0
                elif preds[i] > 0.55:
                    cur_pos = 1
                elif preds[i] < 0.45:
                    cur_pos = -1
                else:
                    cur_pos = 0
                positions[i] = cur_pos
                hold_timer = HOLD_BARS - 1

        pnl = compute_pnl(df_te["close"], positions)

        # Benchmarks
        bh  = compute_pnl(df_te["close"], np.ones(len(df_te)))
        flat = dict(ret=0.0, max_dd=0.0, sharpe=0.0, calmar=0.0, n_trades=0)
        ema12 = df_te["close"].ewm(span=12).mean()
        ema26 = df_te["close"].ewm(span=26).mean()
        ema_sig = (ema12 > ema26).astype(float).values
        ema_pnl = compute_pnl(df_te["close"], ema_sig)

        tim = float(np.mean(positions != 0))
        rnd = random_p_benchmark(df_te["close"], tim)

        logger.info(f"  AUC={auc:.4f} | net_ret={pnl['ret']:.4f} | calmar={pnl['calmar']:.4f} | "
                    f"BH_ret={bh['ret']:.4f} | rand_p95={rnd['p95']:.4f}")

        fold_results.append(dict(
            fold=fold + 1,
            auc=auc, auc_lag=auc_lag if fold == 0 else np.nan,
            model_ret=pnl["ret"], model_calmar=pnl["calmar"],
            model_sharpe=pnl["sharpe"], model_dd=pnl["max_dd"],
            model_trades=pnl["n_trades"],
            bh_ret=bh["ret"], bh_calmar=bh["calmar"],
            ema_ret=ema_pnl["ret"], ema_calmar=ema_pnl["calmar"],
            rand_p50=rnd["p50"], rand_p95=rnd["p95"],
            time_in_market=tim,
        ))

    return dict(fold_results=fold_results, importances=importances)


# ─────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────

def print_report(results: dict, single_auc_df: pd.DataFrame, corr_df: pd.DataFrame,
                 family: str = "funding") -> bool:
    fold_results = results["fold_results"]
    importances  = results["importances"]

    if not fold_results:
        logger.error("No fold results — BLOCKED")
        return False

    fr = pd.DataFrame(fold_results)
    sep = "=" * 70

    logger.info(f"\n{sep}")
    logger.info(f"PHASE 10 — MODEL 3 ({family.upper()}-ONLY) EVALUATION REPORT")
    logger.info(sep)

    # Per-fold table
    logger.info("\n── Per-Fold Results ──")
    cols = ["fold","auc","model_ret","model_calmar","bh_ret","bh_calmar","rand_p95","time_in_market"]
    logger.info("\n" + fr[cols].to_string(index=False, float_format="{:.4f}".format))

    # Aggregate
    mean_auc     = fr["auc"].mean()
    std_auc      = fr["auc"].std()
    min_auc      = fr["auc"].min()
    mean_ret     = fr["model_ret"].mean()
    mean_calmar  = fr["model_calmar"].mean()
    mean_bh_cal  = fr["bh_calmar"].mean()
    mean_p95     = fr["rand_p95"].mean()

    logger.info(f"\n── Aggregate ──")
    logger.info(f"  mean AUC     = {mean_auc:.4f} ± {std_auc:.4f}  (min fold = {min_auc:.4f})")
    logger.info(f"  mean net ROI = {mean_ret:.4f}")
    logger.info(f"  mean Calmar  = {mean_calmar:.4f}  (B&H Calmar = {mean_bh_cal:.4f})")
    logger.info(f"  Random P95   = {mean_p95:.4f}")

    # Feature importance stability
    logger.info("\n── Feature Importance Stability ──")
    fi_rows = []
    for feat, vals in importances.items():
        mu, sd = np.mean(vals), np.std(vals)
        cv = sd / mu if mu > 0 else np.nan
        fi_rows.append(dict(feature=feat, mean_gain=round(mu,1), std_gain=round(sd,1), cv=round(cv,3)))
    fi_df = pd.DataFrame(fi_rows).sort_values("mean_gain", ascending=False)
    logger.info("\n" + fi_df.to_string(index=False))

    # Sanity: noise control should NOT be top feature
    top_feat = fi_df.iloc[0]["feature"] if len(fi_df) else ""
    if top_feat == "noise_control":
        logger.warning("⚠ noise_control is the top feature — model is unstable")

    # Single-feature AUC summary
    logger.info("\n── Single-Feature AUC ──")
    logger.info("\n" + single_auc_df.to_string(index=False))

    logger.info("\n── Correlation with Target ──")
    logger.info("\n" + corr_df.to_string(index=False))

    # Gate decision
    logger.info(f"\n{sep}")
    logger.info("GATE EVALUATION (pre-committed thresholds)")
    logger.info(sep)
    gate_auc    = mean_auc >= GATE_MEAN_AUC
    gate_minfld = min_auc >= GATE_MIN_FOLD
    gate_roi    = mean_ret > mean_p95
    gate_calmar = mean_calmar > mean_bh_cal

    logger.info(f"  mean AUC >= {GATE_MEAN_AUC}:          {'✅ PASS' if gate_auc    else '❌ FAIL'}  ({mean_auc:.4f})")
    logger.info(f"  min fold AUC >= {GATE_MIN_FOLD}:       {'✅ PASS' if gate_minfld else '❌ FAIL'}  ({min_auc:.4f})")
    logger.info(f"  Net ROI > Random P95:        {'✅ PASS' if gate_roi    else '❌ FAIL'}  ({mean_ret:.4f} vs {mean_p95:.4f})")
    logger.info(f"  Calmar > B&H Calmar:         {'✅ PASS' if gate_calmar else '❌ FAIL'}  ({mean_calmar:.4f} vs {mean_bh_cal:.4f})")

    passed = gate_auc and gate_minfld and gate_roi and gate_calmar

    logger.info(f"\n{'='*70}")
    if passed:
        next_step = {
            "funding": "Proceed to Family 2 (basis-only) with explicit sign-off.",
            "basis":   "Basis shows edge. Proceed to funding+basis combined or Phase 11.",
        }.get(family, "Proceed per MODEL3_PROTOCOLS.md.")
        logger.info(f"DECISION: PASS — {family} features show edge. {next_step}")
    else:
        next_step = {
            "funding": "Archive funding-only. Run basis-only (explicit sign-off granted).",
            "basis":   "Archive Phase 10 entirely. Proceed to Phase 11 (Horizon Shift).",
        }.get(family, "Archive. See MODEL3_PROTOCOLS.md for next step.")
        logger.info(f"DECISION: FAIL — {family}-only model does not meet gates.")
        logger.info(f"  → {next_step}")
    logger.info(f"{'='*70}\n")

    return passed


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Phase 10 Model 3 Evaluator")
    parser.add_argument(
        "--family",
        default="funding",
        choices=list(FAMILY_REGISTRY.keys()),
        help="Feature family to evaluate in isolation (default: funding)",
    )
    args = parser.parse_args()
    family = args.family

    cfg = FAMILY_REGISTRY[family]
    data_path = cfg["dataset"]
    features  = cfg["features"]

    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        logger.error(
            f"Run: conda run -n freqtrade python3 scripts/build_model3_exogenous_dataset.py "
            f"--family {family}"
        )
        sys.exit(1)

    logger.info(f"Family: {family}")
    logger.info(f"Loading {data_path}")
    df = pd.read_feather(data_path)
    logger.info(f"Shape: {df.shape} | {df['date'].min()} → {df['date'].max()}")
    logger.info(f"Features: {features}")

    # Safety: no label columns in features
    leak = [c for c in features if c in LABEL_COLS]
    assert not leak, f"CRITICAL: label columns in features: {leak}"

    # Verify all features exist in dataset
    missing = [c for c in features if c not in df.columns]
    if missing:
        logger.error(f"Features missing from dataset: {missing}")
        sys.exit(1)

    # Sanity 1 & 2 on full dataset
    single_auc_df = sanity_single_feature_auc(df, features)
    corr_df = sanity_correlation(df, features)

    # Walk-forward (includes shuffled-label + lag-all on fold 1)
    results = run_walkforward(df, features)

    # Report + gate
    passed = print_report(results, single_auc_df, corr_df, family=family)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
