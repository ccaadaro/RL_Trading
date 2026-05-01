#!/usr/bin/env python3
"""
scripts/build_regime_labels.py

Computes the institutional Kill-Switch layer:
1. Calculates Mahalanobis Multivariate Turbulence.
2. Fits an HMM model and determines semantic market regimes.
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import lightgbm as lgb

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from utils.risk_directors import MahalanobisTurbulence, HMMRegimeModel


def semantic_to_canonical(label: str) -> int:
    if label.endswith("calm"):
        return 0
    if label.endswith("neutral"):
        return 1
    if label in {"high_vol_rebound", "panic_selloff"}:
        return 2
    return -1


def build_causal_hmm_labels(
    df: pd.DataFrame,
    hmm_features: list[str],
    fit_window: int,
    refit_every: int,
    inference_tail: int,
    warmup_bars: int,
    n_components: int,
    n_init: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Label regimes causally: each bar is tagged using an HMM fit only on
    past data, periodically refit on a trailing window.
    """
    n = len(df)
    canonical = np.full(n, -1, dtype=int)
    semantic = np.full(n, "unknown", dtype=object)

    hmm_engine = HMMRegimeModel(
        n_components=n_components,
        n_init=n_init,
        verbose=False,
    )

    hmm_ready = False
    bars_since_fit = 0
    start_i = max(warmup_bars, inference_tail, 2)

    print(
        f"Online HMM labeling: warmup={start_i} fit_window={fit_window} "
        f"refit_every={refit_every} tail={inference_tail}"
    )

    for i in range(start_i, n):
        if (not hmm_ready) or (bars_since_fit >= refit_every):
            fit_start = max(0, i - fit_window)
            fit_df = df.iloc[fit_start:i]
            if len(fit_df) >= max(100, n_components * 20):
                try:
                    hmm_engine.fit(fit_df, hmm_features)
                    hmm_ready = True
                    bars_since_fit = 0
                except Exception:
                    hmm_ready = False

        if hmm_ready:
            tail = df.iloc[max(0, i - inference_tail + 1): i + 1]
            label = hmm_engine.predict_current(tail)
            semantic[i] = label
            canonical[i] = semantic_to_canonical(label)
            bars_since_fit += 1

    return pd.Series(canonical, index=df.index), pd.Series(semantic, index=df.index)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="cache/dollar_bars_btc_2000000_features.feather")
    ap.add_argument("--hmm-fit-window", type=int, default=1000,
                    help="Trailing bars used for each causal HMM refit.")
    ap.add_argument("--hmm-refit-every", type=int, default=500,
                    help="Refit cadence in bars for the causal HMM.")
    ap.add_argument("--hmm-tail", type=int, default=50,
                    help="Recent bars passed to predict_current().")
    ap.add_argument("--hmm-warmup", type=int, default=1000,
                    help="Bars left as unknown before causal HMM labeling starts.")
    ap.add_argument("--hmm-components", type=int, default=3)
    ap.add_argument("--hmm-n-init", type=int, default=3)
    args = ap.parse_args()

    print(f"Loading {args.data}...")
    df = feather.read_feather(args.data)
    
    # Preserve real OOF predictions if they were persisted by train_dollar_alpha.py.
    if "alpha_prob" in df.columns:
        if "oof_valid" not in df.columns:
            df["oof_valid"] = df["alpha_prob"].notna()
        if "oof_source" not in df.columns:
            df["oof_source"] = np.where(df["oof_valid"], "persisted_oof", "unknown")
        print(f"Using persisted alpha_prob column ({int(df['oof_valid'].sum()):,} valid rows).")

    # If no OOF is present, keep compatibility by attaching a clearly-marked
    # in-sample fallback, but never pretend it is valid for statistical gating.
    model_path = Path("models/dollar_alpha_v1/latest_model.txt")
    if "alpha_prob" not in df.columns and model_path.exists():
        print(f"Generating in-sample fallback probabilities from {model_path}...")
        model = lgb.Booster(model_file=str(model_path))
        features = [c for c in df.columns if c.endswith("_feature")]
        preds = model.predict(df[features])
        df["alpha_prob"] = preds
        df["oof_valid"] = False
        df["oof_source"] = "final_model_in_sample_fallback"
        print("Warning: using final-model predictions as fallback only; "
              "validate_pipeline.py will reject them as non-OOF.")
    elif "alpha_prob" not in df.columns:
        print("Warning: Alpha Specialist model not found. alpha_prob left unavailable.")
        df["alpha_prob"] = np.nan
        df["oof_valid"] = False
        df["oof_source"] = "missing"
        
    # We define the multidimensional risk framework for Mahalanobis
    # log_return, volatility_24, aggressor_ratio, intraday_range
    # These must be non-null. The feather from Phase 2 already filtered nans for us!
    risk_vector_cols = [
        "log_return_feature", 
        "volatility_24_feature", 
        "aggressor_ratio", 
        "intraday_range_feature"
    ]
    
    print("\n[Layer 3.1] Calculating Mahalanobis Turbulence Index...")
    t0 = time.time()
    
    # Mahalanobis window of 1000 bars (~3 days in 2M USD BTC volume), 
    # recalculates inverse cov matrix every 250 bars.
    turbulence_engine = MahalanobisTurbulence(window=1000, step=250)
    turbulence_series = turbulence_engine.compute(df, risk_vector_cols)
    
    df["turbulence_score"] = turbulence_series
    
    print(f"Turbulence mapped in {time.time()-t0:.1f}s")
    
    print("\n[Layer 3.2] Building causal Gaussian HMM Regimes...")
    t0 = time.time()

    # D-09 FIX: Using same 4D features as Mahalanobis turbulence for HMM.
    # Adds aggressor_ratio and intraday_range to distinguish microstructure-aware regimes.
    # Falls back to 2D if those columns are not present in the feather.
    preferred_hmm_features = [
        "log_return_feature", "volatility_24_feature",
        "aggressor_ratio", "intraday_range_feature",
    ]
    hmm_features = [f for f in preferred_hmm_features if f in df.columns]
    if len(hmm_features) < 2:
        hmm_features = ["log_return_feature", "volatility_24_feature"]
    print(f"  HMM features ({len(hmm_features)}D): {hmm_features}")


    canonical_regimes, semantic_regimes = build_causal_hmm_labels(
        df=df,
        hmm_features=hmm_features,
        fit_window=args.hmm_fit_window,
        refit_every=args.hmm_refit_every,
        inference_tail=args.hmm_tail,
        warmup_bars=args.hmm_warmup,
        n_components=args.hmm_components,
        n_init=args.hmm_n_init,
    )
    
    df["hmm_canonical_regime"] = canonical_regimes.astype(int)
    df["hmm_semantic_regime"] = semantic_regimes
    df["hmm_label_source"] = "causal_online_hmm"
    
    print(f"Causal HMM fitted/labeled in {time.time()-t0:.1f}s")
    
    print("\nRegime Distribution:")
    print(df["hmm_semantic_regime"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")
    
    out_file = str(Path(args.data).with_name("dollar_bars_btc_2000000_regimes.feather"))
    
    # Save the risk-augmented dataset
    feather.write_feather(df, out_file)
    print(f"\nSaved {len(df):,} rows with Risk/Regime parameters to {out_file}")

if __name__ == "__main__":
    main()
