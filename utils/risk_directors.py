"""
utils/risk_directors.py

Implements institutional Risk & Regime Directors:
1. M-Turbulence (Mahalanobis Distance): Measures multivariate rarity.
2. HMM Regime Modeling: Classifies latent states logically via volatility ordering.
   Refactored for Priority 3: Separate Fit and Predict cycles for online inference.
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import List, Tuple, Optional
from numpy.linalg import inv, pinv, LinAlgError

logger = logging.getLogger(__name__)


try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    pass # Managed in pipeline

class MahalanobisTurbulence:
    """
    Computes rolling Mahalanobis distance to quantify statistical rarity 
    (turbulence) over a multivariate representation of risk.
    """
    def __init__(self, window: int = 1000, step: int = 250):
        self.window = window
        self.step = step

    def compute(self, df: pd.DataFrame, features: List[str]) -> pd.Series:
        # BUG-04 FIX: Vectorized block-wise computation. O(N/step) inversions instead of O(N).
        mat = df[features].fillna(0).values
        n = len(mat)
        distances = np.full(n, np.nan)

        for block_start in range(self.window, n, self.step):
            sample = mat[block_start - self.window: block_start]
            mean = np.mean(sample, axis=0)
            cov_mat = np.cov(sample, rowvar=False)
            try:
                inv_cov = inv(cov_mat)
            except LinAlgError:
                inv_cov = pinv(cov_mat)

            block_end = min(block_start + self.step, n)
            diffs = mat[block_start:block_end] - mean
            # Vectorized Mahalanobis: d_i = diffs[i] @ inv_cov @ diffs[i]
            distances[block_start:block_end] = np.einsum('ij,jk,ik->i', diffs, inv_cov, diffs)

        return pd.Series(distances, index=df.index)

    def rolling_threshold(self, turbulence: pd.Series, quantile: float = 0.95,
                          window: int = 500) -> pd.Series:
        return turbulence.rolling(window, min_periods=window // 2).quantile(quantile)

class HMMRegimeModel:
    """
    Gaussian HMM for latent regime detection.
    Optimized for online inference (separate fit/predict).
    """
    def __init__(self, n_components: int = 3, n_init: int = 3,
                 random_state: int = 42, verbose: bool = True):
        # BUG-05 FIX: n_init reduced from 10 → 3. Each fit was blocking the GIL 10x.
        # 3 random restarts is enough for convergence in practice.
        self.n_components = n_components
        self.n_init       = n_init
        self.random_state = random_state
        self.model        = None
        self.state_map    = {} # Original state -> Canonical (0:Low, 1:Med, 2:High Vol)
        self.features     = None
        self.verbose      = verbose

    def fit(self, df: pd.DataFrame, features: List[str]):
        """
        Trains the HMM, establishes canonical mapping by volatility.
        """
        self.features = features
        X = df[features].fillna(0).values
        
        best_model = None
        best_score = -np.inf
        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 10_000, size=self.n_init)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*Model is not converging.*")
            
            for seed in seeds:
                candidate = GaussianHMM(
                    n_components=self.n_components,
                    covariance_type="full",
                    n_iter=100,
                    random_state=int(seed),
                )
                try:
                    candidate.fit(X)
                    score = candidate.score(X)
                    if score > best_score:
                        best_score = score
                        best_model = candidate
                except Exception:
                    continue

        if best_model is None:
            raise RuntimeError("HMM failed to converge.")

        self.model = best_model
        
        # Determine canonical mapping: Sort states by trace of covariance (volatility proxy)
        state_vols = []
        for i in range(self.n_components):
            vol = np.trace(self.model.covars_[i])
            state_vols.append((i, vol))
            
        state_vols.sort(key=lambda x: x[1])
        self.state_map = {original: canonical for canonical, (original, _) in enumerate(state_vols)}
        
        if self.verbose:
            print(f"[HMM] Model fitted. Score: {best_score:.2f} | States Vol Map: {self.state_map}")
        return self

    def predict_current_state(self, df_tail: pd.DataFrame) -> int:
        """
        Infers the canonical state ID for the latest observation.
        :param df_tail: Last N bars.
        """
        if self.model is None:
            return -1
            
        X = df_tail[self.features].fillna(0).values
        try:
            hidden_states = self.model.predict(X)
            last_orig_state = hidden_states[-1]
            return self.state_map.get(last_orig_state, -1)
        except Exception:
            return -1

    def predict_current(self, df_tail: pd.DataFrame) -> str:
        """
        Infers the regime for the latest observation using online sequence inference.
        :param df_tail: Last N bars (e.g. 50) including the current one.
        """
        if self.model is None:
            return "unknown"
            
        X = df_tail[self.features].fillna(0).values
        try:
            # We predict the sequence and take the last state
            hidden_states = self.model.predict(X)
            last_orig_state = hidden_states[-1]
            canonical_state = self.state_map.get(last_orig_state, -1)
            
            # Semantic labeling (Log Return is features[0])
            assert self.features[0] == "log_return_feature", "HMM first feature must be log_return_feature for semantic labeling"
            returns = df_tail[self.features[0]].values

            avg_ret = np.mean(returns[-24:]) if len(returns) >= 24 else np.mean(returns)
            
            if canonical_state == 0:
                return "bull_calm" if avg_ret >= 0 else "bear_calm"
            elif canonical_state == 1:
                return "bull_neutral" if avg_ret >= 0 else "bear_neutral"
            elif canonical_state == 2:
                return "high_vol_rebound" if avg_ret >= 0 else "panic_selloff"
            return "unknown"
        except Exception as e:
            # BUG-02 FIX: Never use bare except. Log the real error so bugs are visible.
            logger.warning("[HMM] predict_current failed: %s", e, exc_info=False)
            return "unknown"

    def fit_predict(self, df: pd.DataFrame, features: List[str]) -> Tuple[pd.Series, pd.Series]:
        """
        Legacy wrapper for bulk backtesting if needed.
        """
        self.fit(df, features)
        X = df[features].fillna(0).values
        hidden_states = self.model.predict(X)
        
        canonical_states = np.vectorize(self.state_map.get)(hidden_states)
        
        returns = df[features[0]].values
        roll_mean_ret = pd.Series(returns).rolling(24).mean().fillna(0).values
        
        semantic_labels = []
        for i in range(len(canonical_states)):
            c = canonical_states[i]
            r = roll_mean_ret[i]
            if c == 0: s = "bull_calm" if r >= 0 else "bear_calm"
            elif c == 1: s = "bull_neutral" if r >= 0 else "bear_neutral"
            elif c == 2: s = "high_vol_rebound" if r >= 0 else "panic_selloff"
            else: s = "unknown"
            semantic_labels.append(s)
            
        return pd.Series(canonical_states, index=df.index), pd.Series(semantic_labels, index=df.index)
