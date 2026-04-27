"""
Regime-balanced episode sampling for training environments.

Problem: Random episode sampling from chronological training data over-represents
the dominant market regime (typically bullish for BTC). The agent never learns
to value or trade bear/sideways markets.

Solution: Classify each bar into a regime (bear/sideways/bull) using a trend
feature, then sample episode start points uniformly across regimes so the agent
sees equal exposure to all market conditions.
"""

import gymnasium as gym
import numpy as np


class RegimeBalancedWrapper(gym.Wrapper):
    """Override episode start index to balance across market regimes.

    Wraps a TradingEnv and intercepts reset() to pick start indices
    that ensure roughly equal exposure to bull, bear, and sideways regimes.

    Parameters
    ----------
    env : gym.Env
        A TradingEnv (possibly already wrapped by DiscretizeActionWrapper etc.)
    regime_col : str
        Column name in env.unwrapped.df used to classify regimes.
        Should be a trend/momentum feature (e.g. 180-bar rolling return).
    n_bins : int
        Number of regime buckets (default 3: bear / sideways / bull).
    """

    def __init__(self, env: gym.Env, regime_col: str = "trend_return_180_feature",
                 n_bins: int = 3):
        super().__init__(env)
        df = env.unwrapped.df
        trend = df[regime_col].values

        # Tertile thresholds (bear < p33, sideways p33-p67, bull > p67)
        valid = trend[~np.isnan(trend)]
        percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # e.g. [33.3, 66.7]
        thresholds = np.percentile(valid, percentiles)
        labels = np.digitize(trend, thresholds)  # 0=bear, 1=sideways, 2=bull

        # Valid episode start indices
        max_ep = env.unwrapped.max_episode_duration
        if isinstance(max_ep, int):
            valid_starts = np.arange(0, max(1, len(df) - max_ep - 1))
        else:
            valid_starts = np.array([0])

        # Group valid starts by regime of start bar
        self._regime_starts = {}
        for regime_id in range(n_bins):
            mask = labels[valid_starts] == regime_id
            self._regime_starts[regime_id] = valid_starts[mask]

        self._n_regimes = n_bins
        counts = {k: len(v) for k, v in self._regime_starts.items()}
        regime_names = {0: "bear", 1: "sideways", 2: "bull"}
        _desc = "  ".join(f"{regime_names.get(k, k)}={v}" for k, v in counts.items())
        print(f"  [RegimeSampler] {_desc} (total valid starts: {len(valid_starts)})")

    def reset(self, **kwargs):
        # BUG #1 FIX: Instead of manually setting _idx (which gets overridden by super().reset),
        # we pass it via the 'options' dict which TradingEnv now supports.
        regime = np.random.randint(0, self._n_regimes)
        starts = self._regime_starts[regime]
        
        if len(starts) > 0:
            idx = int(np.random.choice(starts))
            options = kwargs.get("options", {}) or {}
            options["force_start_idx"] = idx
            kwargs["options"] = options
            
        return super().reset(**kwargs)
