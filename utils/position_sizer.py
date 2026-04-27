"""
utils/position_sizer.py

Implements Institutional Portfolio Allocation logic.
Refactored for Priority 4: Long-Only and Execution Constraints.
"""

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

class FractionalKellySizer:
    """
    Computes position sizes using a Risk-Constrained Fractional Kelly framework.
    """
    def __init__(self, kelly_fraction: float = 0.5, max_drawdown: float = 0.10,
                 long_only: bool = True, fee_rate: float = 5e-4,
                 slippage_bps: float = 2.0, min_risk_scale: float = 1e-3,
                 max_kelly_abs: float = 1.0):
        """
        :param kelly_fraction: Constant multiplier c_kelly (e.g., 0.5 for Half-Kelly).
        :param max_drawdown:   If running drawdown exceeds this fraction, Kelly is halved again.
        :param long_only:      If True, all negative signals are clipped to 0.
        """
        self.c_kelly      = kelly_fraction
        self.max_drawdown = max_drawdown
        self.long_only    = long_only
        self.fee_rate     = fee_rate
        self.slippage_bps = slippage_bps
        self.min_risk_scale = min_risk_scale
        self.max_kelly_abs = max_kelly_abs

        # Institutional Regime Limits (Phase 8 Refinement)
        # We only allow exposure in Bull or Neutral regimes. 
        # Bear and Panic regimes force zero exposure (Long-Only Logic).
        # Institutional Regime Limits (Phase 5: Single-Asset Allocation)
        # Relaxed limits to allow higher conviction on single-asset (BTC) signals.
        self.regime_limits = {
            "bull_calm":        1.00,  # was 0.25
            "bull_neutral":     0.75,  # was 0.15
            "high_vol_rebound": 0.50,  # was 0.10
            "bear_calm":        0.40,  # was 0.10
            "bear_neutral":     0.20,  # was 0.0
            "panic_selloff":    0.00,  # stay zero
            "unknown":          0.20,  # allow small base
        }

    @property
    def round_trip_cost_rate(self) -> float:
        return float(2.0 * (self.fee_rate + self.slippage_bps * 1e-4))

    def estimate_expected_net_return(
        self,
        probabilities: pd.Series,
        risk_scales: pd.Series,
        cost_rate: float | None = None,
    ) -> pd.Series:
        """
        Convert classification probabilities into an approximate expected
        net return per event/bar by scaling edge by a volatility proxy and
        subtracting trading costs.
        """
        probs = probabilities.astype(float).clip(0.0, 1.0)
        scales = risk_scales.astype(float).clip(lower=self.min_risk_scale)
        edge = 2.0 * probs - 1.0
        gross = edge * scales
        costs = self.round_trip_cost_rate if cost_rate is None else float(cost_rate)
        net = np.sign(gross) * np.maximum(np.abs(gross) - costs, 0.0)
        return pd.Series(net, index=probabilities.index)

    def estimate_expected_barrier_return(
        self,
        probabilities: pd.Series,
        upper_returns: pd.Series,
        lower_returns: pd.Series,
        cost_rate: float | None = None,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Estimate gross/net expected return using the event payoff implied by
        the triple-barrier geometry.
        """
        probs = probabilities.astype(float).clip(0.0, 1.0)
        up = upper_returns.astype(float).clip(lower=0.0).fillna(0.0)
        down = lower_returns.astype(float).clip(lower=0.0).fillna(0.0)
        gross = probs * up - (1.0 - probs) * down
        costs = self.round_trip_cost_rate if cost_rate is None else float(cost_rate)
        net = np.sign(gross) * np.maximum(np.abs(gross) - costs, 0.0)
        return pd.Series(gross, index=probabilities.index), pd.Series(net, index=probabilities.index)
        
    def penalty_function(self, turbulence: pd.Series,
                         adaptive_threshold: pd.Series = None,
                         dof: int = 3) -> pd.Series:
        """
        Lagrangian penalty.
        BUG #17 FIX: anchor defaults to chi2 p95 for the active DOF.
        """
        turbulence = turbulence.astype(float).fillna(0.0)
        from scipy.stats import chi2
        p95_anchor = chi2.ppf(0.95, df=dof)

        if adaptive_threshold is not None:
            anchor = adaptive_threshold.reindex(turbulence.index).fillna(p95_anchor)
        else:
            anchor = pd.Series(p95_anchor, index=turbulence.index)

        excess_turb = np.maximum(turbulence.values - anchor.values, 0.0)
        g_penalties = np.exp(-excess_turb / 10.0)
        g_penalties  = np.where(g_penalties < 0.05, 0.0, g_penalties)
        return pd.Series(g_penalties, index=turbulence.index)

    def size_portfolio(self, probabilities: pd.Series | None, regimes: pd.Series,
                       turbulence: pd.Series,
                       adaptive_threshold: pd.Series = None,
                       running_drawdown: float = 0.0,
                       expected_net_returns: pd.Series | None = None,
                       risk_scales: pd.Series | None = None,
                       dof: int = 3) -> pd.Series:
        """
        Calculates theoretical exposure bounds.
        """

        if expected_net_returns is None:
            if probabilities is None:
                raise ValueError("Provide probabilities or expected_net_returns.")
            if risk_scales is None:
                risk_scales = pd.Series(self.min_risk_scale, index=probabilities.index)
            expected_net_returns = self.estimate_expected_net_return(probabilities, risk_scales)
        else:
            expected_net_returns = expected_net_returns.astype(float).fillna(0.0)
            if risk_scales is None:
                risk_scales = pd.Series(
                    np.maximum(np.abs(expected_net_returns.values), self.min_risk_scale),
                    index=expected_net_returns.index,
                )

        scales = risk_scales.astype(float).fillna(self.min_risk_scale).clip(lower=self.min_risk_scale)
        regimes = regimes.fillna("unknown")
        turbulence = turbulence.reindex(regimes.index).fillna(0.0)

        # 1. D-01 FIX: tanh(Sharpe) Kelly — naturally bounded in (-1, 1).
        sharpe_proxy = expected_net_returns.values / np.maximum(scales.values, self.min_risk_scale)
        kelly_base   = np.tanh(sharpe_proxy)

        # Long-Only Constraint: Clip negative Kelly to 0
        if self.long_only:
            kelly_base = np.maximum(kelly_base, 0.0)
            
        kelly_abs_size  = np.abs(kelly_base)
        kelly_direction = np.sign(kelly_base)

        # 2. Regime hard cap
        f_max = regimes.map(self.regime_limits).fillna(0.0)

        # 3. Turbulence penalty
        # BUG #17 FIX: Pass the actual DOF
        g_turb = self.penalty_function(turbulence, adaptive_threshold, dof=dof)

        # 4. Drawdown constraint
        dd_multiplier = 0.5 if running_drawdown > self.max_drawdown else 1.0
        effective_kelly = self.c_kelly * dd_multiplier

        # 5. Integrate
        constrained_abs_size = effective_kelly * kelly_abs_size * g_turb
        final_abs_size       = np.clip(constrained_abs_size, 0, f_max)

        return pd.Series(final_abs_size * kelly_direction, index=regimes.index)
