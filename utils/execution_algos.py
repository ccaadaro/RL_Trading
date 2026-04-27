"""
utils/execution_algos.py

Implements deterministic Point-of-Volume (POV) and Volume-Clock scheduling router
for institutional execution over Dollar Bars.
"""

import numpy as np

class DeterministicPOVRouter:
    """
    Schedules execution slices across Dollar Bars using a Participation Cap bounded framework.
    Formula: q_b = min( Q_rem / N_rem, rho_max * V_bar, q_LOB_safe )
    """
    def __init__(self, V_bar: float = 2_000_000.0):
        self.V_bar = V_bar
        
        # Base settings
        self.default_horizon_bars = 4
        
    def determine_rho_max(self, regime: str, urgency: float) -> float:
        """
        Determines the participation cap based on Regime and absolute Alpha urgency.
        """
        if regime in ['panic_selloff', 'bear_calm'] and urgency < 0:
            return 0.05 # standard
        
        if urgency > 0.15: # High urgency / edge
            if regime in ['bull_calm', 'bull_neutral', 'high_vol_rebound']:
                return 0.06 # urgent (max 8%)
                
        # the default for calm/passive is 2% participation
        return 0.02
        
    def slice_order(self, 
                    Q_remaining: float, 
                    N_remaining: int, 
                    regime: str, 
                    urgency: float,
                    q_lob_safe: float | None = None) -> float:
        """
        Calculates q_b: capital to execute in the current bar.

        :param q_lob_safe: Notional available from the L2 book.
                           Pass None (or omit) only when no book is available —
                           falls back to 5% of V_bar as conservative floor.
                           Do NOT pass float('inf') — that disables the limit.
        """
        if Q_remaining <= 0.0 or N_remaining <= 0:
            return 0.0

        # Fallback when no depth data is available: 5% of one Dollar Bar (conservative)
        if q_lob_safe is None:
            q_lob_safe = self.V_bar * 0.05

        rho_max = self.determine_rho_max(regime, urgency)

        uniform_slice     = Q_remaining / N_remaining
        participation_limit = rho_max * self.V_bar

        target_fill = min(uniform_slice, participation_limit, q_lob_safe)


        # Ensure we do not fill more than what's needed
        q_b = min(target_fill, Q_remaining)
        
        return q_b
