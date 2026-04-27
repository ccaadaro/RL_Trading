"""
utils/filters.py
────────────────
Advanced sampling filters for event-driven trading.
Based on Marcos López de Prado's "Advances in Financial Machine Learning".
"""

import numpy as np

class SymmetricCUSUMFilter:
    """
    Symmetric CUSUM Filter to detect informative events in a time series.
    It identifies when the cumulative deviations from a level exceed a 
    predefined threshold 'h'.
    """

    def __init__(self):
        self._pos_sum = 0.0
        self._neg_sum = 0.0

    def check(self, x_diff: float, threshold: float) -> bool:
        """
        Updates the CUSUM sums and checks for a crossing.
        
        Args:
            x_diff: The current deviation (e.g., log-return).
            threshold: The boundary 'h' to detect.
            
        Returns:
            True if an event is detected, False otherwise.
        """
        # Update cumulative sums
        self._pos_sum = max(0.0, self._pos_sum + x_diff)
        self._neg_sum = min(0.0, self._neg_sum + x_diff)

        # Check for threshold crossing
        if self._pos_sum > threshold or self._neg_sum < -threshold:
            self.reset()
            return True

        return False

    def reset(self):
        """Resets the internal cumulative sums."""
        self._pos_sum = 0.0
        self._neg_sum = 0.0
