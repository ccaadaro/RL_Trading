"""
utils/hedge_manager.py

Implements institutional Delta-Neutral hedging.
Applies asymmetric short exposure via Perpetual Contracts specifically to cover 
unresolved Spot Execution Lag during high-turbulence systemic events.
"""

import numpy as np

class DynamicHedger:
    def __init__(self, critical_turbulence: float = 10.0, high_stress_turbulence: float = 5.0):
        # Mahalanobis critical thresholds
        self.crit_turb = critical_turbulence
        self.stress_turb = high_stress_turbulence
        
    def determine_lambda_panic(self, regime: str, turbulence: float) -> float:
        """
        Determines the percentage coefficient (lambda) of lagging spot inventory 
        that is allowed to be covered.
        """
        if regime == 'panic_selloff' or turbulence > self.crit_turb:
            return 1.00 # Max Panic: 100% of lag risk can be covered
            
        if turbulence > self.stress_turb:
            return 0.65 # Normal stress: restrict overhedging, cover 65%
            
        return 0.0 # Calm conditions do not warrant expensive market-taker hedges
        
    def calculate_hedge(self, 
                        spot_inventory: float, 
                        target_inventory: float, 
                        regime: str, 
                        turbulence: float) -> float:
        """
        Computes the emergency hedge size.
        Hedge ensures Delta-Neutrality on the residual lag that the VWAP router
        couldn't clear quickly enough.
        
        :param spot_inventory: Fractional portfolio size currently held (e.g., 0.25)
        :param target_inventory: Fractional portfolio size intended (e.g., 0.00)
        :returns: Hedge position size (e.g., -0.25) to cast via Market Order protocols.
        """
        # If we have no inventory, there is nothing to defend
        if spot_inventory == 0.0:
            return 0.0
            
        lambda_panic = self.determine_lambda_panic(regime, turbulence)
        
        if lambda_panic <= 0.0:
            return 0.0
        
        # We only hedge when we are trying to reduce exposure (exiting)
        # and there's a logistical lag preventing the speedy exit.
        
        # Calculate lag (amount of inventory yet to be cleared)
        # Ex: If Spot is +0.25 (Long) and Target is 0.0, lag is 0.25.
        if spot_inventory > 0:
            # We are Long. We only hedge if Target is lower than Spot.
            if target_inventory < spot_inventory:
                execution_lag = spot_inventory - target_inventory
                
                # Formula: min(execution_lag, lambda * spot_inventory)
                hedge_mag = min(execution_lag, lambda_panic * spot_inventory)
                return -hedge_mag # Short hedge

        elif spot_inventory < 0:
            # We are Short. We only hedge if Target is higher than Spot (closing shorts).
            if target_inventory > spot_inventory:
                execution_lag = target_inventory - spot_inventory
                
                # Formula: min(execution_lag, lambda * |spot_inventory|)
                hedge_mag = min(execution_lag, lambda_panic * abs(spot_inventory))
                return hedge_mag # Long hedge
                
        return 0.0
