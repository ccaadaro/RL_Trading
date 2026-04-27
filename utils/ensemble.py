import numpy as np

class RegimeEnsemble:
    """
    Regime-Switching Ensemble to route between a Bull Specialist (e.g. PPO) 
    and a Bear Specialist (e.g. SAC) based on market state.
    """
    def __init__(
        self, 
        bull_model, 
        bear_model,
        t_bull_up: float = 0.03, 
        t_bear_down: float = -0.08, 
        turb_cash: float = 1.5
    ):
        self.bull = bull_model
        self.bear = bear_model
        self.t_bull_up = t_bull_up
        self.t_bear_down = t_bear_down
        self.turb_cash = turb_cash
        self._state = "bull"   # default start state

    def predict(self, obs, *, trend_180: float, turbulence: float, deterministic: bool = True) -> float:
        """
        Produce next action based on current state, gated by hysteresis.
        Returns float action in [-1.0, 1.0]. Returns 0.0 when forcing flat.
        """
        # Risk override 
        if turbulence >= self.turb_cash:
            return 0.0  # force flat
        
        # Hysteresis transitions
        if self._state == "bull" and trend_180 < self.t_bear_down:
            self._state = "bear"
            # Return flat on the transition bar to cleanly switch state 
            # and reset environment tracking (prevent inheriting bad side of a trade)
            return 0.0  
            
        if self._state == "bear" and trend_180 > self.t_bull_up:
            self._state = "bull"
            return 0.0

        model = self.bull if self._state == "bull" else self.bear
        action, _ = model.predict(obs, deterministic=deterministic)
        
        # SB3 predict returns tuple (action, state). 
        # action is typically of shape (1,) or (1, action_dim) depending on env config.
        return float(np.asarray(action).reshape(-1)[0])

    @property
    def current_regime(self) -> str:
        return self._state
