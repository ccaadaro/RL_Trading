"""
trading_env/risk_wrappers.py
────────────────────────────
Multi-level risk-management Gymnasium wrapper for TradingEnv.

Four independent risk layers, applied in priority order:

  Lvl 4 – Regime  : turbulence index > threshold → force HOLD (risk-off)
  Lvl 2 – Portfolio: drawdown > dd_hard           → force HOLD
  Lvl 3 – Strategy : trailing stop at N×ATR below peak price, then cooldown
  Lvl 1 – Order    : transparent for {0, 1} positions (no leverage)

The wrapper intercepts the action *before* passing it to the underlying env,
so it is compatible with DiscretizeActionWrapper stacked beneath it.

Usage in _build_single_env():
    env = TradingEnv(...)
    env = DiscretizeActionWrapper(env, positions=[0.0, 1.0])
    env = MultiLevelRiskWrapper(env, **risk_cfg)
    return Monitor(env)
"""

import numpy as np
import gymnasium as gym
from typing import Any, Dict, Optional, Tuple


class MultiLevelRiskWrapper(gym.Wrapper):
    """
    Multi-level risk overlay for TradingEnv.

    Parameters
    ----------
    env : gymnasium.Env
        The wrapped environment (may include DiscretizeActionWrapper below).
    dd_hard : float
        Portfolio drawdown threshold for full stop-out (default 15%).
    turbulence_col : str
        Column name in env.df for the turbulence feature.
    turbulence_threshold : float
        Normalised turbulence value (≥1.0 ≈ 99th pct) above which we go flat.
    atr_col : str
        Column name in env.df for the ATR feature (used for trailing stop).
    atr_stop_mult : float
        Trailing stop = atr_stop_mult × ATR below the running price high.
    cooldown_steps : int
        Bars to stay flat after a trailing stop fires.
    apply_turbulence : bool
        If False, the regime-level turbulence check is skipped (useful during
        training so the agent can explore turbulent regimes).
    apply_trailing_stop : bool
        If False, trailing stop-loss is skipped.
    """

    def __init__(
        self,
        env: gym.Env,
        dd_hard: float = 0.15,
        turbulence_col: str = "turbulence_feature",
        turbulence_threshold: float = 1.5,
        atr_col: str = "feature_atr",          # match whatever name train_rl uses
        atr_stop_mult: float = 2.0,
        cooldown_steps: int = 24,
        apply_turbulence: bool = True,
        apply_trailing_stop: bool = True,
    ):
        super().__init__(env)
        self.dd_hard               = dd_hard
        self.turbulence_col        = turbulence_col
        self.turbulence_threshold  = turbulence_threshold
        self.atr_col               = atr_col
        self.atr_stop_mult         = atr_stop_mult
        self.cooldown_steps        = cooldown_steps
        self.apply_turbulence      = apply_turbulence
        self.apply_trailing_stop   = apply_trailing_stop

        # Pre-cache feature columns as numpy arrays (avoids df.iloc per step)
        base = self.unwrapped
        df   = base.df
        self._turb_arr: Optional[np.ndarray] = (
            df[turbulence_col].to_numpy(dtype=np.float32)
            if turbulence_col in df.columns else None
        )
        self._atr_arr: Optional[np.ndarray] = (
            df[atr_col].to_numpy(dtype=np.float32)
            if atr_col in df.columns else None
        )

        # Episode state
        self._peak_nav: float           = 1_000.0
        self._current_nav: float        = 1_000.0
        self._in_long: bool             = False
        self._in_short: bool            = False
        self._price_high: float         = 0.0   # trailing high while in LONG
        self._price_low: float          = float('inf')  # trailing low while in SHORT
        self._cooldown: int             = 0     # steps remaining in cooldown

    # ──────────────────────────────────────────────────────────────────────
    # Repr for audit logging
    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"MultiLevelRiskWrapper("
            f"dd_hard={self.dd_hard}, "
            f"turbulence_threshold={self.turbulence_threshold}, "
            f"atr_stop_mult={self.atr_stop_mult}, "
            f"cooldown_steps={self.cooldown_steps}, "
            f"apply_turbulence={self.apply_turbulence}, "
            f"apply_trailing_stop={self.apply_trailing_stop})"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Gymnasium interface
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        obs, info = self.env.reset(**kwargs)
        initial_nav = float(info.get("portfolio_valuation", 1_000.0))
        self._peak_nav    = initial_nav
        self._current_nav = initial_nav
        self._in_long     = False
        self._in_short    = False
        self._price_high  = self._get_price()
        self._price_low   = self._get_price()
        self._cooldown    = 0
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        # ── intercept & modify action based on risk rules ──────────────────
        safe_action = self._apply_risk_rules(action)

        obs, reward, done, truncated, info = self.env.step(safe_action)

        # ── update episode state ───────────────────────────────────────────
        nav = float(info.get("portfolio_valuation", self._current_nav))
        self._current_nav = nav
        self._peak_nav    = max(self._peak_nav, nav)

        price = self._get_price()
        
        # BUG #2 FIX: Resolve semantic meaning from the env's positions list or actual value,
        # not from hardcoded indices. Supporting both Discrete and Box spaces.
        if hasattr(self.env, "action"): # If it's a wrapper like DiscretizeActionWrapper
             acted_val = self.env.action(safe_action)
        elif isinstance(self.env.action_space, gym.spaces.Box):
             acted_val = float(np.asarray(safe_action).flat[0])
        else:
             # Fallback: assume Discrete index if no mapping exists
             acted = int(np.asarray(safe_action).flat[0])
             # Check if we can infer from positions list
             pos_list = getattr(self.unwrapped, "positions", [])
             acted_val = pos_list[acted] if 0 <= acted < len(pos_list) else acted

        now_long  = (acted_val > 0.05)
        now_short = (acted_val < -0.05)

        if now_long:
            self._price_high = max(self._price_high, price)
            self._price_low  = price
        elif now_short:
            self._price_low  = min(self._price_low, price)
            self._price_high = price
        else:
            # Position closed / was already flat — reset trailing levels
            self._price_high = price
            self._price_low  = price

        self._in_long  = now_long
        self._in_short = now_short

        if self._cooldown > 0:
            self._cooldown -= 1

        # Annotate info so metrics can see risk overrides
        info["risk_action"] = float(np.asarray(safe_action).reshape(-1)[0])
        info["drawdown_pct"] = (self._peak_nav - nav) / max(self._peak_nav, 1e-8)
        if self._cooldown > 0:
            info["in_cooldown"] = True

        return obs, reward, done, truncated, info

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _apply_risk_rules(self, action: Any) -> Any:
        """Return a (possibly overridden) action. Priority: turbulence > DD > trailing stop."""
        if self._peak_nav <= 0:
            return action

        # Level 4 – Regime: turbulence override
        if self.apply_turbulence:
            turb = self._get_turbulence()
            if turb > self.turbulence_threshold:
                return self._flat_action()

        # Level 2 – Portfolio: hard drawdown stop
        dd = (self._peak_nav - self._current_nav) / self._peak_nav
        if dd >= self.dd_hard:
            return self._flat_action()

        # Level 3 – Strategy: cooldown after trailing stop
        if self.apply_trailing_stop:
            if self._cooldown > 0:
                return self._flat_action()

            # Trailing stop check (while in LONG or SHORT)
            if self._in_long:
                atr = self._get_atr()
                if atr > 0:
                    stop_level = self._price_high - self.atr_stop_mult * atr
                    price = self._get_price()
                    if price < stop_level:
                        self._cooldown = self.cooldown_steps
                        return self._flat_action()
            elif self._in_short:
                atr = self._get_atr()
                if atr > 0:
                    stop_level = self._price_low + self.atr_stop_mult * atr
                    price = self._get_price()
                    if price > stop_level:
                        self._cooldown = self.cooldown_steps
                        return self._flat_action()

        return action

    def _flat_action(self):
        """Return the action that means 'flat / HOLD' for the wrapped env.

        BUG #2 FIX: Find the index that corresponds to 0.0 position instead
        of hardcoded index 1.
        """
        space = self.env.action_space
        if isinstance(space, gym.spaces.Box):
            return np.zeros(space.shape, dtype=space.dtype)
        
        # If Discrete, find index closest to 0.0 in positions
        if hasattr(self.env, "_positions"):
            idx = np.argmin(np.abs(self.env._positions))
            return int(idx)
        elif hasattr(self.unwrapped, "positions"):
            idx = np.argmin(np.abs(self.unwrapped.positions))
            return int(idx)
            
        return 1 # Fallback

    def _get_price(self) -> float:
        base = self.unwrapped
        try:
            return float(base._price_array[base._idx])
        except Exception:
            return 1.0

    def _get_turbulence(self) -> float:
        if self._turb_arr is None:
            return 0.0
        idx = self.unwrapped._idx
        return float(self._turb_arr[idx]) if 0 <= idx < len(self._turb_arr) else 0.0

    def _get_atr(self) -> float:
        if self._atr_arr is None:
            return 0.0
        idx = self.unwrapped._idx
        return float(self._atr_arr[idx]) if 0 <= idx < len(self._atr_arr) else 0.0
