"""
trading_env/cvar_wrapper.py
───────────────────────────
CVaR-constrained Gymnasium wrapper + SB3 callback.

Implements the Constrained MDP (CMDP) primal-dual (Lagrangian) approach:

  Policy objective (CMDP):
      max  E[cumulative reward]
      s.t. CVaR_α(episode return) ≥ cvar_budget

Translated to an unconstrained problem via Lagrangian relaxation:
      max  E[reward] − λ × max(0, cvar_budget − CVaR_α)

Components
──────────
  CVaRConstraintWrapper  — Gymnasium wrapper applied around TradingEnv.
    • Tracks cumulative reward per episode.
    • At episode end: if avg-step reward ∈ α-tail, subtracts a penalty
      λ × max(0, VaR_α − avg_step_reward) from the terminal reward.
    • λ and VaR_α are set externally by CVaRCallback (set_cvar_params).

  CVaRCallback  — SB3 BaseCallback run during model.learn().
    • Collects per-episode avg-step rewards from Monitor's info["episode"].
    • Every *update_freq* episodes: recomputes VaR_α and CVaR_α;
      gradient-ascends λ: λ ← clip(λ + λ_lr × violation, 0, λ_max)
    • Broadcasts (λ, VaR_α) to all envs via SubprocVecEnv.env_method().
    • Logs to TensorBoard: cvar/var_alpha, cvar/cvar_alpha, cvar/lambda,
      cvar/budget_violation.

VecEnv compatibility
────────────────────
  With SubprocVecEnv: env_method('set_cvar_params', ...) sends the call
  via pickle to each worker process; gym.Wrapper.__getattr__ propagates
  the call through Monitor → CVaRConstraintWrapper.set_cvar_params.

References
──────────
  Altman (1999)              "Constrained Markov Decision Processes"
  Tamar et al. (2015)        "Policy Gradient for Coherent Risk Measures"
  Coache et al. (2022)       "Reinforcement Learning with Constraints"

Calibration notes
─────────────────
  cvar_budget is in units of avg-step reward.  With DSR reward, a badly
  losing episode (−30 % total return over 720 steps, log-scale reward) has
  avg-step reward ≈ log(0.70) / 720 ≈ −4.9e-4.  Setting cvar_budget = −1e-3
  says "I tolerate an α-tail CVaR no worse than −1e-3 per step" — a loose
  initial constraint that tightens as training improves.

Usage
─────
    # In _build_single_env():
    env = CVaRConstraintWrapper(env, alpha=0.05, cvar_budget=-1e-3)
    env = Monitor(env)

    # In main():
    cvar_cb = CVaRCallback(alpha=0.05, cvar_budget=-1e-3, lambda_lr=0.01)
    callback = CallbackList([eval_cb, cvar_cb, ...])
"""

import numpy as np
from collections import deque
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CVaRConstraintWrapper(gym.Wrapper):
    """
    Apply a Lagrangian tail-risk penalty at each episode's terminal step.

    At episode end the wrapper computes:
        tail_shortfall = max(0, VaR_α − episode_avg_reward)
        terminal_reward -= λ × tail_shortfall   (only when shortfall > 0)

    λ and VaR_α are managed centrally by CVaRCallback and pushed here via
    set_cvar_params() — the wrapper itself never updates λ.

    Parameters
    ----------
    alpha        : tail probability α (0.05 = worst 5 % of episodes).
    cvar_budget  : acceptable CVaR floor in avg-step-reward units.
                   Used only as a prior for VaR_α before the callback fires.
    lambda_init  : starting Lagrange multiplier (0.0 = no penalty initially).
    """

    def __init__(
        self,
        env:          gym.Env,
        alpha:        float = 0.05,
        cvar_budget:  float = -1e-3,
        lambda_init:  float = 0.0,
    ):
        super().__init__(env)
        self._alpha       = alpha
        self._lambda      = float(lambda_init)
        self._var_alpha   = float(cvar_budget)   # conservative prior
        self._ep_reward   = 0.0
        self._ep_steps    = 0

    # ── External API ──────────────────────────────────────────────────────────

    def set_cvar_params(self, lambda_val: float, var_alpha: float) -> None:
        """Receive updated (λ, VaR_α) from CVaRCallback."""
        self._lambda    = float(lambda_val)
        self._var_alpha = float(var_alpha)

    def get_cvar_params(self) -> Dict[str, float]:
        return {"lambda": self._lambda, "var_alpha": self._var_alpha}

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        self._ep_reward = 0.0
        self._ep_steps  = 0
        return self.env.reset(**kwargs)

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        self._ep_reward += reward
        self._ep_steps  += 1

        done = terminated or truncated
        if done and self._ep_steps > 0 and self._lambda > 0.0:
            ep_avg_reward  = self._ep_reward / self._ep_steps
            tail_shortfall = max(0.0, self._var_alpha - ep_avg_reward)
            if tail_shortfall > 0.0:
                penalty = self._lambda * tail_shortfall
                reward -= penalty
                info["cvar_tail_shortfall"] = tail_shortfall
                info["cvar_penalty"]        = penalty

        info["cvar_lambda"]    = self._lambda
        info["cvar_var_alpha"] = self._var_alpha
        
        # BUG #7 FIX: Record the raw (unpenalized) average reward for the callback.
        # This prevents the feedback loop where penalty reduces the observed return,
        # which increases the estimated tail risk, leading to even more penalty.
        if done:
            info["episode_avg_raw_reward"] = self._ep_reward / max(self._ep_steps, 1)

        return obs, reward, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
# Callback
# ─────────────────────────────────────────────────────────────────────────────

class CVaRCallback(BaseCallback):
    """
    Estimate CVaR_α from completed episode returns and update Lagrange multiplier.

    Lagrange update rule (gradient ascent on the dual):
        violation = max(0, cvar_budget − CVaR_α)
        λ ← clip(λ + λ_lr × violation, 0, λ_max)

    After each update, broadcasts (λ, VaR_α) to all training environments via
    env_method so CVaRConstraintWrapper uses the latest values.

    Parameters
    ----------
    alpha            : tail probability — must match CVaRConstraintWrapper.
    cvar_budget      : CVaR floor (avg-step-reward units).
    lambda_lr        : Lagrange multiplier learning rate.
    lambda_max       : upper clip for λ (prevents reward collapse).
    buffer_episodes  : rolling window depth for CVaR estimation.
    update_freq      : episodes between Lagrange updates and env broadcasts.
    """

    def __init__(
        self,
        alpha:           float = 0.05,
        cvar_budget:     float = -1e-3,
        lambda_lr:       float = 0.01,
        lambda_max:      float = 10.0,
        buffer_episodes: int   = 100,
        update_freq:     int   = 10,
        verbose:         int   = 0,
    ):
        super().__init__(verbose)
        self._alpha          = alpha
        self._cvar_budget    = cvar_budget
        self._lambda_lr      = lambda_lr
        self._lambda_max     = lambda_max
        self._update_freq    = update_freq
        self._buffer: deque  = deque(maxlen=buffer_episodes)
        self._lambda         = 0.0
        self._var_alpha      = cvar_budget    # prior until enough data
        self._n_episodes     = 0
        self._last_update_ep = 0

    # ── SB3 hooks ─────────────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        # SB3 Monitor populates info["episode"] at the end of each episode
        for info in self.locals.get("infos", []):
            # BUG #7 FIX: Use the raw reward if available to avoid estimation bias.
            if "episode_avg_raw_reward" in info:
                avg_r = float(info["episode_avg_raw_reward"])
            elif "episode" in info:
                ep_r = float(info["episode"]["r"])
                ep_l = max(int(info["episode"]["l"]), 1)
                avg_r = ep_r / ep_l
            else:
                continue
                
            self._buffer.append(avg_r)
            self._n_episodes += 1

        # Trigger Lagrange update every update_freq completed episodes
        min_samples = max(10, int(1.0 / max(self._alpha, 1e-6)))
        if (
            self._n_episodes - self._last_update_ep >= self._update_freq
            and len(self._buffer) >= min_samples
        ):
            self._update_and_broadcast()
            self._last_update_ep = self._n_episodes

        return True

    def _on_training_end(self) -> None:
        if len(self._buffer) >= 5:
            self._update_and_broadcast()

    # ── Core logic ────────────────────────────────────────────────────────────

    def _update_and_broadcast(self) -> None:
        returns   = np.array(self._buffer, dtype=float)
        var_alpha = float(np.percentile(returns, self._alpha * 100))
        tail      = returns[returns <= var_alpha]
        cvar_alpha = float(np.mean(tail)) if len(tail) > 0 else var_alpha

        violation    = max(0.0, self._cvar_budget - cvar_alpha)
        self._lambda = float(np.clip(
            self._lambda + self._lambda_lr * violation,
            0.0,
            self._lambda_max,
        ))
        self._var_alpha = var_alpha

        # TensorBoard
        self.logger.record("cvar/var_alpha",        var_alpha)
        self.logger.record("cvar/cvar_alpha",        cvar_alpha)
        self.logger.record("cvar/lambda",            self._lambda)
        self.logger.record("cvar/budget_violation",  violation)
        self.logger.record("cvar/n_episodes",        self._n_episodes)

        if self.verbose > 0:
            print(
                f"  [CVaR] ep={self._n_episodes:5d}  "
                f"VaR_α={var_alpha:.5f}  CVaR_α={cvar_alpha:.5f}  "
                f"λ={self._lambda:.4f}  violation={violation:.5f}"
            )

        # Broadcast to all training envs (SubprocVecEnv and DummyVecEnv)
        try:
            self.training_env.env_method(
                "set_cvar_params", self._lambda, var_alpha
            )
        except Exception as _e:
            if self.verbose > 0:
                print(f"  [CVaR] env_method broadcast failed: {_e}")
