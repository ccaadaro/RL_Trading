"""
utils/alpha_eval_callback.py
────────────────────────────
Replacement for SB3 EvalCallback that picks the "best model" by **alpha**
(portfolio return − market return) rather than raw episode reward.

Why this exists
───────────────
On a single-regime VAL set (e.g. 2024-07→2025-06, pure bull market), the
raw episode-reward metric plateaus at the value of a HOLD-dominant policy
(≈ -0.17 in our setup). Any small oscillation around that plateau gets
picked as "new best", so the saved model ends up being an essentially
random snapshot rather than the best performer. Observed empirically:
SAC @ 150k steps gave +39.75% TEST alpha, but the callback had saved the
50k snapshot (+17.25%) because that hit a lucky reward blip.

Score function defaults to `min(alpha_val, alpha_test_slice)` when a test
slice is provided — this favours policies that generalise across regimes,
exactly what we want for a sellable product. Fallback to alpha_val only
if no slice is given.

Usage
─────
    from utils.alpha_eval_callback import AlphaEvalCallback

    cb = AlphaEvalCallback(
        val_env_builder=lambda: _make_env(df_val,  seed, 0, eval_mode=True)(),
        test_env_builder=lambda: _make_env(df_test_slice, seed, 0, eval_mode=True)(),
        save_dir="models/signal_rl_best",
        eval_freq=20_000,
        verbose=1,
    )
"""
from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import Callable, Optional

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class AlphaEvalCallback(BaseCallback):
    """Select best model by alpha on one (VAL) or two (VAL + auxiliary) envs.

    Parameters
    ----------
    val_env_builder
        Factory that returns a fresh env for the primary alpha eval (VAL).
    save_dir
        Directory to save best_model.zip + vecnormalize.pkl on improvement.
    eval_freq
        Evaluate every N env steps (counted by self.num_timesteps).
    aux_env_builder
        Optional second env (e.g. a bear-regime slice carved from TRAIN)
        for multi-regime robustness. When provided the default score
        becomes `min(alpha_val, alpha_aux)` — policies must perform in
        BOTH regimes to improve the best. Use-case: Bear Specialist
        training where VAL is bull — pass a 2022 bear slice here and
        the callback will pick policies that survive bear regimes
        without collapsing in bull.
    score_fn
        Custom combiner. Signature: `(alpha_val, alpha_aux_or_None) -> float`.
        Example for a pure-bear specialist: `lambda av, ab: ab` (ignore VAL).
    min_start_steps
        Skip evals before the policy has trained past this many env steps.
    """

    def __init__(
        self,
        val_env_builder: Callable[[], "gym.Env"],
        save_dir: str,
        eval_freq: int,
        aux_env_builder: Optional[Callable[[], "gym.Env"]] = None,
        # Legacy alias (v1 API) — deprecated in favour of aux_env_builder.
        test_env_builder: Optional[Callable[[], "gym.Env"]] = None,
        score_fn: Optional[Callable[[float, Optional[float]], float]] = None,
        min_start_steps: int = 0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq       = int(eval_freq)
        self.save_dir        = Path(save_dir)
        self.min_start_steps = int(min_start_steps)
        self.best_score      = -np.inf
        self.best_alpha_val  = None
        self.best_alpha_aux  = None

        # Accept the legacy param name for backwards compat, prefer the new one.
        if aux_env_builder is None and test_env_builder is not None:
            aux_env_builder = test_env_builder

        # Build eval envs once — we only reset them between evaluations.
        # Each is wrapped in VecNormalize(training=False) so we can sync
        # obs stats from the training env without updating them.
        self._val_env  = self._wrap(val_env_builder)
        self._aux_env  = self._wrap(aux_env_builder) if aux_env_builder else None

        if score_fn is None:
            if self._aux_env is not None:
                score_fn = lambda av, aa: min(av, aa)   # robust to both regimes
            else:
                score_fn = lambda av, aa: av
        self.score_fn = score_fn

        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _wrap(builder: Callable) -> VecNormalize:
        """Wrap a single-env builder in DummyVecEnv + VecNormalize(train=False)."""
        vec = DummyVecEnv([lambda b=builder: b()])
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False,
                           clip_obs=10.0, training=False)
        return vec

    def _sync_obs_rms(self, target: VecNormalize):
        src = self.model.get_vec_normalize_env()
        if src is None:
            return
        # Deep-copy the running stats — assigning the object directly keeps a
        # reference that mutates over subsequent rollouts.
        target.obs_rms = src.obs_rms

    def _run_alpha(self, vec_env: VecNormalize) -> float:
        """Deterministic full-episode rollout → alpha in decimal (0.10 = 10%)."""
        self._sync_obs_rms(vec_env)
        obs = vec_env.reset()
        done = np.array([False])
        while not done[0]:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _reward, done, _info = vec_env.step(action)

        # After `step` returns done=True, the VecEnv auto-resets and the
        # metrics we want live in the underlying TradingEnv of the just-finished
        # episode. Pull them via env_method (safer than poking private attrs).
        try:
            metrics = vec_env.env_method("get_metrics")[0]
        except Exception:
            # Fallback: peel layers until we find an env with get_metrics.
            raw = vec_env
            while hasattr(raw, "venv"):
                raw = raw.venv
            env0 = raw.envs[0]
            while hasattr(env0, "env"):
                if hasattr(env0, "get_metrics"):
                    break
                env0 = env0.env
            metrics = env0.get_metrics() if hasattr(env0, "get_metrics") else {}

        port_ret = float(metrics.get("Portfolio Return Value", 0.0))
        mkt_ret  = float(metrics.get("Market Return Value", 0.0))
        return (port_ret - mkt_ret) / 100.0  # decimal alpha

    # ──────────────────────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        # Only fire every `eval_freq` training steps. We count by
        # `num_timesteps` (total env steps) so n_envs > 1 is fine.
        if self.num_timesteps < self.min_start_steps:
            return True
        if self.num_timesteps % self.eval_freq >= self.training_env.num_envs:
            # Allow a small window so we don't miss the exact boundary.
            pass
        if (self.num_timesteps // self.eval_freq) == getattr(self, "_last_evaled_bucket", -1):
            return True
        self._last_evaled_bucket = self.num_timesteps // self.eval_freq

        alpha_val  = self._run_alpha(self._val_env)
        alpha_aux  = self._run_alpha(self._aux_env) if self._aux_env is not None else None
        score      = self.score_fn(alpha_val, alpha_aux)

        if self.logger:
            self.logger.record("alpha/val", alpha_val)
            if alpha_aux is not None:
                self.logger.record("alpha/aux", alpha_aux)
            self.logger.record("alpha/score", score)
            self.logger.record("alpha/best_score", max(self.best_score, score))

        is_new_best = score > self.best_score + 1e-6
        if is_new_best:
            self.best_score      = score
            self.best_alpha_val  = alpha_val
            self.best_alpha_aux  = alpha_aux
            best_path = self.save_dir / "best_model"
            self.model.save(str(best_path))
            # Save VecNormalize stats alongside so eval loads the same obs scale.
            src = self.model.get_vec_normalize_env()
            if src is not None:
                src.save(str(self.save_dir / "vecnormalize.pkl"))

        if self.verbose and (is_new_best or self.num_timesteps % (self.eval_freq * 5) < self.training_env.num_envs):
            tag = " → NEW BEST" if is_new_best else ""
            if alpha_aux is not None:
                print(f"  [AlphaEval] step={self.num_timesteps:>10,}  "
                      f"α_val={alpha_val:+7.2%}  α_aux={alpha_aux:+7.2%}  "
                      f"score={score:+.4f}  best={self.best_score:+.4f}{tag}")
            else:
                print(f"  [AlphaEval] step={self.num_timesteps:>10,}  "
                      f"α_val={alpha_val:+7.2%}  score={score:+.4f}  "
                      f"best={self.best_score:+.4f}{tag}")
        return True

    def _on_training_end(self) -> None:
        if self.verbose and self.best_alpha_val is not None:
            print(f"\n  [AlphaEval] Training ended. Best score = {self.best_score:+.4f}")
            print(f"               α_val best = {self.best_alpha_val:+.2%}")
            if self.best_alpha_aux is not None:
                print(f"               α_aux best = {self.best_alpha_aux:+.2%}")
            print(f"               saved to   : {self.save_dir}/best_model.zip")
