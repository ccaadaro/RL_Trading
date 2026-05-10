#!/usr/bin/env python3
"""
train_signal_rl.py — RL position sizing on top of LightGBM signal
═══════════════════════════════════════════════════════════════════

Layer 2 of the two-layer architecture:
  LightGBM (done) → signal_prob_v2_feature in parquet (v1 accepted as fallback)
  PPO             → learns WHEN and HOW MUCH to bet based on the signal

Why this is simpler than train_cleanrl.py:
  - Observation: 9 features (vs 104)
  - Policy: small MLP (vs MoE + LSTM)
  - Task: position sizing (vs direction prediction + sizing + risk)
  - Uses SB3 PPO (well-tested, no custom CleanRL needed)

Usage:
    python train_signal_rl.py
    python train_signal_rl.py --timesteps 3000000 --seed 42
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

from trading_env.signal_env import make_signal_env, _SIGNAL_PROB_CANDIDATES
from trading_env.risk_wrappers import MultiLevelRiskWrapper
from trading_env.trading_env import History
from utils.alpha_eval_callback import AlphaEvalCallback


# ══════════════════════════════════════════════════════════════════════════════
# REWARD
# ══════════════════════════════════════════════════════════════════════════════

class SignReward:
    """Economic-PnL reward with optional legacy sign-only mode.

    mode="magnitude" (default): reward = log(portfolio_t / portfolio_{t-1})
        Uses the portfolio's post-fee change, so fees/slippage are already
        accounted for by the env. Trade-change penalty is optional on top
        (kept for continuity with the sign-only run history).

    mode="sign": legacy behaviour — sign(log_ret) × position. Kept so we
        can A/B against the original run on the same parquet.

    The LightGBM layer already decides direction; this reward teaches the
    RL to size by conviction. Magnitude mode gives it a denser, economically
    honest signal — a 2% winning bar is not the same as a 0.05% one.
    """
    def __init__(
        self,
        trade_penalty: float = 2e-4,
        mode: str = "magnitude",
        clip: float = 0.05,   # clip per-bar log return to ±5% (tail protection)
    ):
        if mode not in ("magnitude", "sign"):
            raise ValueError(f"Unknown reward mode: {mode!r}")
        self.trade_penalty = trade_penalty
        self.mode          = mode
        self.clip          = clip

    def __call__(self, history: History) -> float:
        if history["idx", -1] == 0:
            return 0.0

        # Market log-return for logging + sign-mode reward.
        try:
            close_now  = float(history["data_close", -1])
            close_prev = float(history["data_close", -2])
        except (KeyError, IndexError):
            return 0.0
        if close_prev <= 1e-8:
            return 0.0
        log_ret = float(np.log(close_now / close_prev))
        history["reward_raw", -1] = log_ret

        try:
            position = float(history["position", -1])
        except (KeyError, IndexError):
            position = 0.0

        if self.mode == "sign":
            r = float(np.sign(log_ret)) * position
        else:
            # Portfolio PnL already nets fees/slippage — use it directly.
            try:
                pv_now  = float(history["portfolio_valuation", -1])
                pv_prev = float(history["portfolio_valuation", -2])
            except (KeyError, IndexError):
                pv_now = pv_prev = 0.0
            if pv_prev > 1e-10 and pv_now > 1e-10:
                r = float(np.log(pv_now / pv_prev))
            else:
                # Portfolio history unavailable — fall back to signed PnL
                # so the episode isn't silently zero.
                r = log_ret * position
            if self.clip > 0:
                r = float(np.clip(r, -self.clip, self.clip))

        if self.trade_penalty > 0:
            try:
                if history["position_index", -1] != history["position_index", -2]:
                    r -= self.trade_penalty
            except (KeyError, IndexError, ValueError):
                pass
        return r

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CFG = {
    "data": {
        "path":       "cache/data_v1_430340a861af5f8f9bcbd7a4ca16ba95.parquet",
        "train_start": "2019-01-01",
        "train_end":   "2024-06-30",
        "val_start":   "2024-07-01",
        "val_end":     "2025-06-30",
        "test_start":  "2025-07-01",
    },
    "env": {
        "initial_balance":    1_000.0,
        "fee_rate":           5e-4,
        "slippage_bps":       2,
        "episode_bars":       720,    # ~30 days per episode
        "positions":          [-1, 0, 1],
    },
    "risk": {
        "dd_hard":            0.15,   # force flat if drawdown > 15%
        "cooldown_steps":     24,
        "apply_in_training":  True,
        "apply_in_eval":      True,
    },
    "ppo": {
        "policy":             "MlpPolicy",
        "learning_rate":      3e-4,
        "n_steps":            2048,
        "batch_size":         128,
        "n_epochs":           10,
        "gamma":              0.99,
        "gae_lambda":         0.95,
        "clip_range":         0.2,
        "ent_coef":           0.02,
        "vf_coef":            0.5,
        "max_grad_norm":      0.5,
        "policy_kwargs": {
            "net_arch":       [64, 64],   # small MLP — 9 features don't need more
            "activation_fn":  torch.nn.Tanh,
        },
    },
    "training": {
        "total_timesteps":    2_000_000,
        "n_envs":             8,
        "eval_freq":          20_000,    # evaluate every 20k steps
        "n_eval_episodes":    3,
        "patience":           800_000,   # stop if no improvement for 800k steps (iter2: 2x)
        "log_dir":            "logs_signal_rl",
        "model_dir":          "models",
        "seed":               42,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# ENV FACTORY
# ══════════════════════════════════════════════════════════════════════════════

# Iter2 (2026-04-17): trade_penalty 2e-4 → 5e-5 (was ~20% of typical
# hourly move → agent over-preferred HOLD in VAL bull). Clip 0.05 → 0.15
# so big winning bars aren't capped (was cutting off the learning signal
# on high-conviction LONGs during the 2024-25 bull).
_REWARD_FN = SignReward(trade_penalty=5e-5, mode="magnitude", clip=0.15)


def _make_env(df: pd.DataFrame, seed: int, rank: int, eval_mode: bool = False):
    """Returns a callable that creates one env instance (for SubprocVecEnv)."""
    def _init():
        env = make_signal_env(
            df,
            positions          = CFG["env"]["positions"],
            fee_rate           = CFG["env"]["fee_rate"],
            slippage_bps       = CFG["env"]["slippage_bps"],
            portfolio_initial_value = CFG["env"]["initial_balance"],
            initial_position   = 0,
            max_episode_duration = CFG["env"]["episode_bars"] if not eval_mode else "max",
            verbose            = 0,
            name               = f"SignalEnv-{'eval' if eval_mode else rank}",
            log_frequency      = 9999,
            reward_function    = _REWARD_FN,
        )
        if CFG["risk"]["apply_in_training"] or (eval_mode and CFG["risk"]["apply_in_eval"]):
            env = MultiLevelRiskWrapper(
                env,
                dd_hard          = CFG["risk"]["dd_hard"],
                cooldown_steps   = CFG["risk"]["cooldown_steps"],
            )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopCallback(BaseCallback):
    """Stop training if eval metric hasn't improved for `patience` steps."""

    def __init__(self, patience: int, verbose: int = 1):
        super().__init__(verbose)
        self.patience       = patience
        self.best_mean_reward = -np.inf
        self.steps_no_improve = 0

    def _on_step(self) -> bool:
        # EvalCallback stores mean reward in parent's locals — read from logger
        if len(self.model.ep_info_buffer) == 0:
            return True
        mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
        if mean_reward > self.best_mean_reward + 1e-4:
            self.best_mean_reward = mean_reward
            self.steps_no_improve = 0
        else:
            self.steps_no_improve += self.training_env.num_envs
            if self.steps_no_improve >= self.patience:
                if self.verbose:
                    print(f"\n  [EarlyStop] No improvement for {self.patience} steps — stopping.")
                return False
        return True


class SignalAlignmentCallback(BaseCallback):
    """Log how often the agent's position aligns with the LightGBM signal."""

    def __init__(self, log_freq: int = 20_000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq  = log_freq
        self._calls    = 0
        self._aligned  = 0
        self._total    = 0

    def _on_step(self) -> bool:
        self._calls += 1
        if self._calls % self.log_freq != 0:
            return True

        # Try to read recent obs (signal_prob is obs[:,2] — 3rd feature col)
        try:
            obs = self.locals.get("obs_tensor") or self.locals.get("new_obs")
            if obs is None:
                return True
            if hasattr(obs, "cpu"):
                obs = obs.cpu().numpy()
            obs = np.asarray(obs)
            if obs.ndim == 1:
                obs = obs[np.newaxis]

            # signal_prob is index 2 in [vol180, trend180, signal_prob, signal_conf, ...]
            signal_probs = obs[:, 2]
            actions      = self.locals.get("actions")
            if actions is None:
                return True
            actions = np.asarray(actions).flatten()

            # Alignment: long when prob>0.5, short when prob<0.5
            long_signal  = signal_probs > 0.5
            short_signal = signal_probs < 0.5
            long_action  = actions > 0.1
            short_action = actions < -0.1

            aligned = np.sum((long_signal & long_action) | (short_signal & short_action))
            total   = len(actions)
            pct     = 100 * aligned / max(total, 1)

            if self.verbose and self.num_timesteps % (self.log_freq * 5) == 0:
                print(f"  [SignalAlign] step={self.num_timesteps:,}  "
                      f"alignment={pct:.1f}%  n={total}")
            if self.logger:
                self.logger.record("signal/alignment_pct", pct)
        except Exception:
            pass
        return True


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_on_split(model, df_split: pd.DataFrame, label: str, n_episodes: int = 1):
    """Run deterministic episodes on a data split and print metrics."""
    env = make_signal_env(
        df_split,
        positions            = CFG["env"]["positions"],
        fee_rate             = CFG["env"]["fee_rate"],
        slippage_bps         = CFG["env"]["slippage_bps"],
        portfolio_initial_value = CFG["env"]["initial_balance"],
        initial_position     = 0,
        max_episode_duration = "max",
        verbose              = 0,
        name                 = label,
        log_frequency        = 9999,
    )
    if CFG["risk"]["apply_in_eval"]:
        env = MultiLevelRiskWrapper(env, dd_hard=CFG["risk"]["dd_hard"],
                                    cooldown_steps=CFG["risk"]["cooldown_steps"])

    portfolio_returns = []
    market_returns    = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

        # Final metrics from env
        try:
            metrics = env.unwrapped.get_metrics()
            portfolio_returns.append(metrics.get("Portfolio Return Value", 0))
            market_returns.append(metrics.get("Market Return Value", 0))
        except Exception:
            pass

    if portfolio_returns:
        port_ret = np.mean(portfolio_returns)
        mkt_ret  = np.mean(market_returns)
        alpha    = port_ret - mkt_ret
        print(f"  [{label}]  portfolio={port_ret:+.2f}%  market={mkt_ret:+.2f}%  "
              f"alpha={alpha:+.2f}%")

    env.close()
    return np.mean(portfolio_returns) if portfolio_returns else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=CFG["training"]["total_timesteps"])
    parser.add_argument("--seed",      type=int, default=CFG["training"]["seed"])
    parser.add_argument("--envs",      type=int, default=CFG["training"]["n_envs"])
    parser.add_argument("--lr",        type=float, default=CFG["ppo"]["learning_rate"])
    args = parser.parse_args()

    CFG["training"]["total_timesteps"] = args.timesteps
    CFG["training"]["seed"]            = args.seed
    CFG["training"]["n_envs"]          = args.envs
    CFG["ppo"]["learning_rate"]        = args.lr

    print("=" * 65)
    print("  SIGNAL RL — PPO position sizing")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────
    df = pd.read_parquet(CFG["data"]["path"])
    if not any(c in df.columns for c in _SIGNAL_PROB_CANDIDATES):
        raise RuntimeError(
            f"No LightGBM signal column found in parquet "
            f"(looked for {_SIGNAL_PROB_CANDIDATES}). "
            "Run scripts/retrain_signal_v2.py (preferred) or "
            "scripts/train_signal_model.py (legacy)."
        )
    signal_src = next(c for c in _SIGNAL_PROB_CANDIDATES if c in df.columns)
    print(f"  Signal column in use: {signal_src}")

    d = CFG["data"]
    df_train = df[d["train_start"]:d["train_end"]].copy()
    df_val   = df[d["val_start"]:d["val_end"]].copy()
    df_test  = df[d["test_start"]:].copy()

    print(f"  Train : {len(df_train)} bars  "
          f"({df_train.index[0].date()} → {df_train.index[-1].date()})")
    print(f"  Val   : {len(df_val)} bars")
    print(f"  Test  : {len(df_test)} bars")

    # ── Vectorised training envs ──────────────────────────────────────────
    n_envs = CFG["training"]["n_envs"]
    seed   = CFG["training"]["seed"]

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    train_env = vec_cls([
        _make_env(df_train, seed, rank, eval_mode=False)
        for rank in range(n_envs)
    ])
    train_env = VecNormalize(
        train_env,
        norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0,
        gamma=CFG["ppo"]["gamma"],
    )

    # ── Single eval env (val set, full episodes) ──────────────────────────
    eval_env = DummyVecEnv([_make_env(df_val, seed, 0, eval_mode=True)])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False,
        clip_obs=10.0, training=False,
    )
    # Sync normalisation stats from training env
    eval_env.obs_rms = train_env.obs_rms

    # ── Model ────────────────────────────────────────────────────────────
    log_dir   = CFG["training"]["log_dir"]
    model_dir = CFG["training"]["model_dir"]
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    ppo_kwargs = {k: v for k, v in CFG["ppo"].items()}
    ppo_kwargs["verbose"]        = 1
    ppo_kwargs["tensorboard_log"] = log_dir
    ppo_kwargs["seed"]            = seed
    ppo_kwargs["device"]          = "cpu"   # MLP with 9 features is faster on CPU

    model = PPO(env=train_env, **ppo_kwargs)
    print(f"\n  Policy: {CFG['ppo']['policy']}  "
          f"net_arch={CFG['ppo']['policy_kwargs']['net_arch']}")
    print(f"  Obs dim: {train_env.observation_space.shape[0]}")
    print(f"  Device : {model.device}")
    print(f"  n_envs : {n_envs}  lr={CFG['ppo']['learning_rate']}  "
          f"timesteps={CFG['training']['total_timesteps']:,}")

    # ── Callbacks ────────────────────────────────────────────────────────
    best_model_path = f"{model_dir}/signal_rl_best"

    # Alpha-based "best model" selection. The previous EvalCallback used
    # raw episode reward, which plateaus at HOLD-dominant policies on the
    # single-regime VAL set and captures the wrong snapshot.
    alpha_eval_cb = AlphaEvalCallback(
        val_env_builder  = _make_env(df_val, seed, 0, eval_mode=True),
        save_dir         = best_model_path,
        eval_freq        = CFG["training"]["eval_freq"],
        min_start_steps  = 50_000,   # let policy move past near-random behaviour
        verbose          = 1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq   = max(100_000 // n_envs, 1),
        save_path   = f"{model_dir}/checkpoints_signal",
        name_prefix = "signal_rl",
        verbose     = 0,
    )
    align_cb     = SignalAlignmentCallback(log_freq=20_000, verbose=1)
    early_stop   = EarlyStopCallback(patience=CFG["training"]["patience"], verbose=1)

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Training started...")
    print(f"{'─'*65}\n")

    # Only enable the rich progress bar in interactive shells — it raises
    # TqdmExperimentalWarning and crashes at init when stdout is a pipe/log.
    import sys as _sys
    _progress_bar = _sys.stdout.isatty()

    model.learn(
        total_timesteps  = CFG["training"]["total_timesteps"],
        callback         = [alpha_eval_cb, checkpoint_cb, align_cb, early_stop],
        progress_bar     = _progress_bar,
        reset_num_timesteps = True,
    )

    # ── Save final ────────────────────────────────────────────────────────
    final_path = f"{model_dir}/signal_rl_final"
    model.save(final_path)
    train_env.save(f"{model_dir}/signal_rl_vecnorm.pkl")
    print(f"\n  Model saved → {final_path}")

    # ── Final evaluation ──────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  FINAL EVALUATION")
    print(f"{'═'*65}")

    # Load best model for evaluation
    best_model = PPO.load(
        f"{best_model_path}/best_model",
        env=eval_env,
        device=model.device,
    )

    evaluate_on_split(best_model, df_val,  "VAL ")
    evaluate_on_split(best_model, df_test, "TEST")

    train_env.close()
    eval_env.close()
    print("\n  Done.")


if __name__ == "__main__":
    main()
