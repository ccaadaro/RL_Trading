#!/usr/bin/env python3
"""
train_cleanrl.py — CleanRL-style Recurrent PPO for RL Trading
═════════════════════════════════════════════════════════════════

Single-file training loop with explicit PyTorch control.
No stable-baselines3 dependency.

Architecture:
    Actor:  obs → MoE(8 experts) → LSTM(256) → MLP(128→64) → Categorical(3)
    Critic: obs → SimpleMLP(256→128→64) → LSTM(256) → MLP(128→64) → V(s)

Usage:
    python train_cleanrl.py --seed 42
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# ── project imports (no SB3) ─────────────────────────────────────────────────
from trading_env.trading_env import TradingEnv, differential_sharpe_reward
from trading_env.risk_wrappers import MultiLevelRiskWrapper
from trading_env.cvar_wrapper import CVaRConstraintWrapper
from trading_env.regime_sampler import RegimeBalancedWrapper
from utils.schedulers import cosine_lr
from utils.turbulence import add_turbulence_feature
from models.extractors import (
    MoEFeaturesExtractor, SimpleCriticExtractor, get_regime_feature_indices,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CONFIG: Dict[str, Any] = {
    "data": {
        "cache_path": "cache/data_v1_430340a861af5f8f9bcbd7a4ca16ba95.parquet",
        "train_start": "2019-01-01",
        "train_end":   "2024-06-30",
        "val_start":   "2024-07-01",
        "val_end":     "2025-06-30",
        "test_start":  "2025-07-01",
        "eth_path":    "../../data/binance/ETH_USDT-1h.feather",
    },
    "env": {
        "positions":             (-1.0, 0.0, 1.0),  # Run 063: restore HOLD — let agent learn to step aside
        "initial_balance":       1_000.0,
        "fee_rate":              5e-4,
        "training_fee_rate":     0.0,       # Run 055: zero fees to break symmetric noise paralysis
        "slippage_bps":          2,
        "window_size":           None,
        "train_episode_bars":    720,
    },
    "reward": {
        "type":                    "sign",   # Run 063: Sign Reward — clip returns to ±1, kill magnitude-based beta trap
        "eta":                     0.01,
        "demean_market":           False,
        "fractal_alignment_scale": 0.0,
        "trade_penalty":           0.0,
        "inactivity_penalty":      0.0,
        "inactivity_window":       12,
        "reward_scale":            1.0,     # Run 063: normalization handles scaling now
        "conviction_bonus":        0.0,
    },
    "model": {
        "learning_rate":    5e-5,
        "n_steps":          1024,      # Run 057: shorter rollouts → punchier updates, less gradient cancellation
        "batch_size":       16,
        "n_epochs":         3,
        "gamma":            0.80,          # Run 061: short horizon — force critic to predict local microstructure
        "gae_lambda":       0.90,          # Run 055: weight short-lived alpha signals more
        "clip_range":       0.10,
        "ent_coef":         0.05,          # Run 064: strong exploration for 3-action space
        "entropy_floor":    0.4,           # Run 064: fraction of max entropy — floor prevents collapse
        "normalize_advantages": True,      # Run 063: re-enable — stretches ±1 sign rewards to N(0,1)
        "vf_coef":          1.0,
        "max_grad_norm":    0.5,
        "target_kl":        0.01,
        "lstm_hidden_size": 256,
        "n_lstm_layers":    1,
    },
    "training": {
        "total_timesteps": 2_000_000,
        "n_envs":          16,
        "eval_freq":       25_000,
        "patience":        500_000,
        "log_dir":         "logs_stable",
        "seeds":           [42, 123, 456],
    },
    "risk": {
        "dd_hard":              0.15,
        "turbulence_threshold": 1.5,
        "atr_stop_mult":        2.0,
        "cooldown_steps":       24,
        "turbulence_col":       "turbulence_feature",
        "atr_col":              "feature_atr",
        "apply_in_training":    True,
        "apply_in_eval":        True,
    },
    "moe": {
        "enabled":          True,
        "n_experts":        8,
        "expert_dim":       64,
        "gate_hidden_dim":  32,
        "gate_entropy_coef": 0.01,         # Run 064: re-enable — prevent expert starvation
        "gate_temperature":  1.0,          # Run 064: neutral (was 0.5, too peaked)
        "load_balance_alpha": 0.01,        # Run 064: re-enable — penalize routing collapse
        "gate_log_freq":    100_000,
        "gate_monitor_steps": 1_000,
    },
    "cvar": {
        "enabled":          True,
        "alpha":            0.05,
        "cvar_budget":      -1e-3,
        "lambda_lr":        0.01,
        "lambda_max":       10.0,
        "buffer_episodes":  100,
        "update_freq":      10,
        "apply_in_training": True,
        "apply_in_eval":     False,
    },
    "burnin": {
        "burnin_steps":     50_000,
        "actor_lr_mult":    5.0,       # Run 064: 20x→5x — less aggressive, prevents entropy collapse
    },
    "inversion": {
        "enabled":    False,           # Run 063: KILLED — causes bootstrapping paradox (zero expected gradient)
        "p_invert":   0.5,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# REWARD
# ══════════════════════════════════════════════════════════════════════════════

class FractalGuidedReward:
    def __init__(self, alignment_scale: float = 2.5e-4, trade_penalty: float = 2e-3):
        self.alignment_scale = alignment_scale
        self.trade_penalty   = trade_penalty

    def __call__(self, history) -> float:
        pv_now  = history["portfolio_valuation", -1]
        pv_prev = history["portfolio_valuation", -2]
        pnl_r   = float(np.log(pv_now / pv_prev)) if pv_prev > 1e-8 else 0.0
        try:
            position  = float(history["position", -1])
            deception = float(history["data_fractal_deception_feature", -1])
            pnl_r    += self.alignment_scale * position * deception
        except (KeyError, IndexError, ValueError):
            pass
        try:
            pos_now  = float(history["position", -1])
            pos_prev = float(history["position", -2])
            if abs(pos_now - pos_prev) > 1e-6:
                pnl_r -= self.trade_penalty
        except (KeyError, IndexError, ValueError):
            pass
        return pnl_r


class DSRFractalReward:
    def __init__(self, alignment_scale=2.5e-4, trade_penalty=2e-3,
                 eta=0.01, dsr_a_init=0.0, dsr_b_init=0.0,
                 inactivity_penalty=0.0, inactivity_window=12,
                 demean_market=True, conviction_bonus=0.0):
        self.alignment_scale    = alignment_scale
        self.trade_penalty      = trade_penalty
        self.eta                = eta
        self.dsr_a_init         = dsr_a_init
        self.dsr_b_init         = dsr_b_init
        self.inactivity_penalty = inactivity_penalty
        self.inactivity_window  = inactivity_window
        self.demean_market      = demean_market
        self.conviction_bonus   = conviction_bonus

    def __call__(self, history) -> float:
        r = differential_sharpe_reward(
            history,
            eta=self.eta,
            fee_penalty=0.0,
            hold_penalty=0.0,
            demean_market=self.demean_market,
        )
        try:
            position  = float(history["position", -1])
            deception = float(history["data_fractal_deception_feature", -1])
            r += self.alignment_scale * position * deception
        except (KeyError, IndexError, ValueError):
            pass
        try:
            pos_now  = float(history["position", -1])
            pos_prev = float(history["position", -2])
            if abs(pos_now - pos_prev) > 1e-6:
                r -= self.trade_penalty
        except (KeyError, IndexError, ValueError):
            pass
        # Conviction bonus: break zero-mean symmetry for in-market positions
        if self.conviction_bonus > 0:
            try:
                pos = float(history["position", -1])
                if abs(pos) > 1e-6:  # LONG or SHORT
                    r += self.conviction_bonus
            except (KeyError, IndexError, ValueError):
                pass
        if self.inactivity_penalty > 0:
            try:
                pos = float(history["position", -1])
                if abs(pos) < 1e-6:
                    streak = 0
                    for i in range(2, min(self.inactivity_window + 2, len(history))):
                        if abs(float(history["position", -i])) < 1e-6:
                            streak += 1
                        else:
                            break
                    if streak >= self.inactivity_window:
                        r -= self.inactivity_penalty
            except (KeyError, IndexError, ValueError):
                pass
        return r


class SignReward:
    """Run 064: Direction-accuracy reward using market return × position.

    R_t = sign(market_log_return) * position

    Decoupled from portfolio NAV — uses raw market candle direction only.
    Every correct-direction tick = +1, every wrong = -1, HOLD = 0.
    Destroys magnitude-based Beta Trap while preserving gradient signal.
    """
    def __init__(self, trade_penalty: float = 0.0):
        self.trade_penalty = trade_penalty

    def __call__(self, history) -> float:
        if history['idx', -1] == 0:
            return 0.0

        # Market log return (raw candle direction, not portfolio)
        try:
            close_now  = float(history['data_close', -1])
            close_prev = float(history['data_close', -2])
        except (KeyError, IndexError):
            return 0.0
        if close_prev <= 1e-8:
            return 0.0

        market_sign = float(np.sign(np.log(close_now / close_prev)))

        # Position: -1 (SHORT), 0 (HOLD), +1 (LONG)
        try:
            position = float(history['position', -1])
        except (KeyError, IndexError):
            position = 0.0

        # Core reward: direction accuracy × position
        r = market_sign * position

        # Transaction cost penalty
        if self.trade_penalty > 0:
            try:
                if history['position_index', -1] != history['position_index', -2]:
                    r -= self.trade_penalty
            except (KeyError, IndexError, ValueError):
                pass

        history['reward_raw', -1] = float(np.log(close_now / close_prev))
        return r


# ══════════════════════════════════════════════════════════════════════════════
# MARKET INVERSION (Symmetric Data Augmentation)
# ══════════════════════════════════════════════════════════════════════════════

# Keywords that identify directional features (centered ~0, sign = direction)
_DIRECTIONAL_KEYWORDS = (
    "return", "ema_ratio", "ma_bias", "rsi_",
    "bb_position", "cvd_", "delta_", "dollar_delta",
    "fractal_zone_dir", "fractal_deception", "fractal_zone_type",
    "funding_rate", "futures_mark_basis",
    "microprice_trend", "vol_imbalance", "volume_imbalance_intraday",
    "eth_btc_ratio", "eth_btc_trend", "eth_btc_zscore",
    "global_ls_contrarian", "oi_change",
    "cross_section_market_score",
    "cross_neighbor_mean_return", "cross_neighbor_weighted_return",
)


def _build_directional_mask(feature_columns: List[str]) -> np.ndarray:
    """Return boolean mask: True for features whose sign encodes direction."""
    mask = np.zeros(len(feature_columns), dtype=bool)
    for i, col in enumerate(feature_columns):
        if any(kw in col for kw in _DIRECTIONAL_KEYWORDS):
            mask[i] = True
    return mask


class InvertedMarketWrapper(gym.Wrapper):
    """50% chance per episode to flip directional features and reward.

    In the inverted episode the agent sees a mirror universe where bull → bear
    and vice versa, eliminating global directional drift from the expected
    value of any constant policy.
    """

    def __init__(self, env: gym.Env, feature_columns: List[str],
                 p_invert: float = 0.5):
        super().__init__(env)
        self.p_invert = p_invert
        self._dir_mask = _build_directional_mask(feature_columns)
        # Account for dynamic features appended after static features
        n_obs = env.observation_space.shape[-1]
        if n_obs > len(self._dir_mask):
            extra = n_obs - len(self._dir_mask)
            self._dir_mask = np.concatenate([
                self._dir_mask, np.zeros(extra, dtype=bool)
            ])
        self._inverted = False
        n_dir = int(self._dir_mask.sum())
        print(f"    [InvertedMarket] {n_dir}/{len(self._dir_mask)} directional features")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._inverted = np.random.random() < self.p_invert
        if self._inverted:
            obs = self._flip_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._inverted:
            obs = self._flip_obs(obs)
            reward = -reward
        return obs, reward, terminated, truncated, info

    def _flip_obs(self, obs: np.ndarray) -> np.ndarray:
        out = obs.copy()
        if out.ndim == 1:
            out[self._dir_mask[:len(out)]] *= -1.0
        else:
            out[:, self._dir_mask[:out.shape[-1]]] *= -1.0
        return out


# ══════════════════════════════════════════════════════════════════════════════
# ACTION DISCRETIZER
# ══════════════════════════════════════════════════════════════════════════════

class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, positions: Sequence[float] = (0.0, 1.0)):
        super().__init__(env)
        self._positions = np.array(sorted(positions), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self._positions))

    def action(self, action) -> float:
        return float(self._positions[int(action)])


# ══════════════════════════════════════════════════════════════════════════════
# RECURRENT ACTOR-CRITIC
# ══════════════════════════════════════════════════════════════════════════════

def _build_mlp(input_dim: int, layer_dims: List[int]) -> nn.Sequential:
    layers = []
    prev = input_dim
    for dim in layer_dims:
        layers.extend([nn.Linear(prev, dim), nn.ReLU()])
        prev = dim
    return nn.Sequential(*layers)


class RecurrentActorCritic(nn.Module):
    """Decoupled recurrent actor-critic with MoE actor and MLP critic."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lstm_hidden: int = 256,
        n_lstm_layers: int = 1,
        pi_net_arch: List[int] = [128, 64],
        vf_net_arch: List[int] = [128, 64],
        moe_kwargs: Optional[dict] = None,
        critic_features_dim: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.lstm_hidden = lstm_hidden
        self.n_lstm_layers = n_lstm_layers

        # ── Feature extractors ────────────────────────────────────────────
        if moe_kwargs is not None:
            self.pi_extractor = MoEFeaturesExtractor(obs_dim=obs_dim, **moe_kwargs)
        else:
            self.pi_extractor = nn.Sequential(
                nn.Linear(obs_dim, 64), nn.Tanh()
            )
            self.pi_extractor.features_dim = 64

        self.vf_extractor = SimpleCriticExtractor(obs_dim=obs_dim,
                                                   features_dim=critic_features_dim)

        # ── LSTMs (separate for actor and critic) ─────────────────────────
        self.lstm_actor = nn.LSTM(
            self.pi_extractor.features_dim, lstm_hidden,
            num_layers=n_lstm_layers, batch_first=False,
        )
        self.lstm_critic = nn.LSTM(
            self.vf_extractor.features_dim, lstm_hidden,
            num_layers=n_lstm_layers, batch_first=False,
        )

        # ── MLP heads ─────────────────────────────────────────────────────
        self.pi_mlp = _build_mlp(lstm_hidden, pi_net_arch)
        self.vf_mlp = _build_mlp(lstm_hidden, vf_net_arch)

        # ── Output heads ──────────────────────────────────────────────────
        self.action_net = nn.Linear(pi_net_arch[-1], n_actions)
        self.value_net = nn.Linear(vf_net_arch[-1], 1)

        # ── Orthogonal init: flat logits at step 0 so argmax isn't biased
        nn.init.orthogonal_(self.action_net.weight, gain=0.01)
        nn.init.constant_(self.action_net.bias, 0.0)

    @staticmethod
    def _process_sequence(
        lstm: nn.LSTM,
        features: torch.Tensor,
        lstm_states: tuple,
        episode_starts: torch.Tensor,
    ) -> tuple:
        """
        LSTM forward pass with hidden state zeroing at episode boundaries.
        Replicates sb3_contrib's _process_sequence exactly.

        features:       (padded_batch_size, feature_dim)
        lstm_states:    ((n_layers, n_seq, hidden), (n_layers, n_seq, hidden))
        episode_starts: (padded_batch_size,)

        Returns:
            output: (padded_batch_size, hidden_dim)
            new_lstm_states: same shape as input
        """
        n_seq = lstm_states[0].shape[1]
        # (padded_batch_size, feat) → (max_len, n_seq, feat)
        features_seq = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        ep_starts_seq = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        # Fast path: no episode boundaries in batch
        if torch.all(ep_starts_seq == 0.0):
            lstm_out, lstm_states = lstm(features_seq, lstm_states)
            lstm_out = torch.flatten(lstm_out.transpose(0, 1), start_dim=0, end_dim=1)
            return lstm_out, lstm_states

        # Slow path: step-by-step with hidden state resets
        outputs = []
        for feat_step, ep_start_step in zip(features_seq, ep_starts_seq):
            # Zero hidden state where episode starts
            h = (1.0 - ep_start_step).view(1, n_seq, 1) * lstm_states[0]
            c = (1.0 - ep_start_step).view(1, n_seq, 1) * lstm_states[1]
            out, lstm_states = lstm(feat_step.unsqueeze(0), (h, c))
            outputs.append(out)

        lstm_out = torch.flatten(
            torch.cat(outputs, dim=0).transpose(0, 1), start_dim=0, end_dim=1
        )
        return lstm_out, lstm_states

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        lstm_pi: tuple,
        lstm_vf: tuple,
        episode_starts: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        """
        Full forward pass through both actor and critic.

        Returns: action, log_prob, entropy, value, (new_lstm_pi, new_lstm_vf)
        """
        # Actor path
        pi_features = self.pi_extractor(obs)
        pi_out, new_lstm_pi = self._process_sequence(
            self.lstm_actor, pi_features, lstm_pi, episode_starts
        )
        pi_hidden = self.pi_mlp(pi_out)
        logits = self.action_net(pi_hidden)

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.mode if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # Critic path
        vf_features = self.vf_extractor(obs)
        vf_out, new_lstm_vf = self._process_sequence(
            self.lstm_critic, vf_features, lstm_vf, episode_starts
        )
        vf_hidden = self.vf_mlp(vf_out)
        value = self.value_net(vf_hidden).squeeze(-1)

        return action, log_prob, entropy, value, (new_lstm_pi, new_lstm_vf)

    def get_value(
        self,
        obs: torch.Tensor,
        lstm_vf: tuple,
        episode_starts: torch.Tensor,
    ):
        """Critic-only forward for GAE bootstrap."""
        vf_features = self.vf_extractor(obs)
        vf_out, new_lstm_vf = self._process_sequence(
            self.lstm_critic, vf_features, lstm_vf, episode_starts
        )
        vf_hidden = self.vf_mlp(vf_out)
        return self.value_net(vf_hidden).squeeze(-1), new_lstm_vf

    @property
    def actor_parameters(self) -> list:
        return list(self.pi_extractor.parameters()) + \
               list(self.lstm_actor.parameters()) + \
               list(self.pi_mlp.parameters()) + \
               list(self.action_net.parameters())

    @property
    def critic_parameters(self) -> list:
        return list(self.vf_extractor.parameters()) + \
               list(self.lstm_critic.parameters()) + \
               list(self.vf_mlp.parameters()) + \
               list(self.value_net.parameters())


# ══════════════════════════════════════════════════════════════════════════════
# RECURRENT ROLLOUT BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class RecurrentBatch(NamedTuple):
    observations: torch.Tensor       # (padded_batch, obs_dim)
    actions: torch.Tensor            # (padded_batch,)
    old_values: torch.Tensor         # (padded_batch,)
    old_log_probs: torch.Tensor      # (padded_batch,)
    advantages: torch.Tensor         # (padded_batch,)
    returns: torch.Tensor            # (padded_batch,)
    episode_starts: torch.Tensor     # (padded_batch,)
    mask: torch.Tensor               # (padded_batch,)  1=real, 0=padding
    lstm_states_pi: tuple            # ((n_layers, n_seq, H), (n_layers, n_seq, H))
    lstm_states_vf: tuple


class RecurrentRolloutBuffer:
    """Rollout buffer with LSTM state tracking and sequence-padded minibatches."""

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_dim: int,
        n_lstm_layers: int,
        lstm_hidden: int,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.n_lstm_layers = n_lstm_layers
        self.lstm_hidden = lstm_hidden
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.pos = 0
        self.full = False
        self.reset()

    def reset(self):
        self.pos = 0
        self.full = False
        S, E = self.n_steps, self.n_envs
        self.observations = np.zeros((S, E, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((S, E), dtype=np.int64)
        self.rewards = np.zeros((S, E), dtype=np.float32)
        self.values = np.zeros((S, E), dtype=np.float32)
        self.log_probs = np.zeros((S, E), dtype=np.float32)
        self.episode_starts = np.zeros((S, E), dtype=np.float32)
        self.advantages = np.zeros((S, E), dtype=np.float32)
        self.returns = np.zeros((S, E), dtype=np.float32)
        # LSTM states: (n_steps, n_layers, n_envs, hidden)
        L, H = self.n_lstm_layers, self.lstm_hidden
        self.hidden_states_pi = np.zeros((S, L, E, H), dtype=np.float32)
        self.cell_states_pi = np.zeros((S, L, E, H), dtype=np.float32)
        self.hidden_states_vf = np.zeros((S, L, E, H), dtype=np.float32)
        self.cell_states_vf = np.zeros((S, L, E, H), dtype=np.float32)

    def add(self, obs, action, reward, value, log_prob, episode_start,
            lstm_pi, lstm_vf):
        """Store one timestep of data."""
        i = self.pos
        self.observations[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.episode_starts[i] = episode_start
        # LSTM states: (n_layers, n_envs, H) → store at position i
        self.hidden_states_pi[i] = lstm_pi[0].cpu().numpy()
        self.cell_states_pi[i] = lstm_pi[1].cpu().numpy()
        self.hidden_states_vf[i] = lstm_vf[0].cpu().numpy()
        self.cell_states_vf[i] = lstm_vf[1].cpu().numpy()
        self.pos += 1
        if self.pos == self.n_steps:
            self.full = True

    def compute_gae(self, last_values: np.ndarray, last_dones: np.ndarray):
        """GAE-lambda computation. Identical to SB3."""
        last_gae = 0.0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae
        self.returns = self.advantages + self.values

    @staticmethod
    def _swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """(n_steps, n_envs, ...) → (n_envs * n_steps, ...)"""
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        flat = arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
        return flat

    def get(self, batch_size: int):
        """Generator yielding sequence-padded RecurrentBatch minibatches."""
        assert self.full

        # Swap LSTM axes: (S, n_layers, E, H) → (S, E, n_layers, H) for swap_and_flatten
        h_pi = self.hidden_states_pi.swapaxes(1, 2)
        c_pi = self.cell_states_pi.swapaxes(1, 2)
        h_vf = self.hidden_states_vf.swapaxes(1, 2)
        c_vf = self.cell_states_vf.swapaxes(1, 2)

        # swap_and_flatten everything to (n_envs * n_steps, ...)
        flat = {}
        for name, arr in [("observations", self.observations),
                          ("actions", self.actions),
                          ("values", self.values),
                          ("log_probs", self.log_probs),
                          ("advantages", self.advantages),
                          ("returns", self.returns),
                          ("episode_starts", self.episode_starts),
                          ("h_pi", h_pi), ("c_pi", c_pi),
                          ("h_vf", h_vf), ("c_vf", c_vf)]:
            flat[name] = self._swap_and_flatten(arr)

        total = self.n_steps * self.n_envs

        # env_change: first step of each env in the flattened buffer
        env_change = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        env_change[0, :] = 1.0
        env_change = self._swap_and_flatten(env_change)

        # Mild shuffle: split and rotate
        split = np.random.randint(total)
        indices = np.arange(total)
        indices = np.concatenate((indices[split:], indices[:split]))

        start = 0
        while start < total:
            batch_inds = indices[start: start + batch_size]
            yield self._build_batch(batch_inds, flat, env_change)
            start += batch_size

    def _build_batch(self, batch_inds, flat, env_change) -> RecurrentBatch:
        """Build a sequence-padded minibatch from flat buffer indices."""
        device = self.device
        ep_starts = flat["episode_starts"][batch_inds]
        ec = env_change[batch_inds]

        # Sequence boundaries
        seq_start = np.logical_or(ep_starts, ec).flatten()
        seq_start[0] = True
        seq_start_indices = np.where(seq_start)[0]
        seq_end_indices = np.concatenate([
            (seq_start_indices - 1)[1:],
            np.array([len(batch_inds) - 1])
        ])

        n_seq = len(seq_start_indices)

        def _pad(arr, pad_val=0.0):
            seqs = [torch.tensor(arr[s:e+1], device=device)
                    for s, e in zip(seq_start_indices, seq_end_indices)]
            return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True,
                                                    padding_value=pad_val)

        def _pad_flat(arr, pad_val=0.0):
            return _pad(arr, pad_val).flatten()

        max_length = _pad(flat["actions"][batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length

        # Extract LSTM states at sequence starts and reshape to (n_layers, n_seq, H)
        def _get_lstm_start(h_key, c_key):
            h = flat[h_key][batch_inds][seq_start_indices]  # (n_seq, n_layers, H)
            c = flat[c_key][batch_inds][seq_start_indices]
            h = torch.tensor(h, device=device, dtype=torch.float32).swapaxes(0, 1).contiguous()
            c = torch.tensor(c, device=device, dtype=torch.float32).swapaxes(0, 1).contiguous()
            return (h, c)

        obs_padded = _pad(flat["observations"][batch_inds]).reshape(
            padded_batch_size, self.obs_dim)

        return RecurrentBatch(
            observations=obs_padded,
            actions=_pad_flat(flat["actions"][batch_inds]).long(),
            old_values=_pad_flat(flat["values"][batch_inds]),
            old_log_probs=_pad_flat(flat["log_probs"][batch_inds]),
            advantages=_pad_flat(flat["advantages"][batch_inds]),
            returns=_pad_flat(flat["returns"][batch_inds]),
            episode_starts=_pad_flat(flat["episode_starts"][batch_inds]),
            mask=_pad_flat(np.ones_like(flat["returns"][batch_inds])),
            lstm_states_pi=_get_lstm_start("h_pi", "c_pi"),
            lstm_states_vf=_get_lstm_start("h_vf", "c_vf"),
        )


# ══════════════════════════════════════════════════════════════════════════════
# REWARD NORMALIZER (replaces VecNormalize norm_reward=True)
# ══════════════════════════════════════════════════════════════════════════════

class RewardNormalizer:
    """Running reward normalization via discounted returns (Welford online)."""

    def __init__(self, n_envs: int, gamma: float = 0.995,
                 clip: float = 10.0, epsilon: float = 1e-8):
        self.n_envs = n_envs
        self.gamma = gamma
        self.clip = clip
        self.epsilon = epsilon
        self.returns = np.zeros(n_envs, dtype=np.float64)
        # Running mean/var of returns (Welford)
        self.mean = np.float64(0.0)
        self.var = np.float64(1.0)
        self.count = epsilon

    def normalize(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        self.returns = self.returns * self.gamma + rewards
        # Update running stats
        batch_mean = np.mean(self.returns)
        batch_var = np.var(self.returns)
        batch_count = self.n_envs
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = m_2 / tot_count
        self.count = tot_count
        # Reset returns for done envs
        self.returns[dones] = 0.0
        # Normalize
        normed = rewards / np.sqrt(self.var + self.epsilon)
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# CVaR CONTROLLER (replaces CVaRCallback)
# ══════════════════════════════════════════════════════════════════════════════

class CVaRController:
    """Lagrangian CVaR dual variable maintenance."""

    def __init__(self, alpha=0.05, cvar_budget=-1e-3, lambda_lr=0.01,
                 lambda_max=10.0, buffer_episodes=100, update_freq=10):
        self.alpha = alpha
        self.cvar_budget = cvar_budget
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max
        self.update_freq = update_freq
        self.returns_buffer = deque(maxlen=buffer_episodes)
        self.lam = 0.0
        self.var_alpha = 0.0
        self.n_episodes = 0
        self.cvar_alpha = 0.0

    def on_episode_end(self, avg_step_reward: float):
        self.returns_buffer.append(avg_step_reward)
        self.n_episodes += 1

    def maybe_update(self, envs):
        """Check if it's time to update and broadcast."""
        min_episodes = max(10, int(1.0 / self.alpha))
        if self.n_episodes % self.update_freq != 0:
            return
        if len(self.returns_buffer) < min_episodes:
            return

        returns = np.array(self.returns_buffer)
        self.var_alpha = float(np.percentile(returns, self.alpha * 100))
        tail = returns[returns <= self.var_alpha]
        self.cvar_alpha = float(tail.mean()) if len(tail) > 0 else self.var_alpha

        violation = max(0.0, self.cvar_budget - self.cvar_alpha)
        self.lam = float(np.clip(self.lam + self.lambda_lr * violation,
                                  0.0, self.lambda_max))

        # Broadcast to all envs
        try:
            envs.call("set_cvar_params", self.lam, self.var_alpha)
        except Exception:
            # SubprocVecEnv or single env — try alternative
            try:
                for i in range(envs.num_envs):
                    envs.envs[i].set_cvar_params(self.lam, self.var_alpha)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    path = Path(cfg["data"]["cache_path"])
    df = pd.read_parquet(path)
    frac_cols = [c for c in df.columns if "fractal" in c]
    if not frac_cols:
        print("   Computing fractal features...")
        from utils.fractal_features import compute_fractal_zone_features
        ohlcv = df[["open", "high", "low", "close"]].copy()
        frac_df = compute_fractal_zone_features(ohlcv)
        df = df.join(frac_df, how="left")
        frac_cols = list(frac_df.columns)
        df[frac_cols] = df[frac_cols].fillna(0.0)
        print(f"   Fractal features added: {frac_cols}")
    else:
        print(f"   Fractal features cached: {frac_cols}")

    if "turbulence_feature" not in df.columns:
        print("   Computing turbulence index...")
        _turb_candidates = [
            "log_return_1h_feature", "log_return_feature",
            "log_return_24h_feature",
            "volume_z_feature", "volume_zscore_24_feature",
            "volatility_20_feature", "volatility_24_feature",
            "realized_vol_1h_feature",
        ]
        _turb_cols = [c for c in _turb_candidates if c in df.columns]
        if _turb_cols:
            df = add_turbulence_feature(df, return_cols=_turb_cols, window=252, min_periods=60)
            print(f"   Turbulence: mean={df['turbulence_feature'].mean():.3f}  "
                  f"p99={df['turbulence_feature'].quantile(0.99):.3f}  "
                  f"cols={_turb_cols}")
        else:
            df["turbulence_feature"] = 0.0
            print("   [WARN] No return cols found — turbulence set to 0")
    else:
        print("   Turbulence feature cached.")

    _cross_cols = [c for c in df.columns if any(k in c for k in ("eth_btc", "eth_rel", "eth_vol_ratio", "btc_dominance"))]
    if not _cross_cols:
        _eth_path = cfg.get("data", {}).get("eth_path", "")
        if _eth_path:
            try:
                from utils.cross_asset_features import add_cross_asset_features, load_eth_df
                _eth_df = load_eth_df(_eth_path)
                if not _eth_df.empty:
                    df = add_cross_asset_features(df, _eth_df)
                else:
                    print("   Cross-asset features skipped (ETH data not found)")
            except Exception as _e:
                print(f"   [WARN] Cross-asset features skipped: {_e}")
        else:
            print("   Cross-asset features skipped (eth_path not configured)")
    else:
        print(f"   Cross-asset features cached ({len(_cross_cols)} cols)")

    if "slippage_bps_feature" not in df.columns:
        try:
            from utils.slippage import add_slippage_feature
            order_usd = cfg.get("env", {}).get("initial_balance", 1_000.0)
            df = add_slippage_feature(df, order_usd=order_usd, adv_window=24, vol_window=24)
        except Exception as _e:
            print(f"   [WARN] Dynamic slippage skipped: {_e}")
    else:
        print("   Slippage feature cached.")

    feat_cols = [c for c in df.columns if "feature" in c]
    print(f"⚡ Data loaded: {len(df)} bars · {len(feat_cols)} features")
    return df


def split_data(df, cfg):
    d = cfg["data"]
    train = df[d["train_start"]:d["train_end"]].copy()
    val   = df[d["val_start"]:d["val_end"]].copy()
    test  = df[d["test_start"]:].copy()
    return train, val, test


def build_regime_balanced_val(df, cfg, regime_col="trend_return_180_feature"):
    d = cfg["data"]
    oos = df[d["val_start"]:].copy()
    trend = oos[regime_col]
    thresholds = np.percentile(trend.dropna(), [33, 67])
    oos["_regime"] = np.digitize(trend.values, thresholds)
    min_block = 180
    target_bars = 540
    segments = []
    for regime_id, regime_name in [(0, "bear"), (1, "sideways"), (2, "bull")]:
        mask = oos["_regime"] == regime_id
        runs = []
        start = None
        for i, (idx, is_regime) in enumerate(mask.items()):
            if is_regime:
                if start is None:
                    start = idx
            else:
                if start is not None:
                    run_len = len(oos.loc[start:idx]) - 1
                    if run_len >= min_block:
                        runs.append((start, idx, run_len))
                    start = None
        if start is not None:
            run_len = len(oos.loc[start:])
            if run_len >= min_block:
                runs.append((start, oos.index[-1], run_len))
        if not runs:
            chunk = oos[mask].head(target_bars)
            if len(chunk) > 0:
                segments.append(chunk)
                print(f"  [RegimeVal] {regime_name}: {len(chunk)} bars (scattered)")
            continue
        best = max(runs, key=lambda x: x[2])
        seg = oos.loc[best[0]:best[1]].head(target_bars)
        segments.append(seg)
        print(f"  [RegimeVal] {regime_name}: {len(seg)} bars "
              f"({seg.index[0].date()} → {seg.index[-1].date()})")
    if not segments:
        print("  [RegimeVal] WARNING: falling back to chronological val")
        return df[d["val_start"]:d["val_end"]].copy()
    val_df = pd.concat(segments, axis=0)
    val_df = val_df.drop(columns=["_regime"], errors="ignore")
    print(f"  [RegimeVal] Total: {len(val_df)} bars across {len(segments)} regime segments")
    return val_df


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

def _build_single_env(df, cfg, training, name, rank=0):
    env_cfg = cfg["env"]
    reward_cfg = cfg["reward"]
    risk_cfg = cfg.get("risk", {})

    fee = env_cfg["training_fee_rate"] if training else env_cfg["fee_rate"]
    ep_dur = env_cfg.get("train_episode_bars", 720) if training else "max"

    reward_type = reward_cfg.get("type", "fractal")
    if reward_type == "sign":
        reward_fn = SignReward(
            trade_penalty=reward_cfg.get("trade_penalty", 0.0),
        )
    elif reward_type == "dsr_fractal":
        reward_fn = DSRFractalReward(
            alignment_scale=reward_cfg["fractal_alignment_scale"],
            trade_penalty=reward_cfg["trade_penalty"],
            eta=reward_cfg.get("eta", 0.01),
            dsr_a_init=reward_cfg.get("dsr_a_init", 0.0),
            dsr_b_init=reward_cfg.get("dsr_b_init", 0.0),
            inactivity_penalty=reward_cfg.get("inactivity_penalty", 0.0),
            inactivity_window=reward_cfg.get("inactivity_window", 12),
            demean_market=reward_cfg.get("demean_market", True),
            conviction_bonus=reward_cfg.get("conviction_bonus", 0.0),
        )
    else:
        reward_fn = FractalGuidedReward(
            alignment_scale=reward_cfg["fractal_alignment_scale"],
            trade_penalty=reward_cfg["trade_penalty"],
        )

    env = TradingEnv(
        df, positions=list(env_cfg["positions"]),
        trading_fees=0.0, fee_rate=fee,
        slippage_bps=env_cfg.get("slippage_bps", 2),
        portfolio_initial_value=env_cfg.get("initial_balance", 1_000.0),
        initial_position='random' if training else env_cfg["positions"][0],
        windows=env_cfg.get("window_size"),
        max_episode_duration=ep_dur,
        verbose=0, name=f"{name}_{rank}",
        reward_function=reward_fn,
    )
    env = DiscretizeActionWrapper(env, positions=list(env_cfg["positions"]))

    # Market inversion: 50% of training episodes flip directional features + reward
    if training and cfg.get("inversion", {}).get("enabled", False):
        feat_cols = [c for c in df.columns if "feature" in c]
        env = InvertedMarketWrapper(env, feature_columns=feat_cols,
                                     p_invert=cfg["inversion"].get("p_invert", 0.5))

    apply = (not training and risk_cfg.get("apply_in_eval", True)) or \
            (training and risk_cfg.get("apply_in_training", False))
    if risk_cfg and apply:
        env = MultiLevelRiskWrapper(
            env, dd_hard=risk_cfg.get("dd_hard", 0.15) * (1.5 if training else 1.0),
            turbulence_col=risk_cfg.get("turbulence_col", "turbulence_feature"),
            turbulence_threshold=risk_cfg.get("turbulence_threshold", 1.5),
            atr_col=risk_cfg.get("atr_col", "feature_atr"),
            atr_stop_mult=risk_cfg.get("atr_stop_mult", 2.0),
            cooldown_steps=risk_cfg.get("cooldown_steps", 24),
            apply_turbulence=not training, apply_trailing_stop=not training,
        )

    cvar_cfg = cfg.get("cvar", {})
    cvar_apply = (training and cvar_cfg.get("apply_in_training", True)) or \
                 (not training and cvar_cfg.get("apply_in_eval", False))
    if cvar_cfg.get("enabled", False) and cvar_apply:
        env = CVaRConstraintWrapper(
            env, alpha=cvar_cfg.get("alpha", 0.05),
            cvar_budget=cvar_cfg.get("cvar_budget", -1e-3),
        )

    if training:
        env = RegimeBalancedWrapper(env, regime_col="trend_return_180_feature", n_bins=3)

    return env


def make_train_envs(df, cfg):
    """Build SubprocVecEnv-like training environments using multiprocessing."""
    n = cfg["training"]["n_envs"]
    tmp = Path("cache/temp/_train_tmp.parquet")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp)

    def make_fn(rank):
        def _init():
            _df = pd.read_parquet(tmp)
            return _build_single_env(_df, cfg, training=True, name="train", rank=rank)
        return _init

    # Use stable_baselines3 SubprocVecEnv for subprocess-based parallelism
    # This is the ONLY remaining SB3 import — for the multiprocessing wrapper
    from stable_baselines3.common.vec_env import SubprocVecEnv
    return SubprocVecEnv([make_fn(i) for i in range(n)])


def make_eval_env(df, cfg):
    """Single evaluation environment (no vectorization)."""
    return _build_single_env(df, cfg, training=False, name="val")


# ══════════════════════════════════════════════════════════════════════════════
# METRICS & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _sharpe(nav):
    if len(nav) < 2: return -10.0
    r = np.diff(np.log(np.clip(nav, 1e-8, None)))
    r = np.clip(r, -0.5, 0.5)
    mu, sigma = r.mean(), r.std()
    if sigma < 1e-8: return 10.0 if mu > 0 else -10.0
    return float(np.clip((mu / sigma) * np.sqrt(8_760), -10, 10))


def _sortino(nav):
    if len(nav) < 2: return 0.0
    r = np.diff(np.log(np.clip(nav, 1e-8, None)))
    r = np.clip(r, -0.5, 0.5)
    mu = r.mean()
    down = r[r < 0]
    dd_std = down.std() if len(down) > 1 else 1e-8
    if dd_std < 1e-8: return 10.0 if mu > 0 else -10.0
    return float(np.clip((mu / dd_std) * np.sqrt(8_760), -10, 10))


def _calmar(nav):
    if len(nav) < 2: return 0.0
    total_ret = nav[-1] / nav[0] - 1
    peaks = np.maximum.accumulate(nav)
    max_dd = float(np.max(1 - nav / peaks))
    if max_dd < 1e-8: return 10.0 if total_ret > 0 else 0.0
    ann_factor = 8_760 / max(len(nav), 1)
    return float(np.clip((total_ret * ann_factor) / max_dd, -10, 10))


def _max_dd_duration(nav):
    if len(nav) < 2: return 0
    peaks = np.maximum.accumulate(nav)
    in_dd = nav < peaks
    max_dur = cur = 0
    for d in in_dd:
        if d: cur += 1; max_dur = max(max_dur, cur)
        else: cur = 0
    return max_dur


@torch.no_grad()
def evaluate(model: RecurrentActorCritic, env, device: torch.device,
             max_steps: int = 0) -> Dict[str, float]:
    """Run deterministic evaluation on a single (non-vectorized) environment."""
    model.eval()
    obs, info = env.reset()
    if max_steps <= 0:
        try:
            max_steps = len(env.unwrapped.df) - 1
        except Exception:
            max_steps = 10_000

    n_layers = model.n_lstm_layers
    H = model.lstm_hidden
    lstm_pi = (torch.zeros(n_layers, 1, H, device=device),
               torch.zeros(n_layers, 1, H, device=device))
    lstm_vf = (torch.zeros(n_layers, 1, H, device=device),
               torch.zeros(n_layers, 1, H, device=device))
    episode_starts = torch.ones(1, device=device)

    pv, prices, actions_list = [1_000.0], [], []

    for _ in range(max_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, _, _, _, (lstm_pi, lstm_vf) = model.get_action_and_value(
            obs_t, lstm_pi, lstm_vf, episode_starts, deterministic=True
        )
        episode_starts = torch.zeros(1, device=device)

        act_int = action.item()
        obs, reward, terminated, truncated, info = env.step(act_int)

        if "portfolio_valuation" in info:
            pv.append(float(info["portfolio_valuation"]))
        for key in ("data_close", "close"):
            if key in info:
                prices.append(float(info[key]))
                break

        actions_list.append(act_int)

        if terminated or truncated:
            break

    model.train()

    nav = np.array(pv)
    ret = float(nav[-1] / nav[0] - 1) * 100 if len(nav) > 1 else 0.0
    mkt = float(prices[-1] / prices[0] - 1) * 100 if len(prices) > 1 else 0.0
    dd_dur = _max_dd_duration(nav)
    peaks = np.maximum.accumulate(nav)
    dd = float(np.max(1 - nav / peaks)) * 100 if len(nav) > 1 else 0.0

    act = np.array(actions_list)
    # Dynamic action mapping: walk wrapper chain to find DiscretizeActionWrapper
    _env = env
    _positions = np.array([-1.0, 0.0, 1.0])  # fallback
    while _env is not None:
        if hasattr(_env, '_positions'):
            _positions = _env._positions
            break
        _env = getattr(_env, 'env', None)
    n_short = int(np.sum(act == np.searchsorted(_positions, -1.0))) if -1.0 in _positions else 0
    n_hold  = int(np.sum(act == np.searchsorted(_positions, 0.0))) if 0.0 in _positions else 0
    n_long  = int(np.sum(act == np.searchsorted(_positions, 1.0))) if 1.0 in _positions else 0
    n_trades = int(np.sum(np.abs(np.diff(act)) > 0)) if len(act) > 1 else 0

    trade_boundaries = np.where(np.abs(np.diff(act)) > 0)[0] if len(act) > 1 else []
    wins = 0
    for i in range(len(trade_boundaries) - 1):
        s, e = trade_boundaries[i], trade_boundaries[i + 1]
        if e < len(nav) and s < len(nav) and nav[e] > nav[s]:
            wins += 1
    win_rate = wins / max(1, len(trade_boundaries) - 1) * 100

    return {
        "sharpe": _sharpe(nav), "sortino": _sortino(nav), "calmar": _calmar(nav),
        "return": ret, "market": mkt, "alpha": ret - mkt,
        "dd": dd, "dd_duration": dd_dur,
        "trades": n_trades, "win_rate": win_rate,
        "pct_long": n_long / max(1, len(act)) * 100,
        "pct_short": n_short / max(1, len(act)) * 100,
        "pct_hold": n_hold / max(1, len(act)) * 100,
    }


# ══════════════════════════════════════════════════════════════════════════════
# COLLECT ROLLOUTS
# ══════════════════════════════════════════════════════════════════════════════

def collect_rollouts(model, envs, buffer, reward_norm, cvar_ctrl, device,
                     reward_scale=1.0):
    """Fill the rollout buffer with n_steps of experience from n_envs."""
    model.eval()
    n_steps = buffer.n_steps
    n_envs = buffer.n_envs
    n_layers = model.n_lstm_layers
    H = model.lstm_hidden

    # Retrieve persistent state from envs wrapper
    obs = envs._last_obs
    episode_starts = envs._last_episode_starts
    lstm_pi = envs._lstm_pi
    lstm_vf = envs._lstm_vf

    for step in range(n_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        ep_starts_t = torch.tensor(episode_starts, dtype=torch.float32, device=device)

        with torch.no_grad():
            action, log_prob, _, value, (new_lstm_pi, new_lstm_vf) = \
                model.get_action_and_value(obs_t, lstm_pi, lstm_vf, ep_starts_t)

        actions_np = action.cpu().numpy()
        values_np = value.cpu().numpy()
        log_probs_np = log_prob.cpu().numpy()

        # Store BEFORE stepping (pre-action obs and LSTM states)
        buffer.add(obs, actions_np, np.zeros(n_envs), values_np, log_probs_np,
                   episode_starts, lstm_pi, lstm_vf)

        # Step environments
        new_obs, rewards, dones, infos = envs.step(actions_np)

        # Track CVaR episode completions
        if cvar_ctrl is not None:
            for i, info in enumerate(infos):
                if "episode" in info:
                    ep_r = info["episode"]["r"]
                    ep_l = max(info["episode"]["l"], 1)
                    cvar_ctrl.on_episode_end(ep_r / ep_l)

        # Normalize rewards
        normed_rewards = reward_norm.normalize(rewards.astype(np.float64),
                                                dones.astype(bool))
        # Scale post-normalization to amplify signal for critic/advantage
        if reward_scale != 1.0:
            normed_rewards = normed_rewards * reward_scale
        # Update the stored reward (we stored 0 above, now fill in)
        buffer.rewards[step] = normed_rewards

        # Reset LSTM states for done envs
        done_mask = torch.tensor(dones, dtype=torch.float32, device=device)
        lstm_pi = (
            new_lstm_pi[0] * (1.0 - done_mask).view(1, n_envs, 1),
            new_lstm_pi[1] * (1.0 - done_mask).view(1, n_envs, 1),
        )
        lstm_vf = (
            new_lstm_vf[0] * (1.0 - done_mask).view(1, n_envs, 1),
            new_lstm_vf[1] * (1.0 - done_mask).view(1, n_envs, 1),
        )

        obs = new_obs
        episode_starts = dones.astype(np.float32)

    # Bootstrap last values for GAE
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        ep_starts_t = torch.tensor(episode_starts, dtype=torch.float32, device=device)
        last_values, _ = model.get_value(obs_t, lstm_vf, ep_starts_t)
    last_values_np = last_values.cpu().numpy()

    buffer.compute_gae(last_values_np, episode_starts)

    # Persist state for next rollout
    envs._last_obs = obs
    envs._last_episode_starts = episode_starts
    envs._lstm_pi = lstm_pi
    envs._lstm_vf = lstm_vf

    model.train()


# ══════════════════════════════════════════════════════════════════════════════
# PPO UPDATE
# ══════════════════════════════════════════════════════════════════════════════

def ppo_update(model, buffer, actor_opt, critic_opt, cfg, global_step, writer=None):
    """One PPO update: n_epochs over the buffer with sequence-padded minibatches."""
    mp = cfg["model"]
    burnin_cfg = cfg["burnin"]
    burnin_steps = burnin_cfg["burnin_steps"]
    n_epochs = mp["n_epochs"]
    batch_size = mp["batch_size"]
    clip_range = mp["clip_range"]
    ent_coef = mp["ent_coef"]
    entropy_floor_frac = mp.get("entropy_floor", 0.0)
    vf_coef = mp["vf_coef"]
    max_grad_norm = mp["max_grad_norm"]
    target_kl = mp["target_kl"]

    in_burnin = global_step < burnin_steps

    # Logging accumulators
    pg_losses, vf_losses, ent_losses, kl_divs = [], [], [], []
    clip_fracs, lb_losses, raw_adv_stds = [], [], []

    for epoch in range(n_epochs):
        kl_exceeded = False

        for batch in buffer.get(batch_size):
            mask = batch.mask.bool()
            if mask.sum() == 0:
                continue

            _, new_log_prob, entropy, new_value, _ = model.get_action_and_value(
                batch.observations, batch.lstm_states_pi, batch.lstm_states_vf,
                batch.episode_starts, action=batch.actions,
            )

            # Advantage processing
            adv = batch.advantages.clone()
            raw_adv_std = adv[mask].std().item()
            if mp.get("normalize_advantages", True):
                adv[mask] = (adv[mask] - adv[mask].mean()) / (adv[mask].std() + 1e-8)

            # Policy loss (clipped surrogate)
            log_ratio = new_log_prob - batch.old_log_probs
            ratio = torch.exp(log_ratio)
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = torch.max(pg_loss1, pg_loss2)[mask].mean()

            # Value loss (Huber / smooth L1)
            value_loss = F.smooth_l1_loss(new_value[mask], batch.returns[mask])

            # Entropy with floor: boost ent_coef when entropy drops below floor
            mean_entropy = entropy[mask].mean()
            entropy_loss = -mean_entropy
            effective_ent_coef = ent_coef
            if entropy_floor_frac > 0 and not in_burnin:
                n_actions = model.n_actions
                max_ent = math.log(n_actions)
                ent_floor = entropy_floor_frac * max_ent
                if mean_entropy.item() < ent_floor:
                    # 10x boost when below floor to push entropy back up
                    effective_ent_coef = ent_coef * 10.0

            # MoE load balance loss
            lb_loss = model.pi_extractor.get_load_balance_loss() \
                if hasattr(model.pi_extractor, 'get_load_balance_loss') else torch.tensor(0.0)

            # KL divergence check
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio)[mask].mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_range)[mask].float().mean().item()

            if not in_burnin and approx_kl > 1.5 * target_kl:
                kl_exceeded = True
                break

            if in_burnin:
                # CRITIC ONLY — 4 lines that replace the entire CriticBurnInCallback
                critic_opt.zero_grad()
                (vf_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(model.critic_parameters, max_grad_norm)
                critic_opt.step()
            else:
                # FULL UPDATE — both actor and critic
                total_loss = policy_loss + vf_coef * value_loss + effective_ent_coef * entropy_loss + lb_loss
                actor_opt.zero_grad()
                critic_opt.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(list(model.parameters()), max_grad_norm)
                actor_opt.step()
                critic_opt.step()

            # Accumulate metrics
            pg_losses.append(policy_loss.item())
            vf_losses.append(value_loss.item())
            ent_losses.append(entropy_loss.item())
            kl_divs.append(approx_kl)
            clip_fracs.append(clip_frac)
            lb_losses.append(lb_loss.item())
            raw_adv_stds.append(raw_adv_std)

        if kl_exceeded:
            break

    # Explained variance
    y_pred = buffer.values.flatten()
    y_true = buffer.returns.flatten()
    var_y = np.var(y_true)
    ev = 1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 1e-8 else 0.0

    stats = {
        "policy_gradient_loss": np.mean(pg_losses) if pg_losses else 0.0,
        "value_loss": np.mean(vf_losses) if vf_losses else 0.0,
        "entropy_loss": np.mean(ent_losses) if ent_losses else 0.0,
        "approx_kl": np.mean(kl_divs) if kl_divs else 0.0,
        "clip_fraction": np.mean(clip_fracs) if clip_fracs else 0.0,
        "load_balance_loss": np.mean(lb_losses) if lb_losses else 0.0,
        "raw_advantage_std": np.mean(raw_adv_stds) if raw_adv_stds else 0.0,
        "explained_variance": ev,
        "n_updates": len(pg_losses),
    }

    if writer is not None:
        for k, v in stats.items():
            writer.add_scalar(f"train/{k}", v, global_step)

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main(seed: int = 42, resume_dir: str = None):
    cfg = copy.deepcopy(CONFIG)
    mp = cfg["model"]
    tp = cfg["training"]
    moe_cfg = cfg["moe"]
    cvar_cfg = cfg["cvar"]
    burnin_cfg = cfg["burnin"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Seed ──────────────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # ── Run directory ─────────────────────────────────────────────────────
    log_dir = Path(tp["log_dir"])
    log_dir.mkdir(exist_ok=True)
    existing = [d.name for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("Run_")]
    run_num = max([int(d.split("_")[1]) for d in existing], default=0) + 1
    run_dir = log_dir / f"Run_{run_num:03d}_s{seed}"
    run_dir.mkdir(parents=True)

    print("═" * 60)
    print(f"  RL TRADING  (train_cleanrl.py — CleanRL-style)")
    print(f"  Pair       : BTC_USDT  4h")
    print(f"  Positions  : {cfg['env']['positions']}")
    reward_mode = "active" if cfg["reward"].get("demean_market", False) else "raw"
    print(f"  Reward     : {cfg['reward']['type']}  (η={cfg['reward']['eta']}, {reward_mode})")
    print(f"  Fees train : {cfg['env']['training_fee_rate']*100:.2f}%"
          f"  eval: {cfg['env']['fee_rate']*100:.3f}%")
    print(f"  Envs×steps : {tp['n_envs']} × {tp['total_timesteps']//1_000_000}M")
    print(f"  Seed       : {seed}")
    print(f"  Run dir    : {run_dir}")
    print("═" * 60)

    # ── Data ──────────────────────────────────────────────────────────────
    df = load_data(cfg)
    train_df, val_df, test_df = split_data(df, cfg)
    regime_val_df = build_regime_balanced_val(df, cfg)

    print(f"  Train : {train_df.index[0].date()} → {train_df.index[-1].date()}"
          f"  ({len(train_df)} bars)")
    print(f"  Val   : {val_df.index[0].date()} → {val_df.index[-1].date()}"
          f"  ({len(val_df)} bars)")
    print(f"  Test  : {test_df.index[0].date()} → {test_df.index[-1].date()}"
          f"  ({len(test_df)} bars)")

    # DSR warm-start (only needed for DSR-based rewards)
    if cfg["reward"].get("type", "fractal") in ("dsr_fractal", "fractal"):
        close = train_df["close"].values
        log_ret = np.diff(np.log(close))
        if cfg["reward"].get("demean_market", False):
            cfg["reward"]["dsr_a_init"] = 0.0
            cfg["reward"]["dsr_b_init"] = float(max(np.var(log_ret), 1e-8))
        else:
            cfg["reward"]["dsr_a_init"] = float(log_ret.mean())
            cfg["reward"]["dsr_b_init"] = float((log_ret ** 2).mean())
        print(f"  DSR warm-start: A={cfg['reward']['dsr_a_init']:.6f}"
              f"  B={cfg['reward']['dsr_b_init']:.6f}")
    else:
        print(f"  Reward type: {cfg['reward']['type']} (no DSR warm-start needed)")

    # ── Environments ──────────────────────────────────────────────────────
    print("  Initializing SubprocVecEnv...")
    train_envs = make_train_envs(train_df, cfg)
    val_env = make_eval_env(regime_val_df, cfg)

    # Train-eval slice for generalization ratio
    train_eval_bars = min(2160, len(train_df))
    train_eval_df = train_df.iloc[-train_eval_bars:]
    train_eval_env = make_eval_env(train_eval_df, cfg)
    print(f"  Train-eval slice: {train_eval_df.index[0].date()} →"
          f" {train_eval_df.index[-1].date()}"
          f" ({len(train_eval_df)} bars) — for gen ratio")
    print("  Environments ready.\n")

    # ── Model ─────────────────────────────────────────────────────────────
    feature_cols = [c for c in train_df.columns if "feature" in c]
    obs_dim = len(feature_cols) + 4  # +4 for dynamic features
    # Verify obs_dim matches environment
    test_obs, _ = val_env.reset()
    obs_dim = test_obs.shape[-1] if len(test_obs.shape) > 0 else obs_dim

    n_actions = len(cfg["env"]["positions"])

    moe_kwargs = None
    if moe_cfg.get("enabled", False):
        regime_idx = get_regime_feature_indices(feature_cols)
        moe_kwargs = dict(
            n_experts=moe_cfg["n_experts"],
            expert_dim=moe_cfg["expert_dim"],
            gate_hidden_dim=moe_cfg["gate_hidden_dim"],
            regime_feature_idx=regime_idx,
            gate_entropy_coef=moe_cfg.get("gate_entropy_coef", 0.0),
            gate_temperature=moe_cfg.get("gate_temperature", 1.0),
            load_balance_alpha=moe_cfg.get("load_balance_alpha", 0.0),
        )
        print(f"  MoE enabled: {moe_cfg['n_experts']} experts"
              f"  | gate inputs: {len(regime_idx)} regime features")

    model = RecurrentActorCritic(
        obs_dim=obs_dim,
        n_actions=n_actions,
        lstm_hidden=mp["lstm_hidden_size"],
        n_lstm_layers=mp["n_lstm_layers"],
        pi_net_arch=[128, 64],
        vf_net_arch=[128, 64],
        moe_kwargs=moe_kwargs,
        critic_features_dim=moe_cfg.get("expert_dim", 64),
    ).to(device)

    n_actor = sum(p.numel() for p in model.actor_parameters)
    n_critic = sum(p.numel() for p in model.critic_parameters)
    print(f"  Model: actor {n_actor:,} params  |  critic {n_critic:,} params"
          f"  |  total {n_actor + n_critic:,}")

    # ── Optimizers ────────────────────────────────────────────────────────
    base_lr = mp["learning_rate"]
    actor_opt = torch.optim.Adam(model.actor_parameters, lr=base_lr, eps=1e-5)
    critic_opt = torch.optim.Adam(model.critic_parameters, lr=base_lr, eps=1e-5)

    # ── Buffer, reward normalizer, CVaR ───────────────────────────────────
    n_envs = tp["n_envs"]
    n_steps = mp["n_steps"]
    buffer = RecurrentRolloutBuffer(
        n_steps=n_steps, n_envs=n_envs, obs_dim=obs_dim,
        n_lstm_layers=mp["n_lstm_layers"], lstm_hidden=mp["lstm_hidden_size"],
        gamma=mp["gamma"], gae_lambda=mp["gae_lambda"], device=device,
    )
    reward_norm = RewardNormalizer(n_envs=n_envs, gamma=mp["gamma"])
    cvar_ctrl = CVaRController(**{k: v for k, v in cvar_cfg.items()
                                  if k not in ("enabled", "apply_in_training", "apply_in_eval")}) \
                if cvar_cfg.get("enabled", False) else None

    # ── TensorBoard ───────────────────────────────────────────────────────
    writer = SummaryWriter(str(run_dir / "tb"))

    # ── Init persistent env state ─────────────────────────────────────────
    obs = train_envs.reset()
    n_layers = mp["n_lstm_layers"]
    H = mp["lstm_hidden_size"]
    train_envs._last_obs = obs
    train_envs._last_episode_starts = np.ones(n_envs, dtype=np.float32)
    train_envs._lstm_pi = (torch.zeros(n_layers, n_envs, H, device=device),
                            torch.zeros(n_layers, n_envs, H, device=device))
    train_envs._lstm_vf = (torch.zeros(n_layers, n_envs, H, device=device),
                            torch.zeros(n_layers, n_envs, H, device=device))

    # ── Training state ────────────────────────────────────────────────────
    total_timesteps = tp["total_timesteps"]
    eval_freq = tp["eval_freq"]
    patience = tp["patience"]
    n_iterations = total_timesteps // (n_steps * n_envs)
    steps_per_iter = n_steps * n_envs

    burnin_steps = burnin_cfg["burnin_steps"]
    actor_lr_mult = burnin_cfg["actor_lr_mult"]

    best_sharpe = -np.inf
    no_improve = 0
    next_eval = eval_freq
    unfreeze_announced = False

    print(f"  [BurnIn] Critic-only training for {burnin_steps//1000}k steps")
    print(f"  [LR] Permanent asymmetric: actor={actor_lr_mult}x critic")
    if cvar_ctrl:
        print(f"  [CVaR] α={cvar_cfg['alpha']}  budget={cvar_cfg['cvar_budget']:.2e}"
              f"  λ_lr={cvar_cfg['lambda_lr']}")

    # ── Eval log header ───────────────────────────────────────────────────
    print("  ─── EVAL LOG ─────────────────────────────────────────────────────────")
    print("  │  Step  Sharpe  Sortino  Calmar    Return      BH"
          "      DD    Trades  Win%   Long%  Short%  GenRatio")
    print("  ──────────────────────────────────────────────────────────────────────")

    t_start = time.time()
    early_stopped = False

    for iteration in range(n_iterations):
        global_step = iteration * steps_per_iter

        # ── LR scheduling ─────────────────────────────────────────────────
        progress = 1.0 - global_step / total_timesteps
        current_lr = cosine_lr(progress, base=base_lr, min_lr=1e-6)

        for pg in critic_opt.param_groups:
            pg['lr'] = current_lr

        if global_step < burnin_steps:
            # Burn-in: actor optimizer not stepped, but set LR for logging
            for pg in actor_opt.param_groups:
                pg['lr'] = current_lr
        else:
            # Post burn-in: actor permanently at mult × critic LR
            for pg in actor_opt.param_groups:
                pg['lr'] = current_lr * actor_lr_mult
            if not unfreeze_announced:
                print(f"\n  [BurnIn] Actor UNFROZEN at {global_step//1000}k steps"
                      f" — permanent {actor_lr_mult}x LR → {current_lr * actor_lr_mult:.2e}")
                best_sharpe = -np.inf
                no_improve = 0
                unfreeze_announced = True

        # ── Collect rollouts ──────────────────────────────────────────────
        buffer.reset()
        collect_rollouts(model, train_envs, buffer, reward_norm, cvar_ctrl, device,
                         reward_scale=cfg["reward"].get("reward_scale", 1.0))

        # ── PPO update ────────────────────────────────────────────────────
        stats = ppo_update(model, buffer, actor_opt, critic_opt, cfg, global_step, writer)

        # ── CVaR update ───────────────────────────────────────────────────
        if cvar_ctrl:
            cvar_ctrl.maybe_update(train_envs)
            if cvar_ctrl.n_episodes > 0 and cvar_ctrl.n_episodes % 16 == 0:
                print(f"  [CVaR] ep={cvar_ctrl.n_episodes:5d}"
                      f"  VaR_α={cvar_ctrl.var_alpha:.5f}"
                      f"  CVaR_α={cvar_ctrl.cvar_alpha:.5f}"
                      f"  λ={cvar_ctrl.lam:.4f}"
                      f"  violation={max(0, cvar_ctrl.cvar_budget - cvar_ctrl.cvar_alpha):.5f}")

        # ── Logging ───────────────────────────────────────────────────────
        elapsed = time.time() - t_start
        fps = (global_step + steps_per_iter) / max(elapsed, 1)

        writer.add_scalar("time/fps", fps, global_step + steps_per_iter)
        writer.add_scalar("train/actor_lr", actor_opt.param_groups[0]["lr"],
                          global_step + steps_per_iter)
        writer.add_scalar("train/critic_lr", critic_opt.param_groups[0]["lr"],
                          global_step + steps_per_iter)
        writer.add_scalar("rollout/ep_rew_mean",
                          buffer.rewards.sum(axis=0).mean(), global_step + steps_per_iter)

        # ── MoE gate monitoring ───────────────────────────────────────────
        gate_log_freq = moe_cfg.get("gate_log_freq", 100_000)
        if moe_kwargs and (global_step + steps_per_iter) % gate_log_freq < steps_per_iter:
            gw = model.pi_extractor._last_gate_weights
            if gw is not None:
                gw_mean = gw.mean(dim=0).detach().cpu().numpy()
                ent = -(gw * torch.log(gw + 1e-8)).sum(-1).mean().item()
                max_ent = math.log(moe_cfg["n_experts"])
                parts = "  ".join(f"E{i}={w:.3f}" for i, w in enumerate(gw_mean))
                step_k = (global_step + steps_per_iter) // 1000
                print(f"  [MoE {step_k}k]  {parts}  entropy={ent:.3f}/{max_ent:.3f}")
                for i, w in enumerate(gw_mean):
                    writer.add_scalar(f"moe/expert_{i}_mean_weight", w,
                                       global_step + steps_per_iter)
                writer.add_scalar("moe/gate_entropy", ent, global_step + steps_per_iter)

        # ── Evaluation ────────────────────────────────────────────────────
        step_now = global_step + steps_per_iter
        if step_now >= next_eval:
            next_eval += eval_freq

            m = evaluate(model, val_env, device)

            # Generalization ratio
            gen_ratio = float('nan')
            try:
                m_train = evaluate(model, train_eval_env, device)
                train_ret = m_train["return"]
                gen_ratio = m["return"] / max(abs(train_ret), 1.0)
            except Exception:
                pass

            step_k = f"{step_now // 1000}k"
            print(f"  │ {step_k:>5s}  Sharpe {m['sharpe']:+.3f}"
                  f"  Sort {m['sortino']:+.3f}  Calm {m['calmar']:+.3f}"
                  f"  Ret {m['return']:+7.2f}%  BH {m['market']:+7.2f}%"
                  f"  DD {m['dd']:5.1f}% ({m['dd_duration']}b)"
                  f"  T {m['trades']:4d}  W {m['win_rate']:4.1f}%"
                  f"  L {m['pct_long']:4.1f}%  S {m['pct_short']:4.1f}%"
                  f"  GenR {gen_ratio:+.2f}")

            for k, v in m.items():
                writer.add_scalar(f"eval/{k}", v, step_now)
            writer.add_scalar("eval/gen_ratio", gen_ratio if not math.isnan(gen_ratio) else 0, step_now)

            # Checkpoint — save best absolute Sharpe (no gen_ratio gate)
            improved = m["sharpe"] > best_sharpe
            if improved:
                best_sharpe = m["sharpe"]
                no_improve = 0
                torch.save(model.state_dict(), run_dir / "best_model.pt")
                with open(run_dir / "metrics.json", "w") as f:
                    json.dump({**m, "gen_ratio": gen_ratio, "step": step_now, "seed": seed}, f, indent=2)
            else:
                no_improve += eval_freq
                if no_improve >= patience:
                    print(f"\n  Early stop at {step_now:,} — "
                          f"no improvement for {patience//1000}k steps")
                    early_stopped = True
                    break

        # Print iteration stats periodically
        if iteration % 2 == 0:
            phase = "burn-in" if global_step < burnin_steps else "active"
            sys.stdout.write(
                f"\r  [{phase}] step={step_now//1000}k  fps={fps:.0f}"
                f"  pg_loss={stats['policy_gradient_loss']:.2e}"
                f"  vf_loss={stats['value_loss']:.4f}"
                f"  entropy={stats['entropy_loss']:.3f}"
                f"  kl={stats['approx_kl']:.4f}"
                f"  ev={stats['explained_variance']:.4f}"
                f"  adv_std={stats['raw_advantage_std']:.2e}"
                f"  actor_lr={actor_opt.param_groups[0]['lr']:.2e}  "
            )
            sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════════════
    # BEST CHECKPOINT SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n")
    best_path = run_dir / "best_model.pt"
    if best_path.exists():
        try:
            with open(run_dir / "metrics.json") as f:
                best_m = json.load(f)
            print("═" * 60)
            step_str = f"{best_m['step']//1000}k" if 'step' in best_m else '?'
            print(f"  Best val checkpoint : step {step_str}")
            print(f"    Sharpe      : {best_m['sharpe']:+.3f}")
            print(f"    Sortino     : {best_m['sortino']:+.3f}")
            print(f"    Calmar      : {best_m['calmar']:+.3f}")
            print(f"    Return      : {best_m['return']:+.2f}%"
                  f"  (BH {best_m['market']:+.2f}%)")
            print(f"    Alpha       : {best_m['alpha']:+.2f}%")
            print(f"    DD          : {best_m['dd']:.2f}%"
                  f"  (duration: {best_m['dd_duration']} bars)")
            print(f"    Trades      : {best_m['trades']}"
                  f"  (win rate: {best_m['win_rate']:.1f}%)")
            print(f"    Gen ratio   : {best_m.get('gen_ratio', float('nan')):.2f}")
        except Exception:
            pass

        # ── Test evaluation ───────────────────────────────────────────────
        try:
            print("\n  ─── TEST SET EVALUATION ───────────────────────────────────────────")
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
            test_env = make_eval_env(test_df, cfg)
            tm = evaluate(model, test_env, device)
            print(f"    Sharpe  : {tm['sharpe']:+.3f}")
            print(f"    Sortino : {tm['sortino']:+.3f}")
            print(f"    Calmar  : {tm['calmar']:+.3f}")
            print(f"    Return  : {tm['return']:+.2f}%"
                  f"  (BH {tm['market']:+.2f}%)")
            print(f"    Alpha   : {tm['alpha']:+.2f}%")
            print(f"    DD      : {tm['dd']:.2f}%"
                  f"  (duration: {tm['dd_duration']} bars)")
            print(f"    Trades  : {tm['trades']}"
                  f"  ({tm['pct_long']:.1f}% Long,"
                  f" {tm['pct_short']:.1f}% Short,"
                  f" win {tm['win_rate']:.1f}%)")
            test_env.close()
        except Exception:
            traceback.print_exc()

    print(f"\n  Saved to: {run_dir}")

    # Cleanup
    train_envs.close()
    val_env.close()
    train_eval_env.close()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, metavar="RUN_DIR")
    args = parser.parse_args()
    main(seed=args.seed, resume_dir=args.resume)
