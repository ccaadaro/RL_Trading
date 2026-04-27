#!/usr/bin/env python3
"""
train2.py  —  Clean RL training pipeline for BTC/USDT 4h
Key design decisions:
  - positions=(-1, 0, 1) — SHORT, HOLD, LONG
  - DiscretizeActionWrapper — snaps continuous output to {-1, 0, 1}
  - FractalGuidedReward — portfolio PnL + fractal alignment + per-trade penalty
  - 10× training fee (0.5%) to force selective trading
  - SubprocVecEnv with 32 workers
  - Walk-forward eval every 25k steps
"""

import os, sys, warnings, hashlib, pickle, json
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence
from functools import partial

# Force unbuffered stdout so nohup logs stream in real-time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import gymnasium as gym
import torch

sys.path.append(str(Path(__file__).parent))

from trading_env.trading_env import TradingEnv, differential_sharpe_reward
from sb3_contrib import RecurrentPPO
from models.huber_rppo import HuberRecurrentPPO, swap_critic_extractor, CriticBurnInCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, DummyVecEnv, VecNormalize, VecEnv
)
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from utils.turbulence import add_turbulence_feature
from utils.schedulers import cosine_lr
from trading_env.risk_wrappers import MultiLevelRiskWrapper
from trading_env.cvar_wrapper import CVaRConstraintWrapper, CVaRCallback
from trading_env.regime_sampler import RegimeBalancedWrapper
from models.moe_policy import (
    MoEFeaturesExtractor,
    GateMonitorCallback,
    MoELoadBalanceCallback,
    get_regime_feature_indices,
    collect_val_observations,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

CONFIG: Dict[str, Any] = {
    "data": {
        "cache_path": "cache/data_v1_430340a861af5f8f9bcbd7a4ca16ba95.parquet",
        "train_start": "2019-01-01",
        "train_end":   "2024-06-30",
        "val_start":   "2024-07-01",
        "val_end":     "2025-06-30",
        "test_start":  "2025-07-01",
        # Optional: path to ETH/USDT 1h feather for cross-asset features.
        # Set to "" to skip. Relative to working directory when train2.py is run.
        "eth_path":    "../../data/binance/ETH_USDT-1h.feather",
    },
    "env": {
        "positions":             (-1.0, 0.0, 1.0),  # SHORT, HOLD, LONG
        "initial_balance":       1_000.0,
        "fee_rate":              5e-4,          # 0.05% real fee
        "training_fee_rate":     1e-3,          # 0.10% in training = 2×
        "slippage_bps":          2,
        "window_size":           None,
        "train_episode_bars":    720,           # ~120 days on 4h
    },
    "reward": {
        "type":                    "dsr_fractal",   # "dsr_fractal" | "fractal"
        "eta":                     0.01,            # DSR EMA adaptation rate
        "fractal_alignment_scale": 0.0,
        "trade_penalty":           1e-4,
        "inactivity_penalty":      0.0,       # disabled — anti-static penalties cause churn
        "inactivity_window":       12,
    },
    "model": {
        "learning_rate":    5e-5,
        "n_steps":          2048,
        "batch_size":       16,
        "n_epochs":         3,
        "gamma":            0.995,
        "gae_lambda":       0.95,
        "clip_range":       0.10,
        "ent_coef":         0.05,
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
        "patience":        200_000,
        "log_dir":         "logs_stable",
        "seeds":           [42, 123, 456],   # train multiple seeds to assess variance
    },
    "risk": {
        # Portfolio-level: force HOLD above this drawdown
        "dd_hard":              0.15,
        # Regime-level: force HOLD above this normalised turbulence (1.0 = 99th pct)
        "turbulence_threshold": 1.5,
        # Strategy-level: trailing stop at N × ATR below running price high
        "atr_stop_mult":        2.0,
        "cooldown_steps":       24,           # bars to stay flat after stop fires
        # Column names must match what's in df
        "turbulence_col":       "turbulence_feature",
        "atr_col":              "feature_atr",
        # Whether to apply turbulence/trailing-stop during training
        # (False lets the agent explore; True gives cleaner training signal)
        "apply_in_training":    True,
        "apply_in_eval":        True,
    },
    "moe": {
        # Set enabled=True to replace the default MlpLstmPolicy feature extractor
        # with a Mixture-of-Experts extractor (jointly trained, learned regime gating).
        "enabled":          True,
        # Number of expert MLPs.  3 = bull / bear / sideways is the natural choice.
        "n_experts":        8,
        # Output dimension of each expert (fed as input to the LSTM).
        # Smaller than default obs dim gives a bottleneck that forces specialisation.
        "expert_dim":       64,
        # Hidden layer size of the gating MLP (reads regime features → softmax).
        "gate_hidden_dim":  32,
        # Gate entropy coef: penalise collapsed gate distributions.
        # Higher = more uniform routing.  0.1 is standard.
        "gate_entropy_coef": 0.1,
        # How often (timesteps) to log gate-weight statistics to TensorBoard.
        "gate_log_freq":    100_000,
        # Max observations to collect for gate monitoring (from val set).
        "gate_monitor_steps": 1_000,
    },
    "cvar": {
        # CVaR-constrained policy optimization (Coache, Jaimungal & Cartea 2022).
        # Adds a Lagrangian tail-risk penalty at episode end when episode return
        # falls in the α-worst tail.  Provable tail-risk bounds for risk committees.
        "enabled":          True,
        # Tail probability: penalise worst α% of episodes.
        "alpha":            0.05,
        # BUG #8 FIX: cvar_budget was -1e-3 (too large). 
        # Current reward scale is ~1e-5 per step. Setting budget to -2e-5.
        "cvar_budget":      -2e-5,
        # Lagrange multiplier learning rate (much smaller than policy LR).
        "lambda_lr":        0.01,
        # Upper clip for λ — prevents reward collapse if CVaR constraint is violated.
        "lambda_max":       10.0,
        # Rolling window of episode returns used for CVaR estimation.
        "buffer_episodes":  100,
        # Episodes between Lagrange updates and env broadcasts.
        "update_freq":      10,
        # Apply only during training (eval metrics must not include the penalty).
        "apply_in_training": True,
        "apply_in_eval":     False,
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# REWARD
# ──────────────────────────────────────────────────────────────────────────────

class FractalGuidedReward:
    """
    Portfolio PnL reward + fractal alignment bonus + per-trade penalty.
    Picklable class (not closure) for SubprocVecEnv compatibility.
    """
    def __init__(self, alignment_scale: float = 2.5e-4, trade_penalty: float = 2e-3):
        self.alignment_scale = alignment_scale
        self.trade_penalty   = trade_penalty

    def __call__(self, history) -> float:
        pv_now  = history["portfolio_valuation", -1]
        pv_prev = history["portfolio_valuation", -2]
        pnl_r   = float(np.log(pv_now / pv_prev)) if pv_prev > 1e-8 else 0.0

        # Fractal alignment bonus
        try:
            position  = float(history["position", -1])
            deception = float(history["data_fractal_deception_feature", -1])
            pnl_r    += self.alignment_scale * position * deception
        except (KeyError, IndexError, ValueError):
            pass

        # Per-trade penalty: penalise any position change regardless of size
        try:
            pos_now  = float(history["position", -1])
            pos_prev = float(history["position", -2])
            if abs(pos_now - pos_prev) > 1e-6:
                pnl_r -= self.trade_penalty
        except (KeyError, IndexError, ValueError):
            pass

        return pnl_r


class DSRFractalReward:
    """
    Differential Sharpe Ratio base reward (Moody & Saffell 2001) combined with
    the fractal alignment bonus from FractalGuidedReward.

    DSR directly optimises the online Sharpe ratio instead of raw log-PnL,
    making the agent naturally risk-averse without extra reward shaping.
    The EMA state (sharpe_A, sharpe_B) is stored inside History so it resets
    automatically on each episode — no external reset hook needed.

    Picklable class (not closure) for SubprocVecEnv compatibility.
    """
    def __init__(self, alignment_scale: float = 2.5e-4, trade_penalty: float = 2e-3,
                 eta: float = 0.01, dsr_a_init: float = 0.0, dsr_b_init: float = 0.0,
                 inactivity_penalty: float = 5e-5, inactivity_window: int = 12):
        self.alignment_scale = alignment_scale
        self.trade_penalty   = trade_penalty
        self.eta             = eta
        self.dsr_a_init      = dsr_a_init
        self.dsr_b_init      = dsr_b_init
        self.inactivity_penalty = inactivity_penalty
        self.inactivity_window  = inactivity_window

    def __call__(self, history) -> float:
        # DSR base — pass fee_penalty=0.0 to avoid double-counting the trade penalty
        r = differential_sharpe_reward(history, eta=self.eta, fee_penalty=0.0, hold_penalty=0.0)

        # Fractal alignment bonus (same as FractalGuidedReward)
        try:
            position  = float(history["position", -1])
            deception = float(history["data_fractal_deception_feature", -1])
            r += self.alignment_scale * position * deception
        except (KeyError, IndexError, ValueError):
            pass

        # Per-trade penalty applied once here (not inside differential_sharpe_reward)
        try:
            pos_now  = float(history["position", -1])
            pos_prev = float(history["position", -2])
            if abs(pos_now - pos_prev) > 1e-6:
                r -= self.trade_penalty
        except (KeyError, IndexError, ValueError):
            pass

        # Inactivity penalty: penalize extended flat (position=0) periods
        # to break the HOLD Nash equilibrium
        if self.inactivity_penalty > 0:
            try:
                pos = float(history["position", -1])
                if abs(pos) < 1e-6:  # currently flat
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


# ──────────────────────────────────────────────────────────────────────────────
# ACTION DISCRETIZER
# ──────────────────────────────────────────────────────────────────────────────

class DiscretizeActionWrapper(gym.ActionWrapper):
    """
    Replace continuous Box action space with Discrete(K).

    The policy outputs a discrete action index, which is mapped to the
    corresponding position value. This ensures RecurrentPPO uses a
    categorical distribution with proper discrete entropy — avoiding
    the "discretization trap" where a Gaussian policy looks exploratory
    in continuous space but always snaps to the same discrete action.
    """
    def __init__(self, env: gym.Env, positions: Sequence[float] = (0.0, 1.0)):
        super().__init__(env)
        self._positions = np.array(sorted(positions), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self._positions))

    def action(self, action) -> float:
        idx = int(action)
        return float(self._positions[idx])


# ──────────────────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────────────────

def load_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    path = Path(cfg["data"]["cache_path"])
    df = pd.read_parquet(path)

    # Attach fractal features if missing
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

    # Turbulence index (Kritzman & Li 2010) — add if missing
    if "turbulence_feature" not in df.columns:
        print("   Computing turbulence index...")
        # Accept either train2 naming (log_return_1h_feature) or
        # train_rl cache naming (log_return_feature, volatility_24_feature, etc.)
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

    # Cross-asset features (ETH/BTC relative dynamics — optional)
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

    # Dynamic slippage (Almgren & Chriss 2001, square-root impact)
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


def split_data(df: pd.DataFrame, cfg: Dict[str, Any]):
    d = cfg["data"]
    train = df[d["train_start"]:d["train_end"]].copy()
    val   = df[d["val_start"]:d["val_end"]].copy()
    test  = df[d["test_start"]:].copy()
    return train, val, test


def build_regime_balanced_val(df: pd.DataFrame, cfg: Dict[str, Any],
                              regime_col: str = "trend_return_180_feature") -> pd.DataFrame:
    """Build a validation set with equal exposure to bull, bear, and sideways regimes.

    Instead of a single chronological slice that may be dominated by one regime,
    stitch together ~3-month segments from different regimes within the val+test
    date range.  This forces the early-stopping checkpoint to reward multi-regime
    competence rather than overfitting the macro trend of a single period.

    Returns a concatenated DataFrame (with a synthetic continuous index) that
    TradingEnv can walk through sequentially.
    """
    d = cfg["data"]
    oos = df[d["val_start"]:].copy()  # all out-of-sample data

    trend = oos[regime_col]
    thresholds = np.percentile(trend.dropna(), [33, 67])

    # Label each bar
    oos["_regime"] = np.digitize(trend.values, thresholds)  # 0=bear, 1=side, 2=bull

    # Find contiguous regime blocks of at least 180 bars (~1 month on 4h)
    min_block = 180
    target_bars = 540  # ~3 months per regime segment

    segments = []
    for regime_id, regime_name in [(0, "bear"), (1, "sideways"), (2, "bull")]:
        mask = oos["_regime"] == regime_id
        # Find longest contiguous run of this regime
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
            # Fallback: just take the first target_bars bars of this regime
            chunk = oos[mask].head(target_bars)
            if len(chunk) > 0:
                segments.append(chunk)
                print(f"  [RegimeVal] {regime_name}: {len(chunk)} bars (scattered)")
            continue

        # Pick the longest run and take up to target_bars from it
        best = max(runs, key=lambda x: x[2])
        seg = oos.loc[best[0]:best[1]].head(target_bars)
        segments.append(seg)
        print(f"  [RegimeVal] {regime_name}: {len(seg)} bars "
              f"({seg.index[0].date()} → {seg.index[-1].date()})")

    if not segments:
        print("  [RegimeVal] WARNING: no regime segments found, falling back to chronological val")
        return df[d["val_start"]:d["val_end"]].copy()

    # Concatenate and reset index to create a synthetic continuous timeline
    val_df = pd.concat(segments, axis=0)
    val_df = val_df.drop(columns=["_regime"], errors="ignore")

    print(f"  [RegimeVal] Total: {len(val_df)} bars across {len(segments)} regime segments")
    return val_df


# ──────────────────────────────────────────────────────────────────────────────
# ENV BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def _build_single_env(df: pd.DataFrame, cfg: Dict[str, Any],
                      training: bool, name: str, rank: int = 0) -> Monitor:
    env_cfg    = cfg["env"]
    reward_cfg = cfg["reward"]
    risk_cfg   = cfg.get("risk", {})

    fee = env_cfg["training_fee_rate"] if training else env_cfg["fee_rate"]
    ep_dur = env_cfg.get("train_episode_bars", 720) if training else "max"

    reward_fn: FractalGuidedReward | DSRFractalReward
    if reward_cfg.get("type", "fractal") == "dsr_fractal":
        reward_fn = DSRFractalReward(
            alignment_scale=reward_cfg["fractal_alignment_scale"],
            trade_penalty=reward_cfg["trade_penalty"],
            eta=reward_cfg.get("eta", 0.01),
            dsr_a_init=reward_cfg.get("dsr_a_init", 0.0),
            dsr_b_init=reward_cfg.get("dsr_b_init", 0.0),
            inactivity_penalty=reward_cfg.get("inactivity_penalty", 5e-5),
            inactivity_window=reward_cfg.get("inactivity_window", 12),
        )
    else:
        reward_fn = FractalGuidedReward(
            alignment_scale=reward_cfg["fractal_alignment_scale"],
            trade_penalty=reward_cfg["trade_penalty"],
        )

    env = TradingEnv(
        df,
        positions=list(env_cfg["positions"]),
        trading_fees=0.0,
        fee_rate=fee,
        slippage_bps=env_cfg.get("slippage_bps", 2),
        portfolio_initial_value=env_cfg.get("initial_balance", 1_000.0),
        initial_position='random' if training else 0.0,
        windows=env_cfg.get("window_size"),
        max_episode_duration=ep_dur,
        verbose=0,
        name=f"{name}_{rank}",
        reward_function=reward_fn,
    )
    env = DiscretizeActionWrapper(env, positions=list(env_cfg["positions"]))

    # Multi-level risk wrapper (turbulence, drawdown, trailing stop)
    apply = (not training and risk_cfg.get("apply_in_eval", True)) or \
            (training and risk_cfg.get("apply_in_training", False))
    if risk_cfg and apply:
        env = MultiLevelRiskWrapper(
            env,
            dd_hard=risk_cfg.get("dd_hard", 0.15) * (1.5 if training else 1.0),
            turbulence_col=risk_cfg.get("turbulence_col", "turbulence_feature"),
            turbulence_threshold=risk_cfg.get("turbulence_threshold", 1.5),
            atr_col=risk_cfg.get("atr_col", "feature_atr"),
            atr_stop_mult=risk_cfg.get("atr_stop_mult", 2.0),
            cooldown_steps=risk_cfg.get("cooldown_steps", 24),
            apply_turbulence=not training,    # regime-level only in eval
            apply_trailing_stop=not training, # trailing stop only in eval
        )

    # CVaR constraint wrapper (Lagrangian tail-risk penalty — training only)
    cvar_cfg = cfg.get("cvar", {})
    cvar_apply = (training and cvar_cfg.get("apply_in_training", True)) or \
                 (not training and cvar_cfg.get("apply_in_eval", False))
    if cvar_cfg.get("enabled", False) and cvar_apply:
        env = CVaRConstraintWrapper(
            env,
            alpha=cvar_cfg.get("alpha", 0.05),
            cvar_budget=cvar_cfg.get("cvar_budget", -1e-3),
        )

    # Regime-balanced episode sampling (training only)
    if training:
        env = RegimeBalancedWrapper(env, regime_col="trend_return_180_feature", n_bins=3)

    return Monitor(env)


def make_train_envs(df: pd.DataFrame, cfg: Dict[str, Any]) -> VecNormalize:
    n = cfg["training"]["n_envs"]
    # Save to temp parquet for workers
    tmp = Path("cache/temp/_train_tmp.parquet")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp)

    def make_fn(rank: int):
        def _init():
            _df = pd.read_parquet(tmp)
            return _build_single_env(_df, cfg, training=True, name="train", rank=rank)
        return _init

    vec_env = SubprocVecEnv([make_fn(i) for i in range(n)])
    # Normalize rewards only (not obs) — gives the value function well-conditioned
    # learning targets regardless of raw DSR reward scale.
    return VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                        clip_reward=10.0, gamma=cfg["model"]["gamma"])


def make_val_env(df: pd.DataFrame, cfg: Dict[str, Any]) -> DummyVecEnv:
    return DummyVecEnv([lambda: _build_single_env(df, cfg, training=False, name="val")])


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def _sharpe(nav: np.ndarray) -> float:
    if len(nav) < 2:
        return -10.0
    r = np.diff(np.log(np.clip(nav, 1e-8, None)))
    r = np.clip(r, -0.5, 0.5)
    mu, sigma = r.mean(), r.std()
    if sigma < 1e-8:
        return 10.0 if mu > 0 else -10.0
    return float(np.clip((mu / sigma) * np.sqrt(8_760), -10, 10))


def _sortino(nav: np.ndarray) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    if len(nav) < 2:
        return 0.0
    r = np.diff(np.log(np.clip(nav, 1e-8, None)))
    r = np.clip(r, -0.5, 0.5)
    mu = r.mean()
    down = r[r < 0]
    dd_std = down.std() if len(down) > 1 else 1e-8
    if dd_std < 1e-8:
        return 10.0 if mu > 0 else -10.0
    return float(np.clip((mu / dd_std) * np.sqrt(8_760), -10, 10))


def _calmar(nav: np.ndarray) -> float:
    """Calmar ratio = annualised return / max drawdown."""
    if len(nav) < 2:
        return 0.0
    total_ret = nav[-1] / nav[0] - 1
    peaks = np.maximum.accumulate(nav)
    max_dd = float(np.max(1 - nav / peaks))
    if max_dd < 1e-8:
        return 10.0 if total_ret > 0 else 0.0
    ann_factor = 8_760 / max(len(nav), 1)  # bars → annualised
    return float(np.clip((total_ret * ann_factor) / max_dd, -10, 10))


def _max_dd_duration(nav: np.ndarray) -> int:
    """Max drawdown duration in bars."""
    if len(nav) < 2:
        return 0
    peaks = np.maximum.accumulate(nav)
    in_dd = nav < peaks
    max_dur = 0
    cur = 0
    for d in in_dd:
        if d:
            cur += 1
            max_dur = max(max_dur, cur)
        else:
            cur = 0
    return max_dur


def evaluate(model: RecurrentPPO, env: VecEnv, max_steps: int = 0) -> Dict[str, float]:
    if max_steps <= 0:
        try:
            max_steps = len(env.get_attr("df")[0]) - 1
        except Exception:
            max_steps = 10_000

    obs = env.reset()
    lstm_state = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    pv, prices, actions = [1_000.0], [], []

    for _ in range(max_steps):
        action, lstm_state = model.predict(
            obs, state=lstm_state, episode_start=episode_starts, deterministic=True
        )
        obs, _, dones, infos = env.step(action)
        episode_starts = dones

        info = infos[0] if isinstance(infos, list) else infos
        if "portfolio_valuation" in info:
            pv.append(float(info["portfolio_valuation"]))
        for key in ("data_close", "close"):
            if key in info:
                prices.append(float(info[key]))
                break

        a = float(np.asarray(action).flat[0])
        actions.append(a)

        if dones[0]:
            break

    nav = np.array(pv)
    ret = float(nav[-1] / nav[0] - 1) * 100 if len(nav) > 1 else 0.0
    mkt = float(prices[-1] / prices[0] - 1) * 100 if len(prices) > 1 else 0.0
    sharpe = _sharpe(nav)
    sortino = _sortino(nav)
    calmar = _calmar(nav)
    dd_dur = _max_dd_duration(nav)

    peaks = np.maximum.accumulate(nav)
    dd = float(np.max(1 - nav / peaks)) * 100 if len(nav) > 1 else 0.0

    act = np.array(actions)
    # Discrete actions: 0=SHORT, 1=HOLD, 2=LONG
    n_short = int((act == 0).sum())
    n_hold  = int((act == 1).sum())
    n_long  = int((act == 2).sum())
    n_trades = int(np.sum(np.abs(np.diff(act)) > 0)) if len(act) > 1 else 0

    # Win rate: count trades where portfolio value increased
    trade_boundaries = np.where(np.abs(np.diff(act)) > 0)[0] if len(act) > 1 else []
    wins = 0
    for i in range(len(trade_boundaries) - 1):
        start_idx = trade_boundaries[i]
        end_idx = trade_boundaries[i + 1]
        if end_idx < len(nav) and start_idx < len(nav):
            if nav[end_idx] > nav[start_idx]:
                wins += 1
    win_rate = wins / max(1, len(trade_boundaries) - 1) * 100

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "return": ret,
        "market": mkt,
        "alpha": ret - mkt,
        "dd": dd,
        "dd_duration": dd_dur,
        "trades": n_trades,
        "win_rate": win_rate,
        "pct_long": n_long / max(1, len(act)) * 100,
        "pct_short": n_short / max(1, len(act)) * 100,
        "pct_hold": n_hold / max(1, len(act)) * 100,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CALLBACK
# ──────────────────────────────────────────────────────────────────────────────

class EvalCallback(BaseCallback):
    """
    Validation callback with early stopping and generalization ratio tracking.

    Generalization ratio (Sheppert 2025) = val_return / train_eval_return.
    A ratio < 0.5 signals overfitting.  Logged at each eval step.

    Parameters
    ----------
    train_eval_env : optional VecEnv over the last N bars of training data.
        When provided, train performance is measured alongside val performance.
    """

    def __init__(self, val_env: VecEnv, cfg: Dict[str, Any], save_dir: Path,
                 train_eval_env: Optional[VecEnv] = None, seed: int = 42):
        super().__init__(verbose=0)
        self.val_env        = val_env
        self.train_eval_env = train_eval_env
        self.eval_freq      = cfg["training"]["eval_freq"]
        self.patience       = cfg["training"]["patience"]
        self.save_dir       = save_dir
        self.seed           = seed
        self.best           = -np.inf
        self.no_improve     = 0
        self.history: List[Dict] = []
        self._next_eval     = self.eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True
        self._next_eval += self.eval_freq

        m = evaluate(self.model, self.val_env)
        step_k = f"{self.num_timesteps // 1000}k"

        # Generalization ratio: val_return vs recent train performance
        gen_ratio_str = ""
        if self.train_eval_env is not None:
            t = evaluate(self.model, self.train_eval_env)
            train_ret = t["return"]
            val_ret   = m["return"]
            # Use absolute returns so ratio sign is meaningful
            gen_ratio = val_ret / max(abs(train_ret), 1.0)
            gen_ratio_str = f"  GenR {gen_ratio:>+.2f}"
            m["gen_ratio"]   = gen_ratio
            m["train_return"] = train_ret

        self.history.append({"step": self.num_timesteps, **m})

        line = (
            f"  │ {step_k:>5}  Sharpe {m['sharpe']:>+6.3f}  "
            f"Sort {m['sortino']:>+6.3f}  Calm {m['calmar']:>+6.3f}  "
            f"Ret {m['return']:>+7.2f}%  BH {m['market']:>+7.2f}%  "
            f"DD {m['dd']:>5.1f}% ({m['dd_duration']}b)  "
            f"T {m['trades']:>4d}  W {m['win_rate']:>4.1f}%  "
            f"L {m['pct_long']:>4.1f}%  S {m['pct_short']:>4.1f}%{gen_ratio_str}"
        )
        print(line, flush=True)
        # Write eval line to a separate file that we can always read
        with open(self.save_dir / "evals.log", "a") as f:
            f.write(line + "\n")

        gen_ratio = m.get("gen_ratio", float("nan"))
        passes_gen_gate = (gen_ratio != gen_ratio) or gen_ratio > 0.3  # NaN passes

        if m["sharpe"] > self.best and passes_gen_gate:
            self.best       = m["sharpe"]
            self.no_improve = 0
            self.model.save(str(self.save_dir / "best"))

            # Persist key metrics so TurbulenceGatedEnsemble can weight by val Sharpe
            metrics_payload = {
                "best_val_sharpe":  float(m["sharpe"]),
                "best_val_sortino": float(m["sortino"]),
                "best_val_calmar":  float(m["calmar"]),
                "best_val_return":  float(m["return"]),
                "best_val_alpha":   float(m["alpha"]),
                "best_val_dd":      float(m["dd"]),
                "best_val_dd_dur":  int(m["dd_duration"]),
                "best_val_trades":  int(m["trades"]),
                "best_val_winrate": float(m["win_rate"]),
                "best_step":        self.num_timesteps,
                "gen_ratio":        float(m.get("gen_ratio", float("nan"))),
                "seed":             self.seed,
            }
            metrics_file = self.save_dir / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics_payload, f, indent=2)
        else:
            self.no_improve += self.eval_freq
            if self.no_improve >= self.patience:
                print(f"\n  Early stop at {self.num_timesteps:,} — "
                      f"no improvement for {self.patience//1000}k steps")
                return False

        return True


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main(seed: int = 42, resume_dir: Optional[str] = None, fold: int = 0):
    cfg = CONFIG
    tp  = cfg["training"]
    ep  = cfg["env"]

    if fold > 0:
        offset = pd.Timedelta(days=90 * fold)
        for key in ["train_start", "train_end", "val_start", "val_end", "test_start"]:
            dt = pd.to_datetime(cfg["data"][key]) + offset
            cfg["data"][key] = str(dt.date())

    # Reproducibility
    import random, torch as _torch
    random.seed(seed)
    np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)

    # Find next run dir
    log_root = Path(tp["log_dir"])
    existing = sorted(log_root.glob("Run_*"))
    run_num  = len(existing) + 1
    run_dir  = log_root / f"Run_{run_num:03d}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  RL TRADING  (train2.py)")
    print(f"  Pair       : BTC_USDT  4h")
    print(f"  Positions  : {ep['positions']}")
    print(f"  Reward     : {cfg['reward'].get('type', 'fractal')}  (η={cfg['reward'].get('eta', 'n/a')})")
    print(f"  Fees train : {ep['training_fee_rate']*100:.2f}%  eval: {ep['fee_rate']*100:.3f}%")
    print(f"  Envs×steps : {tp['n_envs']} × {tp['total_timesteps']//1_000_000}M")
    print(f"  Seed       : {seed}")
    print(f"  Run dir    : {run_dir}")
    print(f"{'─'*60}\n")

    # Data
    df = load_data(cfg)
    train_df, val_df, test_df = split_data(df, cfg)
    print(f"  Train : {train_df.index[0].date()} → {train_df.index[-1].date()}  ({len(train_df)} bars)")
    print(f"  Val   : {val_df.index[0].date()} → {val_df.index[-1].date()}  ({len(val_df)} bars)")
    print(f"  Test  : {test_df.index[0].date()} → {test_df.index[-1].date()}  ({len(test_df)} bars)\n")

    # Build regime-balanced val set for early-stopping
    # (forces checkpoints to reward multi-regime competence, not macro trend overfitting)
    regime_val_df = build_regime_balanced_val(df, cfg)
    print()

    # Precompute DSR warm-start from training data (avoids bimodal reward signal)
    if cfg["reward"].get("type") == "dsr_fractal":
        log_rets = np.diff(np.log(train_df["close"].values))
        cfg["reward"]["dsr_a_init"] = float(log_rets.mean())
        cfg["reward"]["dsr_b_init"] = float((log_rets ** 2).mean())
        print(f"  DSR warm-start: A={cfg['reward']['dsr_a_init']:.6f}  "
              f"B={cfg['reward']['dsr_b_init']:.6f}")

    # Environments
    print("  Initializing SubprocVecEnv...", flush=True)
    train_envs = make_train_envs(train_df, cfg)
    val_env    = make_val_env(regime_val_df, cfg)  # regime-balanced val for early-stopping

    # Train-eval env: last 90 days of training data — for generalization ratio
    # Use ~2160 bars of 1h data (90 days) or whole train if shorter
    n_train_eval = min(2_160, len(train_df))
    train_eval_df = train_df.iloc[-n_train_eval:].copy()
    train_eval_env = make_val_env(train_eval_df, cfg)
    print(f"  Train-eval slice: {train_eval_df.index[0].date()} → "
          f"{train_eval_df.index[-1].date()} ({len(train_eval_df)} bars) — for gen ratio")
    print("  Environments ready.\n")

    # Model
    mp       = cfg["model"]
    moe_cfg  = cfg.get("moe", {})
    moe_on   = moe_cfg.get("enabled", False)

    policy_kwargs = dict(
        lstm_hidden_size=mp["lstm_hidden_size"],
        n_lstm_layers=mp["n_lstm_layers"],
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        enable_critic_lstm=True,
        shared_lstm=False,
        share_features_extractor=False,  # decouple actor/critic gradient flows
    )

    # ── Mixture-of-Experts feature extractor ──────────────────────────────────
    gate_monitor_cb = None
    if moe_on:
        feature_cols = [c for c in train_df.columns if "feature" in c]
        regime_idx   = get_regime_feature_indices(feature_cols)
        print(f"  MoE enabled: {moe_cfg['n_experts']} experts  "
              f"| gate inputs: {len(regime_idx)} regime features")

        policy_kwargs.update(dict(
            features_extractor_class  = MoEFeaturesExtractor,
            features_extractor_kwargs = dict(
                n_experts          = moe_cfg.get("n_experts", 3),
                expert_dim         = moe_cfg.get("expert_dim", 64),
                gate_hidden_dim    = moe_cfg.get("gate_hidden_dim", 32),
                regime_feature_idx = regime_idx,
                gate_entropy_coef  = moe_cfg.get("gate_entropy_coef", 0.1),
            ),
        ))
        # Shrink LSTM input to match expert_dim output
        policy_kwargs["net_arch"] = dict(pi=[128, 64], vf=[128, 64])

        # Collect a sample of val observations for gate monitoring
        print("  Collecting val observations for gate monitor...", flush=True)
        val_obs_sample = collect_val_observations(
            val_df, cfg,
            max_steps=moe_cfg.get("gate_monitor_steps", 1_000),
        )
        # Regime labels from market_regime_feature column (if present)
        regime_col = "market_regime_feature"
        if regime_col in val_df.columns:
            regime_labels = val_df[regime_col].fillna(-1).values[:len(val_obs_sample)]
        else:
            regime_labels = None

        gate_monitor_cb = GateMonitorCallback(
            val_obs       = val_obs_sample,
            log_freq      = moe_cfg.get("gate_log_freq", 100_000),
            regime_labels = regime_labels,
        )
        lb_cb = MoELoadBalanceCallback(lb_lr=1e-3, verbose=1)
    else:
        print("  MoE disabled — using default MlpLstm feature extractor")

    if resume_dir is not None:
        checkpoint = Path(resume_dir) / "best.zip"
        print(f"  Resuming from {checkpoint} (lr → {mp['learning_rate']}  target_kl → {mp['target_kl']})")
        model = HuberRecurrentPPO.load(checkpoint, env=train_envs, device="cuda")
        model.target_kl    = mp["target_kl"]
        model.n_epochs     = mp["n_epochs"]
        model.batch_size   = mp["batch_size"]
        model.learning_rate = mp["learning_rate"]
        for pg in model.policy.optimizer.param_groups:
            pg["lr"] = mp["learning_rate"]
        reset_timesteps = False
    else:
        model = HuberRecurrentPPO(
            policy="MlpLstmPolicy",
            env=train_envs,
            policy_kwargs=policy_kwargs,
            learning_rate=partial(cosine_lr, base=mp["learning_rate"], min_lr=1e-6),
            n_steps=mp["n_steps"],
            batch_size=mp["batch_size"],
            n_epochs=mp["n_epochs"],
            gamma=mp["gamma"],
            gae_lambda=mp["gae_lambda"],
            clip_range=mp["clip_range"],
            ent_coef=mp["ent_coef"],
            vf_coef=mp["vf_coef"],
            max_grad_norm=mp["max_grad_norm"],
            target_kl=mp["target_kl"],
            verbose=1,
            seed=seed,
            device="cuda",
            tensorboard_log=str(run_dir),
        )
        # Decouple: replace the critic's MoE extractor with a dedicated MLP
        if moe_on:
            swap_critic_extractor(model, features_dim=moe_cfg.get("expert_dim", 64))
        reset_timesteps = True

    eval_cb = EvalCallback(val_env, cfg, run_dir, train_eval_env=train_eval_env, seed=seed)
    burnin_cb = CriticBurnInCallback(burnin_steps=200_000, verbose=1)
    burnin_cb._eval_callback = eval_cb  # reset patience when actor unfreezes
    callbacks = [burnin_cb, eval_cb]  # burnin FIRST so unfreeze resets patience before eval checks it
    if gate_monitor_cb:
        callbacks.append(gate_monitor_cb)
        callbacks.append(lb_cb)

    # CVaR constraint callback (Lagrangian tail-risk, applied in training only)
    cvar_cfg = cfg.get("cvar", {})
    if cvar_cfg.get("enabled", False) and cvar_cfg.get("apply_in_training", True):
        cvar_cb = CVaRCallback(
            alpha=cvar_cfg.get("alpha", 0.05),
            cvar_budget=cvar_cfg.get("cvar_budget", -1e-3),
            lambda_lr=cvar_cfg.get("lambda_lr", 0.01),
            lambda_max=cvar_cfg.get("lambda_max", 10.0),
            buffer_episodes=cvar_cfg.get("buffer_episodes", 100),
            update_freq=cvar_cfg.get("update_freq", 10),
            verbose=1,
        )
        callbacks.append(cvar_cb)
        print(f"  CVaR constraint: α={cvar_cfg['alpha']}  "
              f"budget={cvar_cfg['cvar_budget']:.2e}  "
              f"λ_lr={cvar_cfg['lambda_lr']}")

    callback = CallbackList(callbacks)

    print("  ─── EVAL LOG ─────────────────────────────────────────────────────────────")
    print("  │  Step  Sharpe  Sortino  Calmar    Return      BH      DD    Trades  Win%   Long%  Short%  GenRatio")
    print("  ──────────────────────────────────────────────────────────────────────────")

    model.learn(
        total_timesteps=tp["total_timesteps"],
        callback=callback,
        reset_num_timesteps=reset_timesteps,
    )

    # Final summary
    print(f"\n{'═'*60}")
    history = eval_cb.history
    if history:
        best = max(history, key=lambda x: x["sharpe"])
        last_gen = best.get("gen_ratio", float("nan"))
        print(f"  Best val checkpoint : step {best['step']//1000}k")
        print(f"    Sharpe      : {best['sharpe']:>+.3f}")
        print(f"    Sortino     : {best['sortino']:>+.3f}")
        print(f"    Calmar      : {best['calmar']:>+.3f}")
        print(f"    Return      : {best['return']:>+.2f}%  (BH {best['market']:>+.2f}%)")
        print(f"    Alpha       : {best['alpha']:>+.2f}%")
        print(f"    DD          : {best['dd']:.2f}%  (duration: {best['dd_duration']} bars)")
        print(f"    Trades      : {best['trades']}  (win rate: {best['win_rate']:.1f}%)")
        print(f"    Gen ratio   : {last_gen:.2f}  (target > 0.5)")

    # Test set evaluation
    print(f"\n  ─── TEST SET EVALUATION ──────────────────────────────────────────────────")
    try:
        # Reload best checkpoint — need to swap critic extractor before state_dict loads
        best_model = HuberRecurrentPPO(
            policy="MlpLstmPolicy",
            env=val_env,
            policy_kwargs=policy_kwargs,
            device="cuda",
            seed=seed,
        )
        if moe_on:
            swap_critic_extractor(best_model, features_dim=moe_cfg.get("expert_dim", 64))
        # Now load the state dict from the saved checkpoint
        import zipfile, io
        zip_path = str(run_dir / "best.zip")
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open("policy.pth") as f:
                state_dict = torch.load(io.BytesIO(f.read()), map_location="cuda", weights_only=True)
        best_model.policy.load_state_dict(state_dict)

        test_env   = make_val_env(test_df, cfg)
        t = evaluate(best_model, test_env)
        print(f"    Sharpe  : {t['sharpe']:>+.3f}")
        print(f"    Sortino : {t['sortino']:>+.3f}")
        print(f"    Calmar  : {t['calmar']:>+.3f}")
        print(f"    Return  : {t['return']:>+.2f}%  (BH {t['market']:>+.2f}%)")
        print(f"    Alpha   : {t['alpha']:>+.2f}%")
        print(f"    DD      : {t['dd']:.2f}%  (duration: {t['dd_duration']} bars)")
        print(f"    Trades  : {t['trades']}  ({t['pct_long']:.1f}% Long, {t['pct_short']:.1f}% Short, win {t['win_rate']:.1f}%)")
        test_env.close()
    except Exception as e:
        import traceback
        print(f"    Test eval failed: {e}")
        traceback.print_exc()

    train_envs.close()
    val_env.close()
    train_eval_env.close()
    print(f"\n  Saved to: {run_dir}")
    return callback


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (run with multiple seeds to assess variance)")
    parser.add_argument("--resume", type=str, default=None, metavar="RUN_DIR",
                        help="Resume from best.zip in RUN_DIR (e.g. logs_stable/Run_007_s42)")
    parser.add_argument("--fold", type=int, default=0,
                        help="Walk-forward fold index (shifts train/val/test by N*90 days)")
    args = parser.parse_args()
    main(seed=args.seed, resume_dir=args.resume, fold=args.fold)
