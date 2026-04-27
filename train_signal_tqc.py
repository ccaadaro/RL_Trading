#!/usr/bin/env python3
"""
train_signal_tqc.py — TQC (Truncated Quantile Critics) variant of Layer-2 RL
════════════════════════════════════════════════════════════════════════════

Why TQC here (vs PPO / SAC)
───────────────────────────
TQC models the DISTRIBUTION of the Q-value via quantile regression (25
quantiles × 5 critics by default = 125 quantiles total). Two concrete
wins for trading:

  1. Fat tails: crypto 1h returns are not Gaussian. The mean-Q estimate
     SAC/PPO fit is a single number that collapses distribution info.
     TQC keeps it — the actor picks actions under a RISK-AWARE Q.

  2. Risk control: `top_quantiles_to_drop_per_net` discards the N most
     optimistic quantiles per critic before taking the min. Setting
     `drop=2` ≈ using CVaR_0.92 of the value estimate. Produces a
     naturally conservative policy without reward shaping.

Same env, same reward (SignReward magnitude/clip=0.15), same risk
wrapper as PPO/SAC trainers — so comparisons across algorithms are
clean.

Run:
    python train_signal_tqc.py                    # default 500k steps
    python train_signal_tqc.py --drop 3 --seed 42 # more risk-averse
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from trading_env.signal_env import make_signal_env, _SIGNAL_PROB_CANDIDATES
from trading_env.risk_wrappers import MultiLevelRiskWrapper
from utils.alpha_eval_callback import AlphaEvalCallback

# Reuse reward + eval helpers from the PPO script so algorithm is the
# only variable in an A/B.
from train_signal_rl import SignReward, EarlyStopCallback, evaluate_on_split


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
        "episode_bars":       720,
        "positions":          [-1, 0, 1],
    },
    "risk": {
        "dd_hard":            0.15,
        "cooldown_steps":     24,
        "apply_in_training":  True,
        "apply_in_eval":      True,
    },
    "tqc": {
        "policy":                       "MlpPolicy",
        "learning_rate":                3e-4,
        "buffer_size":                  300_000,
        "learning_starts":              10_000,
        "batch_size":                   256,
        "tau":                          0.005,
        "gamma":                        0.99,
        "train_freq":                   1,
        "gradient_steps":               1,
        "ent_coef":                     "auto",
        "target_entropy":               "auto",
        # ── TQC-specific risk dial ────────────────────────────────────────
        # 0 = standard distributional critic (no truncation)
        # 2 = drop top 2 quantiles per net → actor optimises ~CVaR_0.92
        # Higher = more conservative / fewer trades
        "top_quantiles_to_drop_per_net": 2,
        "policy_kwargs": {
            "net_arch":           [64, 64],
            # TQC uses a per-critic quantile head; n_quantiles controls the
            # resolution. 25 is the paper default, works well in practice.
            "n_quantiles":        25,
            "n_critics":          5,
            "activation_fn":      torch.nn.ReLU,
        },
    },
    "training": {
        "total_timesteps":    500_000,
        "eval_freq":          10_000,
        "patience":           200_000,
        "log_dir":            "logs_signal_tqc",
        "model_dir":          "models",
        "seed":               42,
    },
}


# Same reward as PPO/SAC — algorithm A/B.
_REWARD_FN = SignReward(trade_penalty=5e-5, mode="magnitude", clip=0.15)


# ══════════════════════════════════════════════════════════════════════════════
# ENV FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def _make_env(df: pd.DataFrame, seed: int, eval_mode: bool = False):
    def _init():
        env = make_signal_env(
            df,
            positions               = CFG["env"]["positions"],
            fee_rate                = CFG["env"]["fee_rate"],
            slippage_bps            = CFG["env"]["slippage_bps"],
            portfolio_initial_value = CFG["env"]["initial_balance"],
            initial_position        = 0,
            max_episode_duration    = CFG["env"]["episode_bars"] if not eval_mode else "max",
            verbose                 = 0,
            name                    = f"SignalEnv-TQC-{'eval' if eval_mode else 'train'}",
            log_frequency           = 9999,
            reward_function         = _REWARD_FN,
        )
        if CFG["risk"]["apply_in_training"] or (eval_mode and CFG["risk"]["apply_in_eval"]):
            env = MultiLevelRiskWrapper(
                env,
                dd_hard        = CFG["risk"]["dd_hard"],
                cooldown_steps = CFG["risk"]["cooldown_steps"],
            )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=CFG["training"]["total_timesteps"])
    parser.add_argument("--seed",      type=int, default=CFG["training"]["seed"])
    parser.add_argument("--lr",        type=float, default=CFG["tqc"]["learning_rate"])
    parser.add_argument("--drop",      type=int, default=CFG["tqc"]["top_quantiles_to_drop_per_net"],
                        help="Quantiles dropped per critic (0=risk-neutral, 2=~CVaR_0.92).")
    args = parser.parse_args()

    CFG["training"]["total_timesteps"] = args.timesteps
    CFG["training"]["seed"]            = args.seed
    CFG["tqc"]["learning_rate"]        = args.lr
    CFG["tqc"]["top_quantiles_to_drop_per_net"] = args.drop

    print("=" * 65)
    print(f"  SIGNAL RL — TQC position sizing  (drop={args.drop})")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────
    df = pd.read_parquet(CFG["data"]["path"])
    if not any(c in df.columns for c in _SIGNAL_PROB_CANDIDATES):
        raise RuntimeError(
            f"No LightGBM signal column found "
            f"(looked for {_SIGNAL_PROB_CANDIDATES}). "
            "Run scripts/retrain_signal_v2.py first."
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

    # ── Envs ──────────────────────────────────────────────────────────────
    seed = CFG["training"]["seed"]

    train_env = DummyVecEnv([_make_env(df_train, seed, eval_mode=False)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0,
        gamma=CFG["tqc"]["gamma"],
    )

    # ── Model ────────────────────────────────────────────────────────────
    log_dir   = CFG["training"]["log_dir"]
    model_dir = CFG["training"]["model_dir"]
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    tqc_kwargs = {k: v for k, v in CFG["tqc"].items()}
    tqc_kwargs["verbose"]         = 1
    tqc_kwargs["tensorboard_log"] = log_dir
    tqc_kwargs["seed"]            = seed
    tqc_kwargs["device"]          = "cpu"

    model = TQC(env=train_env, **tqc_kwargs)
    pk = CFG['tqc']['policy_kwargs']
    print(f"\n  Policy     : {CFG['tqc']['policy']}  net_arch={pk['net_arch']}")
    print(f"  Obs dim    : {train_env.observation_space.shape[0]}")
    print(f"  Critics    : {pk['n_critics']}  "
          f"Quantiles/net: {pk['n_quantiles']}  "
          f"Drop top N  : {CFG['tqc']['top_quantiles_to_drop_per_net']}")
    print(f"  Device     : {model.device}")
    print(f"  Buffer     : {CFG['tqc']['buffer_size']:,}  "
          f"warmup={CFG['tqc']['learning_starts']:,}")
    print(f"  lr={CFG['tqc']['learning_rate']}  "
          f"batch={CFG['tqc']['batch_size']}  "
          f"tau={CFG['tqc']['tau']}  "
          f"timesteps={CFG['training']['total_timesteps']:,}")

    # ── Callbacks ────────────────────────────────────────────────────────
    best_model_path = f"{model_dir}/signal_tqc_best"

    alpha_eval_cb = AlphaEvalCallback(
        val_env_builder  = _make_env(df_val, seed, eval_mode=True),
        save_dir         = best_model_path,
        eval_freq        = CFG["training"]["eval_freq"],
        min_start_steps  = CFG["tqc"]["learning_starts"] + 10_000,
        verbose          = 1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = f"{model_dir}/checkpoints_signal_tqc",
        name_prefix = "signal_tqc",
        verbose     = 0,
    )
    early_stop = EarlyStopCallback(patience=CFG["training"]["patience"], verbose=1)

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Training started...")
    print(f"{'─'*65}\n")

    import sys as _sys
    _progress_bar = _sys.stdout.isatty()

    model.learn(
        total_timesteps      = CFG["training"]["total_timesteps"],
        callback             = [alpha_eval_cb, checkpoint_cb, early_stop],
        progress_bar         = _progress_bar,
        reset_num_timesteps  = True,
    )

    # ── Save final ────────────────────────────────────────────────────────
    final_path = f"{model_dir}/signal_tqc_final"
    model.save(final_path)
    train_env.save(f"{model_dir}/signal_tqc_vecnorm.pkl")
    print(f"\n  Model saved → {final_path}")

    # ── Final evaluation ──────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  FINAL EVALUATION — TQC (drop={args.drop})")
    print(f"{'═'*65}")

    best_model = TQC.load(
        f"{best_model_path}/best_model",
        device=model.device,
    )

    evaluate_on_split(best_model, df_val,  "VAL ")
    evaluate_on_split(best_model, df_test, "TEST")

    train_env.close()
    print("\n  Done.")


if __name__ == "__main__":
    main()
