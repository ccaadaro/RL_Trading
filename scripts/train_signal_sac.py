#!/usr/bin/env python3
"""
train_signal_sac.py — SAC variant of the Layer-2 RL position sizer
══════════════════════════════════════════════════════════════════════

A/B sibling of `train_signal_rl.py` (PPO). Same env, same reward, same
data splits — different algorithm. Motivation for this file:

  PPO is on-policy and exploration is limited to Gaussian noise on top
  of a deterministic mean. In iter1 the eval reward stuck at -0.17 from
  step 0 onwards — the policy converged too fast to a conservative
  HOLD-biased regime and never recovered. SAC's entropy-regularised
  objective with an auto-tuned temperature keeps exploration going, and
  its off-policy replay buffer squeezes more learning out of the same
  number of env steps.

Run:
    python train_signal_sac.py                  # default 500k steps
    python train_signal_sac.py --timesteps 1_000_000 --seed 123

Compared to PPO (same reward, same signal v2, same risk wrapper):
    - n_envs = 1            (SAC gradient step per env step; no SubprocVec)
    - learning_starts = 10k (fill buffer before training kicks in)
    - total_timesteps = 500k (SAC is sample-efficient; half of PPO budget)
    - patience = 200k       (scaled proportionally)
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from trading_env.signal_env import make_signal_env, _SIGNAL_PROB_CANDIDATES
from trading_env.risk_wrappers import MultiLevelRiskWrapper
from utils.alpha_eval_callback import AlphaEvalCallback

# Reuse PPO-side building blocks — this keeps reward/eval definitions in
# one place so an A/B run compares ALGORITHMS, not env/reward variants.
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
        "positions":          [-1, 0, 1],   # kept for parity — action_space is Box(-1,1)
    },
    "risk": {
        "dd_hard":            0.15,
        "cooldown_steps":     24,
        "apply_in_training":  True,
        "apply_in_eval":      True,
    },
    "sac": {
        "policy":              "MlpPolicy",
        "learning_rate":       3e-4,
        "buffer_size":         300_000,
        "learning_starts":     10_000,
        "batch_size":          256,
        "tau":                 0.005,
        "gamma":               0.99,
        "train_freq":          1,
        "gradient_steps":      1,
        "ent_coef":            "auto",      # auto-tuned temperature
        "target_entropy":      "auto",      # default: -action_dim = -1
        "policy_kwargs": {
            "net_arch":        [64, 64],
            # SAC's default activation is ReLU; keep it to stay close to
            # SB3 baselines and avoid a confound vs PPO's Tanh.
            "activation_fn":   torch.nn.ReLU,
        },
    },
    "training": {
        "total_timesteps":    500_000,
        "eval_freq":          10_000,
        "n_eval_episodes":    1,
        "patience":           200_000,
        "log_dir":            "logs_signal_sac",
        "model_dir":          "models",
        "seed":               42,
    },
}


# Reward is built inside main() so CLI flags can tune clip without touching
# module state. Default matches PPO iter2 for like-for-like A/Bs.
def _build_reward(clip: float = 0.15, trade_penalty: float = 5e-5):
    return SignReward(trade_penalty=trade_penalty, mode="magnitude", clip=clip)


# ══════════════════════════════════════════════════════════════════════════════
# ENV FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def _make_env(df: pd.DataFrame, seed: int, eval_mode: bool = False, reward_fn=None):
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
            name                    = f"SignalEnv-SAC-{'eval' if eval_mode else 'train'}",
            log_frequency           = 9999,
            reward_function         = reward_fn if reward_fn is not None else _build_reward(),
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
    parser.add_argument("--lr",        type=float, default=CFG["sac"]["learning_rate"])
    parser.add_argument("--buffer",    type=int, default=CFG["sac"]["buffer_size"])
    parser.add_argument("--clip",      type=float, default=0.15,
                        help="Per-bar reward clip (0 = no clip, frees distributional Q tails).")
    parser.add_argument("--trade-penalty", type=float, default=5e-5)
    parser.add_argument("--aux-start", type=str, default=None,
                        help="Start date for bear-slice aux eval (YYYY-MM-DD). "
                             "Slice must live inside the training window.")
    parser.add_argument("--aux-end",   type=str, default=None,
                        help="End date for bear-slice aux eval (YYYY-MM-DD).")
    parser.add_argument("--tag",       type=str, default="",
                        help="Suffix appended to model/log directories (e.g. 'bear').")
    parser.add_argument("--score-mode", type=str, default="min",
                        choices=["min", "aux_only", "val_only", "weighted"],
                        help="How AlphaEvalCallback combines alpha_val/alpha_aux. "
                             "'min'     → robust-both-regimes (default), "
                             "'aux_only'→ pure bear specialist (ignores VAL), "
                             "'val_only'→ legacy val-based selection, "
                             "'weighted'→ 0.5*val + 0.5*aux.")
    args = parser.parse_args()

    CFG["training"]["total_timesteps"] = args.timesteps
    CFG["training"]["seed"]            = args.seed
    CFG["sac"]["learning_rate"]        = args.lr
    CFG["sac"]["buffer_size"]          = args.buffer

    tag_suffix = f"_{args.tag}" if args.tag else ""
    CFG["training"]["log_dir"]   = f"logs_signal_sac{tag_suffix}"
    # Model directory stays "models/" but best/final paths get the tag.

    print("=" * 65)
    print(f"  SIGNAL RL — SAC position sizing (tag='{args.tag}' clip={args.clip}")
    if args.aux_start:
        print(f"              bear-slice aux: {args.aux_start} → {args.aux_end})")
    print("=" * 65)

    reward_fn = _build_reward(clip=args.clip, trade_penalty=args.trade_penalty)

    # ── Load data ─────────────────────────────────────────────────────────
    df = pd.read_parquet(CFG["data"]["path"])
    if not any(c in df.columns for c in _SIGNAL_PROB_CANDIDATES):
        raise RuntimeError(
            f"No LightGBM signal column found in parquet "
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

    # ── Bear-slice aux env for model selection (optional) ────────────────
    df_aux = None
    if args.aux_start and args.aux_end:
        df_aux = df_train[args.aux_start:args.aux_end].copy()
        if len(df_aux) < 100:
            raise RuntimeError(
                f"Bear slice {args.aux_start} → {args.aux_end} has only "
                f"{len(df_aux)} bars — check dates (must be inside train window)."
            )
        mkt_ret_aux = (df_aux["close"].iloc[-1] / df_aux["close"].iloc[0] - 1) * 100
        print(f"  Aux   : {len(df_aux)} bars  market_return={mkt_ret_aux:+.1f}%  "
              f"(bear-slice used for model selection)")

    # ── Single training env (SAC: off-policy, n_envs=1) ───────────────────
    seed = CFG["training"]["seed"]

    train_env = DummyVecEnv([_make_env(df_train, seed, eval_mode=False, reward_fn=reward_fn)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0,
        gamma=CFG["sac"]["gamma"],
    )

    # ── Model ────────────────────────────────────────────────────────────
    log_dir   = CFG["training"]["log_dir"]
    model_dir = CFG["training"]["model_dir"]
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    sac_kwargs = {k: v for k, v in CFG["sac"].items()}
    sac_kwargs["verbose"]         = 1
    sac_kwargs["tensorboard_log"] = log_dir
    sac_kwargs["seed"]            = seed
    sac_kwargs["device"]          = "cpu"

    model = SAC(env=train_env, **sac_kwargs)
    print(f"\n  Policy     : {CFG['sac']['policy']}  "
          f"net_arch={CFG['sac']['policy_kwargs']['net_arch']}")
    print(f"  Obs dim    : {train_env.observation_space.shape[0]}")
    print(f"  Device     : {model.device}")
    print(f"  Buffer     : {CFG['sac']['buffer_size']:,}  "
          f"warmup={CFG['sac']['learning_starts']:,}")
    print(f"  lr={CFG['sac']['learning_rate']}  "
          f"batch={CFG['sac']['batch_size']}  "
          f"tau={CFG['sac']['tau']}  "
          f"timesteps={CFG['training']['total_timesteps']:,}")

    # ── Callbacks ────────────────────────────────────────────────────────
    best_model_path = f"{model_dir}/signal_sac_best{tag_suffix}"

    # Alpha-based best-model selection. --score-mode controls how
    # alpha_val and alpha_aux are combined into the "best" signal:
    #   - min       → robust to both regimes (default Bull-ish general)
    #   - aux_only  → pure Bear Specialist (ignores VAL; the ensemble
    #                 routes away from this model in bull anyway)
    #   - val_only  → legacy bull-only
    #   - weighted  → 0.5*val + 0.5*aux
    _score_fns = {
        "min":       lambda av, aa: min(av, aa) if aa is not None else av,
        "aux_only":  lambda av, aa: aa if aa is not None else av,
        "val_only":  lambda av, aa: av,
        "weighted":  lambda av, aa: 0.5*av + 0.5*aa if aa is not None else av,
    }
    score_fn = _score_fns[args.score_mode]
    print(f"  Score mode : {args.score_mode}")

    alpha_eval_cb = AlphaEvalCallback(
        val_env_builder  = _make_env(df_val, seed, eval_mode=True, reward_fn=reward_fn),
        aux_env_builder  = (_make_env(df_aux, seed, eval_mode=True, reward_fn=reward_fn)
                            if df_aux is not None else None),
        save_dir         = best_model_path,
        eval_freq        = CFG["training"]["eval_freq"],
        min_start_steps  = CFG["sac"]["learning_starts"] + 10_000,
        score_fn         = score_fn,
        verbose          = 1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = f"{model_dir}/checkpoints_signal_sac{tag_suffix}",
        name_prefix = "signal_sac",
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
    final_path = f"{model_dir}/signal_sac_final{tag_suffix}"
    model.save(final_path)
    train_env.save(f"{model_dir}/signal_sac_vecnorm{tag_suffix}.pkl")
    print(f"\n  Model saved → {final_path}")

    # ── Final evaluation ──────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  FINAL EVALUATION — SAC (tag='{args.tag}')")
    print(f"{'═'*65}")

    best_model = SAC.load(
        f"{best_model_path}/best_model",
        device=model.device,
    )

    evaluate_on_split(best_model, df_val,  "VAL ")
    evaluate_on_split(best_model, df_test, "TEST")
    if df_aux is not None:
        evaluate_on_split(best_model, df_aux, "AUX ")

    train_env.close()
    print("\n  Done.")


if __name__ == "__main__":
    main()
