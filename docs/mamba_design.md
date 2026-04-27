"""
models/mamba_design.py
──────────────────────
DESIGN DOCUMENT — Mamba SSM Architecture for RL Trading
========================================================

This file is a **design document**, not executable code. It describes the
architecture, integration strategy, and implementation steps for replacing
the LSTM memory core in RecurrentPPO with a Mamba (S4/S6) State Space Model.

Status: APPROVED DESIGN — implementation pending.

1. Motivation
─────────────
LSTM limitations in the current system:

  - Fixed hidden state (h_t, c_t) cannot capture dependencies beyond ~200 steps
    effectively. With 4h bars, this is ~33 days — too short for macro regime cycles.
  - O(L) sequential computation with no parallelism during training forward pass.
  - Hidden state size (1024 × 2 layers = 2048 params per env) creates memory pressure
    with 128 parallel environments.

Mamba advantages:

  - Selective State Space: input-dependent gating (like LSTM) but with O(L) or O(L log L)
    parallel scan for training, making it 3-5x faster on long sequences.
  - Theoretically unlimited memory horizon — state accumulates information continuously.
  - Smaller state footprint: d_state=16 × d_model=256 = 4096 floats vs LSTM's
    2 × 1024 × 2 = 4096 floats — similar memory but better information density.

2. Architecture
───────────────

    Observation (N features)
         │
         ▼
    MoE Feature Extractor (8 experts) → (expert_dim,) embedding
         │
         ▼
    ┌─────────────────────────────────────────────┐
    │  Mamba Block (replaces LSTM)                │
    │                                              │
    │  Input projection: expert_dim → d_model      │
    │  Mamba SSM:                                  │
    │    - d_model = 256                           │
    │    - d_state = 16 (hidden state dimension)   │
    │    - d_conv = 4  (local conv before SSM)     │
    │    - expand = 2  (inner dimension = 512)     │
    │  Output projection: d_model → pi_dim, vf_dim │
    │                                              │
    │  State: (d_model × d_state) = 4096 floats   │
    │  Carried across steps during inference.      │
    └─────────────────────────────────────────────┘
         │
         ├──► Policy head: pi_net → action distribution
         │
         └──► Value head:  vf_net → V(s)


3. SB3 Integration Strategy
───────────────────────────
The main challenge: sb3-contrib's RecurrentPPO expects hidden states as
(h_n, c_n) tuples stored in the rollout buffer. We need an adapter.

Option A — Subclass RecurrentActorCriticPolicy:
    - Override `_build_mlp_extractor()` to inject Mamba instead of LSTM
    - Override `forward()` and `predict()` for custom state handling
    - Create `MambaStateAdapter` that wraps the Mamba conv_state + ssm_state
      as a fake (h, c) tuple: h = ssm_state.view(layers, batch, -1),
      c = conv_state.view(layers, batch, -1)
    - Pros: Clean integration, uses SB3's existing rollout buffer
    - Cons: Fragile coupling to SB3 internal API

Option B — Custom policy from scratch:
    - Implement ActorCriticPolicy without inheriting from RecurrentActorCriticPolicy
    - Use Mamba natively without adapter hacks
    - Pros: Clean, no adapter needed
    - Cons: Must reimplement rollout sequence chunking, GAE computation, etc.

RECOMMENDATION: Option A with MambaStateAdapter. The SB3 API is stable enough
and the adapter is a thin wrapper (~50 lines).

4. MambaStateAdapter Sketch
───────────────────────────

    class MambaStateAdapter:
        \"\"\"Wraps Mamba internal state as (h, c) tuple for SB3 compatibility.\"\"\"

        def __init__(self, d_model: int, d_state: int, d_conv: int):
            self.d_model = d_model
            self.d_state = d_state
            self.d_conv = d_conv

        def to_lstm_state(self, ssm_state, conv_state):
            \"\"\"Pack Mamba state into (h, c) for rollout buffer storage.\"\"\"
            # ssm_state: (batch, d_model, d_state) → flatten to (1, batch, d_model * d_state)
            h = ssm_state.reshape(1, ssm_state.shape[0], -1)
            # conv_state: (batch, d_inner, d_conv) → flatten
            c = conv_state.reshape(1, conv_state.shape[0], -1)
            return (h, c)

        def from_lstm_state(self, h, c, batch_size):
            \"\"\"Unpack (h, c) back to Mamba state tensors.\"\"\"
            ssm_state = h.squeeze(0).reshape(batch_size, self.d_model, self.d_state)
            conv_state = c.squeeze(0).reshape(batch_size, -1, self.d_conv)
            return ssm_state, conv_state


5. FinMamba + MoE Fusion for Multi-Asset Portfolios
───────────────────────────────────────────────────

Target: Trade a portfolio of N stocks (e.g., S&P 500 components) with
cross-asset spatial awareness + temporal regime awareness.

Architecture:

    Per-asset OHLCV features (T bars × F features × N assets)
         │
         ▼
    MoE Feature Extractor (per-asset, shared weights)
    → regime-aware embeddings (N × expert_dim)
    [Handles: "WHEN" — temporal market regime routing]
         │
         ▼
    FinMamba Graph Attention Network
    → cross-asset attention with industry decay matrix
    [Handles: "WHAT" — cross-asset correlation structure]
         │
         ▼
    Multi-Level Mamba SSM (per-asset, shared weights)
    → temporal embeddings with ultra-long memory
    [Handles: "HOW LONG" — long-range temporal dependencies]
         │
         ▼
    Portfolio Signal Head
    → per-asset position scores → softmax → portfolio weights

Key design decisions:
  - MoE weights are shared across assets (parameter efficiency)
  - GAT adjacency comes from sector-industry decay matrix (prior) ×
    rolling correlation matrix (posterior), as in finmamba_arch.py
  - Mamba processes each asset's sequence independently (no cross-asset
    temporal mixing — that's the GAT's job)

6. Input Universality for Equities
──────────────────────────────────

Current BTC/ETH-specific features to replace:
  - fear_and_greed module → not available for equities
  - btc_dominance → crypto-specific
  - funding_rate → crypto perpetual futures only

Universal feature set (works for any asset class):
  - OHLCV (open, high, low, close, volume) — always available
  - Log returns (1h, 4h, 24h) — computable from close
  - Volatility (rolling 20/60 period) — computable from returns
  - Volume z-score (rolling) — computable from volume
  - ATR — computable from OHLCV
  - RSI, MACD — computable from close via pandas_ta
  - Market-wide: VIX (equities), DXY, sector ETF returns

Scaling: RobustScaler fitted per-asset on training data only (already
implemented in finmamba_signal.py, line 190). This handles outliers
better than StandardScaler for fat-tailed financial distributions.

7. Implementation Steps
───────────────────────

Step 1: Create models/mamba_policy.py
  - MambaStateAdapter class
  - MambaActorCriticPolicy(RecurrentActorCriticPolicy) with overrides
  - Unit test: forward pass with random obs, verify state shape

Step 2: Benchmark on current BTC/USDT data
  - Train Mamba policy vs LSTM policy on identical data/rewards
  - Compare: wall-clock time, val Sharpe, memory usage
  - Accept Mamba if: >= 0.9x LSTM Sharpe AND >= 1.5x training speed

Step 3: Multi-asset data pipeline
  - Extend utils/feature_pipeline.py (Phase 4) to handle N assets
  - Build sector-industry decay matrix from GICS codes
  - Compute rolling correlation matrix for posterior adjacency

Step 4: Integrate FinMamba GAT
  - Wire MoE → GAT → Mamba → Portfolio Signal
  - Graph sparsification via InceptionBlock (already in finmamba_arch.py)
  - Multi-Level Mamba (already in finmamba_arch.py, with fallback to LSTM)

Step 5: Production deployment
  - State persistence: serialize Mamba state to Redis between ticks
  - Ensure inference latency < 100ms for live trading

8. Dependencies
───────────────

  Required:
    pip install mamba-ssm        # GPU-accelerated Mamba (requires CUDA)
    pip install causal-conv1d    # Dependency of mamba-ssm

  Fallback:
    finmamba_arch.py already has LSTM fallback when mamba_ssm is unavailable
    (lines 7-11, 92-101). The same pattern should be used in mamba_policy.py.

9. Risk Assessment
──────────────────

  Low risk:
    - Mamba is well-tested in NLP/vision (Mamba-2, Jamba, etc.)
    - finmamba_arch.py already imports and uses it successfully

  Medium risk:
    - SB3 state adapter may break on SB3 version updates
    - Mamba training dynamics differ from LSTM — may need hyperparameter re-tune

  High risk:
    - Multi-asset portfolio allocation is a fundamentally harder problem than
      single-asset HOLD/LONG — needs extensive validation
    - Cross-asset GAT with N>100 stocks is O(N^2) in attention — may need
      graph sparsification (already in finmamba_arch.py but untested at scale)
"""
