# Experimental Log: Trend Capture Calibration

## Current Focus: Alpha Intensity vs. Execution Costs
We are investigating the optimal multiplier for the fast alpha signal to balance trend capture against transaction costs.

### Validation Study (In Progress)
- **Script**: `scripts/global_alpha_replay.py`
- **Levels**: `[0.3, 0.5, 0.7, 1.0]`
- **Success Criteria**:
    1. **Turnover**: Must remain < 2.5x monthly to avoid edge erosion.
    2. **Profitability**: Net BPS per trade must be significantly positive.
    3. **Gatekeeper Health**: Meta-model veto distribution must remain stable at higher alpha intensities.

### Hypotheses
- **0.3x (Previous Baseline)**: Too conservative; acts as a "wall" rather than a filter during rallies.
- **0.7x (Candidate)**: Should improve time-in-market during clear trends without disparately increasing turnover.
- **1.0x (Aggressive)**: May trigger the Meta-model's "over-trading" protections too frequently.

## Phase 8: Micro-Resolution Experiments (FAILED)
- **Status**: Completed (2026-04-28)
- **Hypothesis**: $2M Dollar Bars provide optimal SNR for alpha.
- **Results**:
  | Theta | Bars/Day | AUC (OOS) | Net BPS/Trade | TO (Monthly) | Verdict |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | $2M | 422 | 0.505 | - | 0.83x | FAILED (Cost Drag) |
  | $20M | 29 | - | - | - | ERROR (Window Collision) |
  | $50M | 11.5 | 0.521 | -24.6 | 60.4x | FAILED (Cost Drag) |
- **Conclusion**: Micro-resolution dollar bars ($2M-$50M) are non-viable for institutional capital due to extreme cost-drag (commissions + slippage).
- **Archive**: Tagged as `baseline_2m_failed_cost_drag`.

## Phase 9: Coarse Timeframe Pivot & Architectural Stabilization (IN PROGRESS)
- **Status**: Researching (2026-04-29)
- **Objective**: Pivot to $20M-$500M theta ranges and 1-hour candles to eliminate noise and cost-drag.
- **Key Changes**:
  - Implemented `MarketDataProvider` abstraction in `InstitutionalDollarStrategy.py`.
  - Stabilized `compute_triple_barrier.py` with adaptive `min_periods`.
  - Global migration from `oof_pred` to `alpha_prob`.
  - Launched battery: $20M, $50M, $100M, $200M, $500M.
- **Target Metric**: `net_bps_per_trade > 0`.
