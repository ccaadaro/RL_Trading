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

## Status
Background replay is running. Results will be tabulated here upon completion.
