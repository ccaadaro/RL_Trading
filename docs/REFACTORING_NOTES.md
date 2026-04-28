# Refactoring & Stabilization Notes (2026-04-28)

## Critical Stability Fixes
1. **Rolling Window Collision**:
   - **Issue**: `ValueError` when calculating daily volatility due to a 29-bar window vs 50-bar `min_periods`.
   - **Fix**: Set `MIN_BARS_FOR_INFERENCE = 50` and defined `_WINDOW_DAILY = 1000`.
2. **Constructor Synchronization**:
   - **Issue**: `TypeError` in `_ZmqListener` due to mismatched arguments in `bot_start`.
   - **Fix**: Synchronized signature to include `alpha_slow_model` and initialized internal state attributes.

## Semantic Refactoring
### Transition: `oof_pred` → `alpha_prob`
- **Context**: The term `oof_pred` (Out-of-Fold) was legacy from the training pipeline. 
- **Change**: Renamed to `alpha_prob` (or `blended_alpha_prob`) to correctly represent the real-time nature of the signal during live inference.
- **Coverage**: Applied to `InstitutionalDollarStrategy.py` and all validation scripts.

## Logic Adjustments
- **Signal Multiplier**: Increased from 0.3x to 0.7x (experimental) to improve trend capture. 0.3x was identified as a bottleneck that "strangled" high-conviction entries.
