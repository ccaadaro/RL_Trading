# Phase 12: Risk-Managed BTC Exposure (Phase 12B)

## 1. Hypothesis
BTC directional alpha is too weak at retail latency, but volatility, liquidity stress, drawdown, and crowding signals may help modulate long exposure. A dynamic exposure policy may improve Calmar and reduce drawdowns versus Buy & Hold.

## 2. Dataset
- Source: 1h OHLCV (Freqtrade/Binance) + Exogenous (Funding, Basis)
- Timeframe: 1h
- History: 2021-01-01 to Present
- Key Features:
  - Realized Volatility (24h, 72h, 168h, 30d)
  - Drawdown from recent highs (30d, 90d, 180d)
  - Volatility Z-scores and Shocks
  - Funding/Basis Z-scores
  - Turbulence / Mahalanobis score

## 3. Candidate Policies
- **Strategy 0**: Buy & Hold (Exposure = 1.0)
- **Strategy 1**: Fixed Volatility Targeting (Exposure = Target_Vol / Realized_Vol)
- **Strategy 2**: Drawdown-Aware Vol Targeting (Reduce exposure as DD increases)
- **Strategy 3**: Volatility Shock De-risking (Reduce exposure on extreme vol spikes)
- **Strategy 4**: Liquidity/Stress Overlay (Exogenous features as risk gates)
- **Strategy 5**: ML Risk Model (Predicting tail risk/volatility, not direction)

## 4. Metrics
- Primary: Calmar Ratio, Max Drawdown
- Secondary: Net ROI, Sharpe, Sortino, Turnover, Capture Ratio vs B&H
- Operational: Time-in-market, Cost drag

## 5. Benchmarks
- Buy & Hold BTC
- Static 50/50 Allocation
- Random Exposure (matched time-in-market and turnover)
- Simple EMA/SMA filters

## 6. Kill Criteria
- If simple vol targeting cannot improve Calmar vs B&H, pause before ML.
- If rule-based risk overlays do not improve Calmar or drawdown, do not build ML.
- If ML risk model improves validation but fails observed test, archive.
- If turnover or cost drag becomes material, simplify.
- If any leakage sanity check fails, mark BLOCKED.

## 7. Validation Splits
- Chronological walk-forward validation.
- Initial Training: 2021-2023
- Validation: 2024
- Observed Test: 2025-Present
- All splits documented and leak-free (using purge/embargo).

## 8. No-Deployment Rule
No real capital deployment until a full dry-run period is completed and performance is verified against all success criteria.
