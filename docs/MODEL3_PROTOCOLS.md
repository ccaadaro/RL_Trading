# MODEL 3 PROTOCOLS — Phase 10: Exogenous Alpha

_Version 1.0 — 2026-05-07_

---

## Status Summary

| Phase | Hypothesis | Gate | Result | Decision |
|-------|-----------|------|--------|----------|
| Phase 8 | $50M dollar-bar microstructure-only alpha | AUC > 0.55 across 4 folds | FAIL — AUC ≈ 0.505, 3/4 folds zero entries | Archive. Microstructure at bar-completion is not alpha. |
| Phase 9 | 1h OHLCV trend-only LightGBM | AUC > 0.55, Calmar > B&H after costs | FAIL — corrected AUC ≈ 0.54, returns collapse to −81% with costs | Archive. Trend-only OHLCV insufficient for cost-adjusted alpha. |
| **Phase 10** | Exogenous alpha: funding, basis, OI | AUC ≥ 0.53, Net ROI > Random P95, Calmar > B&H | **ARCHIVED** — funding AUC 0.506, basis AUC 0.535 (1 gate pass, 3 fail). No economic edge post-friction. | Phase 10 closed. Proceed to Phase 11. |
| **Phase 11** | Horizon shift: daily candles, 7d holds | AUC ≥ 0.52, economic edge after costs | **NEXT** | Activate after Phase 10 archive. |
| Phase 12 | Risk-managed BTC exposure (Rule-based) | Calmar > B&H AND DD reduction ≥ 25% | **ARCHIVED** — Best Calmar 1.18 vs B&H 1.61 (Daily Rebalance) | Phase 12B closed. BTC P&L Project TERMINATED. |

---

## Phase 10 — Model 3: Exogenous Alpha

### Core Hypothesis

Funding rate, perp-spot basis, and open interest delta carry predictive information for BTC/USDT price direction that is **not present** in 1h OHLCV or simple microstructure features.

Rationale: These are derivatives of market positioning, not price itself. They reflect structural imbalances (crowded longs, carry pressure, leverage buildup) that may resolve directionally with some lag — a lag exploitable at hourly resolution without HFT-grade infrastructure.

### Kill Criteria (Pre-Committed — Must Not Change After Seeing Results)

A model is archived if **any** of the following are true:
- mean AUC < 0.53
- min fold AUC < 0.51
- Net ROI ≤ Random P95
- Calmar ≤ Buy & Hold Calmar
- shuffled-label AUC materially above 0.50
- any negative timestamp lag (count_negative_lag > 0)
- single-feature AUC > 0.85 (treat as leakage, audit before proceeding)

If Model 3 fails: proceed to Phase 11.

---

## Feature Family Build Order

Features are added **one family at a time**. Each family must independently pass the timestamp audit and sanity checks before the next is added.

### Family 1: Funding Rate (CURRENT)

**Source**: Binance FAPI — 8h settlement cycle  
**Data file**: `BTC_USDT_USDT-8h-funding_rate.feather`  
**Cache output**: `cache/btc_1h_phase10_funding.feather`

**Timestamp safety rule**: For each 1h bar at time `t`, use only the last funding value published **strictly before** `t`. Implementation: `merge_asof` with `direction='backward'`.

**Features**:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `funding_last` | Last funding rate before bar open | Raw positioning pressure |
| `funding_8h_zscore_30d` | (funding_last − μ_30d) / σ_30d | Normalized over 30d rolling window |
| `funding_8h_zscore_90d` | (funding_last − μ_90d) / σ_90d | Normalized over 90d rolling window |
| `funding_abs_zscore_30d` | abs(funding_8h_zscore_30d) | Extremity regardless of sign |
| `funding_sign` | sign(funding_last) | Direction indicator |

**Interpretation**:
- Extreme positive funding → crowded longs → possible mean reversion
- Extreme negative funding → forced bearishness → possible squeeze
- Direction is **not hardcoded** — model learns from data

**Timestamp Audit Requirement**:

| Column | Threshold |
|--------|-----------|
| count_negative_lag | Must be exactly 0 |
| min_lag | Must be > 0 seconds |
| p50_lag (seconds) | Report only |
| p95_lag (seconds) | Report only |

---

### Family 2: Perp-Spot Basis (PENDING — awaits Family 1 result)

**Source**: `BTC_USDT_USDT-8h-mark.feather` (perp mark) + `BTC_USDT-1h.feather` (spot)

**Features**:

| Feature | Formula |
|---------|---------|
| `basis_now` | (perp_mark_close − spot_close) / spot_close |
| `basis_zscore_30d` | rolling z-score of basis_now |
| `basis_mean_24h` | 24h rolling mean of basis_now |
| `basis_change_24h` | basis_now − basis_now_24h_ago |
| `basis_compression` | basis_now − basis_mean_24h |

Only activate if Family 1 (funding) shows AUC ≥ 0.51 on its own.

---

### Family 3: Open Interest Delta (PENDING)

**Source**: `BTC_USDT_USDT-4h-open_interest.parquet`

**Note**: Binance retains only ~28 days of OI history via the FAPI endpoint. The current cached file spans 2026-02-18 to 2026-04-11 (~52 days). This is **insufficient** for 4-fold walk-forward training. Options:
1. Download OI from alternative source (CoinGlass, Glassnode)
2. Use a shorter backtest window — but this reduces fold quality
3. **Decision**: OI will only be evaluated if an alternative historical source can be found with ≥ 2 years of 1h or 4h data.

**Features** (pending data availability):

| Feature | Formula | Interpretation |
|---------|---------|---------------|
| `oi_change_1h` | oi_now − oi_1h_ago (normalized) | Short-term OI flow |
| `oi_change_8h` | oi_now − oi_8h_ago (normalized) | Session-level OI shift |
| `oi_change_24h_zscore` | z-score of 24h OI change | Extreme OI accumulation |
| `oi_price_confirm` | sign(ret_24h) × oi_change_24h_zscore | Leveraged trend vs. squeeze |
| `oi_divergence` | ret_24h_zscore − oi_change_24h_zscore | Price/OI divergence |

---

### Family 4: Options Skew (DEFERRED)

Only attempt after Families 1-3 show positive AUC signals. Requires expiry selection, delta interpolation, stale quote handling — expensive to build correctly. Defer unless needed.

---

## Mandatory Sanity Checks Per Family

Before training any model, each feature family must pass ALL of the following:

1. **Timestamp Audit**: count_negative_lag = 0
2. **Single-Feature AUC Scan**: Each feature must have AUC < 0.65 (> 0.85 = immediate leakage audit)
3. **Correlation with Target**: |corr| < 0.5
4. **Noise Control**: At least one `noise_control` (shuffled) column added. Must not rank as important.
5. **Shuffled-Label Test**: AUC with shuffled y must collapse to ≈ 0.50 (±0.01)
6. **Lag-All-Features Test**: Shifting all features by 1 bar must degrade AUC

---

## Evaluation Configuration

### Target
Primary: `triple_barrier_48h` (TP=2.5%, SL=1.2%, vertical=48h) — inherits from best surviving Phase 9 target.

Secondary (only if primary fails and there is economic reason): `trend_48h`, `trend_72h`, `target_barrier_3.5tp_1.5sl_72h`

**Rule**: Target may not be selected based on AUC alone. Selection must be economic.

### Model Configuration

```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.03,
    'n_estimators': 300,   # in [200, 400]
    'max_depth': 3,         # in [2, 3]
    'num_leaves': 15,       # in [7, 15]
    'min_data_in_leaf': 100,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 1.0,
    'verbose': -1,
}
```

**Rule**: No hyperparameter tuning after seeing validation results.

### Walk-Forward Configuration

- Splits: Strict chronological, **no shuffle**
- Folds: 4
- Purge window: **72 bars** (3 days, exceeds 48h target horizon)
- Embargo window: **72 bars**
- Expanding training window

### Benchmarks

Every model result must be compared against:
1. Cash / flat (0% return)
2. Buy & Hold
3. Always Long
4. EMA/HMA trend baseline
5. Random matched time-in-market (P50 and P95)
6. Random matched trade count (P50 and P95)

### Required Output Metrics

Per fold and aggregated (mean ± std):
- AUC, Brier score
- ROI gross, ROI net
- Max drawdown
- Calmar, Sharpe
- Turnover, trades/month, time-in-market
- Net bps/trade, gross bps/trade, cost drag
- Capture ratio vs B&H
- Model ROI vs Random P95
- Feature importance stability (std / mean per feature)

---

## Acceptance Gate

A model **passes** only if all of the following are simultaneously true:

| Criterion | Threshold |
|----------|-----------|
| mean AUC | ≥ 0.53 |
| min fold AUC | ≥ 0.51 |
| Net ROI | > Random P95 |
| Calmar | > Buy & Hold Calmar |
| turnover | Acceptable (< 2.5×/month typical) |
| No single fold explains all returns | Verified |
| No leakage test fails | Verified |

---

## Phase 11 — Horizon Shift (PENDING)

**Hypothesis**: Alpha may exist but cost drag at 1h rebalance is too hostile for weak signals.

**Configuration**:
- Daily candles
- 7-day holds
- ~25 bps round-trip
- Same exogenous features
- Lower decision frequency

**Kill criterion**: mean AUC < 0.52, or no economic edge after costs and Random P95.

---

## Phase 12 — Strategic Decision (TERMINAL)

**Status**: [PROJECT CLOSED] - 2026-05-08

If Phase 10 and Phase 11 both fail: **no Phase 13 exists**.

**Final Result**: Rule-based de-risking (Phase 12B) failed to beat the Buy & Hold Calmar ratio even with daily rebalancing. The "cost of protection" exceeds the drawdown benefit on a risk-adjusted basis.

**RESEARCH CLOSED — NOT DEPLOYABLE**

Three honest options:
- **A**: Different instrument (Cross-sectional crypto factor ranking across 50+ altcoins)
- **C**: Stop — treat as successful infrastructure and research project

---

## Research Ledger

### Phase 8
- **Hypothesis**: $50M dollar-bar microstructure-only alpha
- **Gate**: AUC > 0.55 across 4-fold WF
- **Result**: AUC ≈ 0.505; 3/4 folds had zero entries; 2-fold was sampling artifact
- **Decision**: FAIL — archive. Microstructure features not sufficient as primary alpha at bar-completion latency.

### Phase 9
- **Hypothesis**: 1h OHLCV trend-only LightGBM ≥ 0.55 AUC, positive Calmar after costs
- **Gate**: AUC ≥ 0.55, Net ROI > B&H, Calmar > 0.5
- **Result**: After leakage audit: corrected AUC ≈ 0.54 (below gate). Returns collapse from +101% to −81% after costs. 2021–2026 regime truncation pushed AUC below 0.55.
- **Decision**: FAIL — archive. Trend-only OHLCV provides insufficient edge post-friction.

### Phase 10 — Model 3: Exogenous Alpha
- **Hypothesis**: Funding rate, perp-spot basis, and OI carry independent directional signal
- **Gate**: mean AUC ≥ 0.53, min fold AUC ≥ 0.51, Net ROI > Random P95, Calmar > B&H
- **Result (Funding-only, 2026-05-07)**: FAIL — mean AUC 0.506 ± 0.020; min fold 0.493; net ROI −7.1% vs Random P95 +45.6%; Calmar 0.003 vs B&H 1.77. All 4 gates failed. Sanity checks clean (shuffled-label AUC 0.493, no timestamp leakage, single-feature AUCs all < 0.54).
- **Result (Basis-only, 2026-05-07)**: FAIL — mean AUC 0.535 ± 0.020; min fold 0.509; net ROI −0.8% vs Random P95 +41.5%; Calmar 0.169 vs B&H 1.77. AUC gate marginal pass (0.535 > 0.53) but 3/4 gates failed. The model cannot generate positive net ROI after costs despite directional signal. Timestamp audit clean (min_lag=28,800s, count_negative_lag=0). Shuffled-label AUC 0.511, no structural leakage.
- **OI data gap**: Binance FAPI retains only 28 days of OI history. Cached file spans 2026-02-18 → 2026-04-11. Insufficient for training. Requires CoinGlass or Glassnode for historical OI.
- **Decision**: Phase 10 ARCHIVED. Both funding and basis fail the economic gates. The weak directional signal in basis (AUC ~0.53) is consumed entirely by execution costs at 1h resolution. Noise control ranked below all real features in both runs — models are stable, not leaking. Proceeding to Phase 11 (Horizon Shift).
