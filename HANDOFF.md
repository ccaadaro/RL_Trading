# Agent Handoff: Institutional Alpha Pipeline (BTC/USDT)

## 1. Project Context
*   **System**: InstitutionalDollarStrategy (ZMQ-based shell for aggTrade bars).
*   **Asset**: BTC/USDT ($2M Dollar Bars).
*   **Goal**: Reach Net PnL > 0 after 0.05% fees + 0.02% slippage (Total 0.07% per side).
*   **Branch**: `feat/tv-indicators-features` (3 commits ahead of origin, not pushed yet).

## 2. Architecture Verdict (2026-04-28)

After a full audit session, the gate-based architecture is **structurally falsified**:

| Strategy | ROI | Trades | Net bps/trade |
|---|---|---|---|
| Buy & Hold | +276% | 0 | — |
| Alpha+Meta (default dyn gate) | +0.00% | 0 | — |
| Alpha+Meta (static gate) | -76.61% | 1589 | -1.7 |
| EMA50>200 trend | -23.87% | 970 | — |
| Regime-participation sizing | -79% | 4439 | — |

**Root cause**: 308k bars in 2 years = 422 bars/day. At 14bps/roundtrip, any
model making >30 trades/month needs AUC > 0.55 just to break even. Current
alpha (AUC OOS = 0.5104 with TB labels) produces -1.7 bps net/trade at $2M.

**Key bugs found and fixed in `scripts/global_alpha_replay.py`**:
- Feature window mismatch train vs inference (zscore win 200 vs 10000, etc.)
- HMM `fit_predict` in-sample leakage → walk-forward HMM now used
- Gate diagnostics: Institutional alpha NEVER exceeds 0.55 (max prob 0.61)

## 3. Current Experiment in Progress

**Coarse bar battery**: testing whether signal quality improves at $20M/$50M.

```bash
# Running as background process (PID ~524057)
python scripts/run_coarse_bar_battery.py \
  --thetas 20000000 50000000 --hold-days 3 --pt-sl 2.0

# Monitor
tail -f logs/battery_coarse_bars.log
```

Auto-calibrated parameters:
| theta | bars/day | daily_window | vertical_bars | cusum_span |
|---|---|---|---|---|
| $2M | 288 | 288 | 864 | 1152 |
| $20M | 29 | 29 | 86 | 115 |
| $50M | 12 | 12 | 35 | 46 |

**Decision gate**: net_bps > 0 AND TO < 50/month → only then proceed.

## 4. Next Steps (in strict order)

1. Wait for battery to finish: `cat logs/battery_coarse_bars.log`
2. Run `--eval-only` on the results to see the summary table.
3. **If $20M or $50M shows net_bps > 0**: retrain the full alpha there.
4. **If all still negative**: the alpha features themselves are the issue, not the resolution. Pivot to redesigning the target with more discriminative features (L2 orderbook depth, funding rate regime, cross-asset correlation regime).
5. **Do NOT** retrain at $2M. Do NOT tune the meta-model v1. Concluded architecturally infeasible.

## 5. File References

*   `scripts/global_alpha_replay.py`: Replay engine with benchmarks + gate diagnostics.
*   `scripts/regime_participation_replay.py`: Sizing-based alternative (tested, still fails at $2M).
*   `scripts/run_coarse_bar_battery.py`: Coarse bar experiment battery.
*   `scripts/build_dollar_bars.py`: Builds dollar bars from raw aggTrades.
*   `scripts/compute_triple_barrier.py`: CUSUM + Triple Barrier labeling.
*   `scripts/build_features_dollar.py`: Feature engineering on dollar bars.
*   `scripts/train_dollar_alpha.py`: LightGBM alpha model with uniqueness+recency weights.
*   `models/dollar_alpha_v1/latest_model.txt`: Elite v2.1 (21 features), AUC 0.5104.
*   `models/meta_model_v1/gatekeeper.txt`: Meta-model (do not continue investing in this).
*   `logs/battery_coarse_bars.log`: Live output of current battery run.

## 6. Literature grounding (from user analysis)

The regime-participation hypothesis is correct in principle but fails at $2M
because signal-flipping × cost dominates. The literature (López de Prado,
2020; CUSUM+TB in crypto) shows:
- Dollar bars are the WEAKEST sampler in crypto empirically; CUSUM+TB beats them.
- Next-bar labeling induces flips; triple barrier aligns target with economic decision.
- AUC 0.52 is not zero alpha; the question is whether it covers 14bps/RT cost.
- Economic value ≠ linear function of AUC. Measure net bps/trade, Calmar, capture ratio.

The answer from the battery determines the path forward.
