# Institutional BTC Trading — Research Repository

A systematic, forensic research project investigating directional alpha and risk-managed BTC/USDT exposure. Twelve research phases were executed under strict pre-committed performance gates (AUC, Calmar ratio, Net ROI vs. Random P95). **All phases were falsified. The project is archived.**

---

## Verdict

> No stable directional alpha or risk-adjusted BTC exposure strategy survived leakage-safe, cost-realistic walk-forward validation at 14 bps round-trip cost. Buy-and-hold remained unbeaten across all regimes and horizons tested.

---

## Repository Structure

```
.
├── InstitutionalDollarStrategy.py  # Main Freqtrade strategy (archived, not deployable)
├── scripts/                        # Data pipeline, training, and evaluation scripts
│   ├── build_*.py                  # Dataset construction (dollar bars, OHLCV, exogenous)
│   ├── train_*.py                  # Model training (LightGBM, RL, ensemble stacking)
│   ├── evaluate_*.py               # Walk-forward evaluation per phase
│   ├── download_*.py               # Raw data acquisition (Binance agg-trades, futures)
│   ├── monitor_*.py                # Live shadow-mode monitoring
│   └── *.sh                        # Shell launch helpers
├── utils/                          # Shared library (signal features, risk, position sizing)
│   ├── signal_features.py
│   ├── risk_directors.py           # HMM regime model, Mahalanobis turbulence
│   ├── position_sizer.py           # Fractional Kelly sizing
│   ├── filters.py                  # Symmetric CUSUM filter
│   └── data_providers.py           # ZMQ dollar-bar + Freqtrade candle providers
└── docs/                           # Research notes, phase reports, architecture docs
    ├── PHASE12_RISK_MANAGED_BTC.md
    ├── MODEL3_PROTOCOLS.md
    ├── PIPELINE_ARCHITECTURE.md
    ├── EXPERIMENTAL_LOG.md
    └── ...
```

---

## Research Phases

| Phase | Hypothesis | Signal | Verdict | Key Finding |
|-------|-----------|--------|---------|-------------|
| 8 | $50M dollar-bar microstructure alpha | CVD, aggressor ratio, whale order flow | **FAIL** | Predictive at ms latency; signal collapses at bar-completion (~1–2 min) |
| 9 | 1h trend-following OHLCV | EMA/HMA slopes, realized volatility | **FAIL** | Initial AUC lift was data leakage; corrected evaluation found no edge |
| 10 | Exogenous market structure | Perp-spot basis, funding rates | **FAIL** | Basis carries weak directional info; consumed by execution costs |
| 11 | Horizon shift (7-day holds) | All prior signals, longer window | **FAIL** | Longer horizon does not rescue weak signals through cost dilution |
| 12 | Risk-managed BTC exposure | Volatility targeting, drawdown scaling | **FAIL** | Drawdown cut 30–33%; underperforms buy-and-hold in bull regimes |

### Methodology

- **Walk-forward validation**: 4-fold temporal separation with strict purging and embargo windows.
- **Leakage forensics**: Comprehensive timestamp and causality checks on all exogenous features.
- **Cost model**: 14 bps round-trip, integrated into all P&L calculations.
- **Kill criteria**: Pre-committed before each phase (no post-hoc gate adjustment).

---

## Strategy Architecture (Archived)

`InstitutionalDollarStrategy.py` is a **thin Freqtrade shell**. All intelligence runs outside the polling loop:

```
market_data_daemon.py ──ZMQ PUB──► InstitutionalDollarStrategy
                                         └── _ZmqListener (background thread)
                                                  │
                                          on each Dollar Bar:
                                          Features → LightGBM → HMM → Kelly sizing
                                                  │
                                          _latest_signal (thread-safe dict)
                                                  │
                                         populate_entry/exit_trend()
                                                  │
                                         Freqtrade order routing
```

**Not compatible with native Freqtrade backtesting.** Designed for `--dry-run` or live use only.

### Core Modules

| Module | Role |
|--------|------|
| `utils/signal_features.py` | OHLCV feature engineering |
| `utils/risk_directors.py` | HMM regime detection, Mahalanobis turbulence filter |
| `utils/position_sizer.py` | Fractional Kelly position sizing |
| `utils/filters.py` | Symmetric CUSUM entry filter |
| `utils/data_providers.py` | ZMQ dollar-bar ingestion + Freqtrade candle fallback |

---

## Running the Research Pipeline

Each phase follows the same three-step pattern:

```bash
# 1. Build dataset
python scripts/build_dollar_bars.py          # Phase 8 (microstructure)
python scripts/build_1h_dataset.py           # Phase 9 (trend)
python scripts/build_model3_exogenous_dataset.py  # Phase 10 (exogenous)

# 2. Train model
python scripts/train_dollar_alpha.py
python scripts/train_signal_walkforward.py

# 3. Evaluate
python scripts/evaluate_walkforward.py
python scripts/evaluate_phase12_risk_managed_btc.py
```

Data acquisition (requires Binance API or local data):
```bash
python scripts/download_binance_futures_data.py
python scripts/download_aggtrades.py
python scripts/process_aggtrades_to_bars.py
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Trading framework | [Freqtrade](https://www.freqtrade.io/) |
| ML models | LightGBM, Scikit-learn |
| RL training | CleanRL, Stable-Baselines3 (SAC, TQC) |
| Regime detection | hmmlearn (Hidden Markov Models) |
| Data processing | Pandas, NumPy, Feather, Parquet |
| Real-time transport | ZeroMQ (ZMQ PUB/SUB) |
| Hyperparameter search | Optuna |
| Language | Python 3.12 |

---

## Disclaimer

This repository is for research and educational purposes only. Nothing here constitutes financial advice. Trading cryptocurrencies carries significant risk of loss. Past research results do not indicate future performance.
