# Institutional BTC Trading Strategy and Research

A professional quantitative trading research repository focused on BTC/USDT alpha generation and risk-managed exposure. This project encompasses a multi-phase research pipeline evaluating microstructure signals, trend-following models, and exogenous market-structure variables.

## Project Overview

The repository contains the InstitutionalDollarStrategy, a production-grade Freqtrade strategy, along with a comprehensive suite of research scripts for data engineering, model evaluation, and backtesting. The project transitioned through 12 research phases, systematically testing and falsifying various alpha hypotheses to identify stable, tradeable signals in modern crypto markets.

## Research Phases and Results

The project followed a rigorous, forensic research protocol with pre-committed performance gates (AUC, Calmar, Net ROI vs. Random P95).

### Phase 8: Microstructure-Only Alpha (Falsified)
Tested $50M dollar-bar microstructure signals (CVD, aggressor ratio, whale order flow). Concluded that microstructure is predictive at the millisecond level but lacks sufficient signal-to-noise ratio at bar-completion latency (~1-2 min) to cover execution costs.

### Phase 9: Trend-Following OHLCV (Falsified)
Evaluated 1h candle trend features (EMA/HMA slopes, realized volatility). Initial high performance was identified as data leakage. Corrected evaluation showed that trend-only alpha is insufficient to cover trading frictions at hourly resolution.

### Phase 10: Exogenous Alpha (Falsified)
Investigated perp-spot basis and funding rates. While basis carries weak directional information, the signal is consumed by execution costs at high rebalancing frequencies.

### Phase 11: Horizon Shift (Falsified)
Tested longer holding horizons (7-day holds) to reduce turnover and cost drag. The hypothesis that a longer horizon would make weak signals tradeable was not supported by cost-realistic walk-forward validation.

### Phase 12: Risk-Managed BTC Exposure (Terminal)
Pivoted to dynamic exposure management (volatility targeting, drawdown-aware scaling). Successfully reduced max drawdown by 30-33%, providing significant bear protection, though Buy and Hold remains the superior strategy in strong bull regimes.

## Technical Architecture

### Core Components
- InstitutionalDollarStrategy: Main Freqtrade strategy implementing the signal engine and risk controls.
- Market Data Daemon: ZMQ-based service for real-time microstructure data ingestion.
- Signal Engine: Regime-aware processor using Hidden Markov Models (HMM) for market context.
- Telemetry Pipeline: High-performance ZMQ PUB/SUB for real-time monitoring and diagnostics.

### Research Framework
- Automated Walk-Forward Validation: 4-fold temporal separation with strict purging and embargoing to prevent leakage.
- Forensic Auditing: Comprehensive timestamp and causality checks for all exogenous features.
- Realistic Simulation: Integrated transaction costs (14bps round-trip) and execution latency modeling.

## Current Status

Research Closed. Across all phases, no stable directional alpha or risk-adjusted exposure strategy surpassed the pre-committed institutional gates after leakage-safe, cost-realistic evaluation. The project is currently in archival status.

## Tech Stack

- Core Logic: Python 3.x
- Trading Framework: Freqtrade
- Machine Learning: LightGBM, Scikit-learn
- Data Processing: Pandas, NumPy, Feather
- Networking: ZeroMQ (ZMQ)
- Optimization: Optuna

## Disclaimer

This repository is for research and educational purposes only. Nothing in this project constitutes financial advice. Trading cryptocurrencies involves significant risk of loss.
