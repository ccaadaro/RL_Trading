# Institutional Pipeline Architecture

## System Overview
The strategy operates as a hybrid real-time inference system designed for institutional-grade execution on high-frequency Dollar Bars ($2,000,000 notional).

## Data Ingestion
- **Source**: `market_data_daemon.py`
- **Transport**: ZMQ PUB/SUB (IPC/TCP)
- **Topics**: `DOLLAR_BAR`, `BOOK_TICKER`

## Intelligence Layer (The "Brain")
1. **Alpha Models**: 
   - Primary: LightGBM trained on microstructure features (CVD, Aggressor Ratio).
   - Ensemble: Blended fast (1m-equivalent) and slow (1h-equivalent) signals.
2. **Risk Engines**:
   - **Mahalanobis Turbulence**: Measures multivariate regime rarity.
   - **HMM (Hidden Markov Model)**: Latent state detection with canonical volatility mapping (0: Calm, 1: Neutral, 2: High Vol).
3. **Meta-Model Gatekeeper**: 
   - A secondary LightGBM vette specifically trained to filter alpha signals based on current regime, expected costs (bps), and book imbalance.

## Execution Shell
- **Freqtrade Strategy**: `InstitutionalDollarStrategy.py`
- **Watchdog**: Ensures the ZMQ listener thread is healthy.
- **Clock Decoupling**: Uses a Symmetric CUSUM filter to only rebalance on "Information Arrival" events, minimizing turnover.
