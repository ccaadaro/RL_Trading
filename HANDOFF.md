# Agent Handoff: Institutional Alpha Pipeline (BTC/USDT)

## 1. Project Context
*   **System**: InstitutionalDollarStrategy (ZMQ-based shell for aggTrade bars).
*   **Asset**: BTC/USDT ($2M Dollar Bars).
*   **Goal**: Reach Net PnL > 0 after 0.05% fees + 0.02% slippage (Total 0.07% per side).
*   **Branch**: `feat/tv-indicators-features` (Already pushed).

## 2. Model Status
*   **Alpha v2.1 "Elite" (Model C)**: 21 features (Microstructure + WVF + %R). AUC OOS: 0.5176.
*   **Institutional Base (Model A)**: 14 features (Solo Microestructura).
*   **Validation Results**:
    *   **Audit Fix**: Corrected a scale error in `global_alpha_replay.py`. Real monthly turnover is ~40x (not 2.24x).
    *   **Alpha Noise**: The model captures +15% gross alpha, but pays -67% in costs. Expected bps per trade is < 7bps hurdle.
    *   **Conclusion**: Pure directional alpha is not enough. We need a "Gatekeeper".

## 3. Work in Progress: Meta-Model Gatekeeper
We implemented a binary classifier (LightGBM) to veto Alpha signals based on market context.

*   **Logic**: `y_meta = 1` if `forward_return_50_bars > 2 * costs`.
*   **Results (OOS Evaluation)**:
    *   **AUC**: 0.9827 (Exceptionally high, likely due to regime/volatility predictive power).
    *   **Precision (Tradeability)**: 95.02%.
    *   **Net Bps Improvement**: Average net bps increased from **289 bps** (all alpha signals) to **2436 bps** (meta-filtered).
    *   **Note**: High values are likely due to clustering during volatility expansions.
*   **Current State**:
    *   `scripts/generate_metamodel_data.py`: **COMPLETED**.
    *   `scripts/train_metamodel.py`: **COMPLETED**. Model saved at `models/meta_model_v1/gatekeeper.txt`.

## 4. Next Recommended Steps
1.  **Finish Data Generation**: Wait for `scripts/generate_metamodel_data.py` to complete.
2.  **Train Meta-Model**: Run `python scripts/train_metamodel.py`.
3.  **Evaluate Net Bps**: Ensure `Avg Net Bps (Meta-Filtered) > 0` in the OOS set.
4.  **Integrate**: Modify `InstitutionalDollarStrategy.py` to check `meta_model.predict()` before entry.
5.  **Final Validation**: Run `global_alpha_replay.py` with the combined logic.

## 5. File References
*   `InstitutionalDollarStrategy.py`: Main strategy shell.
*   `scripts/global_alpha_replay.py`: Validation engine (corrected logic).
*   `utils/signal_features.py`: Feature sets definitions.
*   `utils/risk_directors.py`: HMM & Turbulence implementation.
*   `reports/`: Equity curves with drawdown shading.
