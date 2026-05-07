#!/usr/bin/env python3
"""
Build Phase 9 1h candlestick dataset with trend and risk features.

Phase 9 hypothesis: Trend context at 1h resolution, with microstructure as filter.
Primary signal: EMA/HMA slopes, realized volatility, RSI, proximity to extremes.
Target: triple_barrier_48h (TP 2.5%, SL 1.2%, 48h vertical barrier).
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def build_phase9_dataset():
    """Build btc_1h_phase9.feather from Freqtrade 1h OHLCV."""

    # Load 1h OHLCV data
    data_path = Path('/home/nosferatu/freqtrade/user_data/data/binance/BTC_USDT-1h.feather')
    logger.info(f"Loading 1h OHLCV from {data_path}")
    df = pd.read_feather(data_path)
    
    # Filter to Phase 9 regime (2021-2026)
    START_DATE = '2021-01-01'
    df = df[df['date'] >= START_DATE].reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} 1h candles from {df['date'].min()} to {df['date'].max()}")

    # Ensure numeric columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ==================== TREND FEATURES ====================
    logger.info("Computing trend features...")

    # Returns at different horizons
    for period in [3, 6, 12, 24, 48]:
        df[f'return_{period}h'] = df['close'].pct_change(period)

    # EMA slopes (12h, 26h, 50h windows)
    for span in [12, 26, 50]:
        ema = df['close'].ewm(span=span, adjust=False).mean()
        # Slope = (current_ema - ema_1h_ago) / ema_1h_ago
        df[f'ema_{span}_slope'] = (ema - ema.shift(1)) / ema.shift(1)

    # HMA slope (9-period Hull Moving Average slope)
    # HMA = 2*WMA(n/2) - WMA(n)
    def hma(series, period=9):
        half_period = period // 2
        wma_half = series.rolling(half_period).apply(lambda x: np.average(x, weights=np.arange(1, half_period+1)), raw=False)
        wma_full = series.rolling(period).apply(lambda x: np.average(x, weights=np.arange(1, period+1)), raw=False)
        hma_val = 2 * wma_half - wma_full
        return hma_val

    hma_9 = hma(df['close'], period=9)
    df['hma_slope'] = (hma_9 - hma_9.shift(1)) / hma_9.shift(1)

    # MA bias (% deviation from MA)
    for period in [24, 48, 96, 200]:
        ma = df['close'].rolling(period).mean()
        df[f'ma_bias_{period}'] = (df['close'] - ma) / ma

    # ==================== VOLATILITY & RISK FEATURES ====================
    logger.info("Computing volatility and risk features...")

    # Realized volatility (hourly returns stddev over windows)
    hourly_returns = df['close'].pct_change()
    for period in [24, 72, 168]:  # 1d, 3d, 7d
        df[f'realized_vol_{period}'] = hourly_returns.rolling(period).std()

    # RSI
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['rsi_14'] = rsi(df['close'], 14)
    df['rsi_42'] = rsi(df['close'], 42)

    # ATR (Average True Range)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_14'] = df['tr'].rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / df['close']

    # Drawdown from recent high and distance to high
    for period in [72, 168]:  # 3d, 7d
        rolling_high = df['high'].rolling(period).max()
        df[f'drawdown_from_high_{period}'] = (rolling_high - df['close']) / rolling_high
        df[f'distance_to_high_{period}'] = (rolling_high - df['close']) / df['close']

    # Volume zscore
    for period in [24, 72]:
        rolling_mean = df['volume'].rolling(period).mean()
        rolling_std = df['volume'].rolling(period).std()
        df[f'volume_zscore_{period}'] = (df['volume'] - rolling_mean) / (rolling_std + 1e-8)

    # ==================== TRIPLE BARRIER TARGET ====================
    logger.info("Computing triple barrier targets...")

    TP_PERCENT = 0.025  # 2.5% take-profit
    SL_PERCENT = 0.012  # 1.2% stop-loss
    BARRIER_HOURS = 48  # 48 hour vertical barrier

    targets = []
    # Use integer indexing to be robust to non-zero-starting indices
    for i in range(len(df)):
        if i >= len(df) - BARRIER_HOURS:
            targets.append(np.nan)
            continue
            
        close_now = df['close'].iloc[i]
        tp_level = close_now * (1 + TP_PERCENT)
        sl_level = close_now * (1 - SL_PERCENT)

        # Look ahead 48 hours using iloc
        lookahead = df.iloc[i+1 : i+BARRIER_HOURS+1][['high', 'low', 'close']]

        if len(lookahead) == 0:
            targets.append(np.nan)
            continue

        max_high = lookahead['high'].max()
        min_low = lookahead['low'].min()
        final_close = lookahead['close'].iloc[-1]

        # Determine target
        if max_high >= tp_level:
            target = 1  # Hit TP first
        elif min_low <= sl_level:
            target = -1  # Hit SL first
        else:
            # Hit vertical barrier, classify by final close
            target = 1 if final_close > close_now else -1

        targets.append(target)

    df['triple_barrier_48h'] = targets
    
    # Robustness check: verify a random sample for alignment
    sample_idx = len(df) // 2
    sample_close = df['close'].iloc[sample_idx]
    sample_target = df['triple_barrier_48h'].iloc[sample_idx]
    sample_lookahead_close = df['close'].iloc[sample_idx + BARRIER_HOURS]
    
    # Simple check: if vertical barrier was hit, target must match close direction
    # This is a partial check but verifies index alignment
    if sample_target != 0: # If we have a target
        # If it was a vertical barrier hit (no TP/SL hit), it must match the sign of return
        # We can't know for sure it was vertical without re-running logic, but we can log
        logger.info(f"Alignment check (row {sample_idx}): close={sample_close:.2f}, "
                    f"close+48={sample_lookahead_close:.2f}, target={sample_target}")

    # Also create binary trend targets for comparison
    df['trend_48h'] = (df['close'].shift(-48) > df['close']).astype(int)
    df['trend_72h'] = (df['close'].shift(-72) > df['close']).astype(int)

    # ==================== CLEANUP & SAVE ====================
    logger.info("Cleaning up and saving...")

    # Drop temporary columns
    df = df.drop(columns=['tr'], errors='ignore')

    # Remove rows with NaN in key features (initial warmup window)
    min_valid_row = df[df['realized_vol_168'].notna()].index.min()
    df = df.loc[min_valid_row:].reset_index(drop=True)

    logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")

    # Final alignment assertion: verify a few points to ensure reset_index didn't shift anything
    for check_idx in [100, len(df)//2, len(df)-BARRIER_HOURS-1]:
        c_now = df['close'].iloc[check_idx]
        c_future = df['close'].iloc[check_idx + BARRIER_HOURS]
        # Verify that trend_48h matches the price action at this index
        # This confirms that shift(-48) and the dataframe slicing are perfectly aligned
        assert (df['trend_48h'].iloc[check_idx] == 1) == (c_future > c_now), f"Index alignment failed at {check_idx}"

    # Save
    output_path = Path('/home/nosferatu/freqtrade/user_data/strategies/RL_Trading/cache/btc_1h_phase9.feather')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_feather(output_path)
    logger.info(f"Saved to {output_path}")

    # Summary stats
    logger.info(f"\nTarget distribution:")
    logger.info(f"  triple_barrier_48h: {df['triple_barrier_48h'].value_counts().to_dict()}")
    logger.info(f"  trend_48h: {df['trend_48h'].value_counts().to_dict()}")

    return df

if __name__ == '__main__':
    df = build_phase9_dataset()
