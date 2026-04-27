"""
Functions for feature engineering, data cleaning, and preprocessing.
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Union

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators as features to the DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Standard indicators
    df["atr_feature"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["volatility_20_feature"] = df["close"].pct_change().rolling(20, min_periods=1).std()
    
    # Calendar features
    df["hour_sin"] = np.sin(2*np.pi*df.index.hour/24)
    df["hour_cos"] = np.cos(2*np.pi*df.index.hour/24)
    df["dow_sin"] = np.sin(2*np.pi*df.index.dayofweek/7)
    df["dow_cos"] = np.cos(2*np.pi*df.index.dayofweek/7)
    
    # Volume-based indicators
    df["obv_feature"] = ta.obv(df["close"], df["volume"])
    df["vwap_feature"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"], length=14)
    df["mfi_feature"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
    df["cci_feature"] = ta.cci(df["high"], df["low"], df["close"], length=20)
    
    # Stochastic indicators
    sto = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    df["stoch_k_feature"] = sto["STOCHk_14_3_3"]
    df["stoch_d_feature"] = sto["STOCHd_14_3_3"]
    
    # Daily RSI resampled to hourly
    daily_rsi = ta.rsi(df["close"].resample("1D").last().ffill(), length=14)
    df["daily_rsi_feature"] = daily_rsi.reindex(df.index, method="ffill")
    
    # Return-based features
    df["ret_1h_feature"] = np.log(df["close"]).diff()
    df["ret_24h_feature"] = np.log(df["close"]).diff(24)
    
    # Z-score volume metrics by day
    vol_cols = ["volume", "amount", "buy_vol", "sell_vol", "buy_cost", "sell_cost"]
    for col in vol_cols:
        if col in df.columns:
            dz = df[col].groupby(df.index.date).transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9)
            )
            df[f"{col}_z_feature"] = dz.fillna(0)
    
    # Liquidity features
    df["spread_ratio"] = (df["high"] - df["low"])/df["volume"].rolling(24).mean()
    
    # Smart money indicators
    df["whale_flow"] = np.log1p(df["buy_cost"] - np.log1p(df["sell_cost"])).rolling(12).mean()
    
    # PPO (Percentage Price Oscillator)
    ppo_result = ta.ppo(df["close"])
    for col in ppo_result.columns:
        df[f"ppo_{col.lower()}_feature"] = ppo_result[col]
    
    # KST (Know Sure Thing)
    kst_result = ta.kst(df["close"])
    for col in kst_result.columns:
        df[f"kst_{col.lower()}_feature"] = kst_result[col]
    
    # Price-volume divergence
    df["price_vol_divergence_feature"] = (
        (df["close"].pct_change(5) > 0) & (df["volume"].pct_change(5) < 0)
    ).astype(int) - (
        (df["close"].pct_change(5) < 0) & (df["volume"].pct_change(5) > 0)
    ).astype(int)
    
    return df

def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market regime classification features.
    
    Args:
        df: DataFrame with price and volatility data
        
    Returns:
        DataFrame with added market regime features
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Market trend determination
    trend_up = df['close'] > df['close'].rolling(200, min_periods=1).mean()

    # Regime classification
    conditions = [
        trend_up & (df['volatility_20_feature'] > 0.03),   # bull + high vol
        trend_up & (df['volatility_20_feature'] <= 0.03),  # bull + low vol
        ~trend_up & (df['volatility_20_feature'] > 0.04),  # bear + high vol
        ~trend_up & (df['volatility_20_feature'] <= 0.04)  # bear + low vol
    ]
    choices = ['bull_high_vol', 'bull_low_vol',
               'bear_high_vol', 'bear_low_vol']

    # Create string regime labels
    df['market_regime'] = np.select(conditions, choices, default='neutral')

    # Define numeric regime codes
    regime_codes = {
        'bear_low_vol'  : 0,
        'bear_high_vol' : 1,
        'bull_low_vol'  : 2,
        'bull_high_vol' : 3,
        'neutral'       : -1
    }

    # Create numeric regime features
    df['market_regime_feature'] = (
        pd.Categorical(df['market_regime'],
                       categories=regime_codes.keys())
          .rename_categories(regime_codes)
          .astype('int8')
    )

    df['market_regime_code'] = (
        pd.Categorical(df['market_regime'], categories=regime_codes.keys())
          .rename_categories(regime_codes)
          .astype('int8')
    )

    # Drop the string column as it's not needed
    df.drop(columns='market_regime', inplace=True)
    
    return df

def add_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Add support and resistance features.
    
    Args:
        df: DataFrame with OHLCV data
        window: Window size for support/resistance detection
        
    Returns:
        DataFrame with support/resistance features
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Identify local highs/lows
    highs = df['high'].rolling(window=window, center=True).apply(
        lambda x: 1 if x.iloc[len(x)//2] == max(x) else 0)
    lows = df['low'].rolling(window=window, center=True).apply(
        lambda x: 1 if x.iloc[len(x)//2] == min(x) else 0)
    
    # Calculate proximity to support/resistance
    df['at_resistance_feature'] = highs.rolling(window).sum() / window
    df['at_support_feature'] = lows.rolling(window).sum() / window
    
    # Calculate position within price range
    close = df['close']
    upper_band = close.rolling(window).max()
    lower_band = close.rolling(window).min()
    
    df['range_position_feature'] = (close - lower_band) / (upper_band - lower_band + 1e-9)
    
    return df

def add_advanced_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add more sophisticated market regime features.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with advanced regime features
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Volatility regimes - multiple timeframes
    for window in [12, 36, 72]:
        vol = np.log(df['close']).diff().rolling(window).std()
        df[f'vol_regime_{window}h_feature'] = pd.qcut(
            vol, 5, labels=False, duplicates='drop').astype(float)
    
    # Trend strength using ADX
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx_feature'] = adx['ADX_14']
    df['trend_strength_feature'] = df['adx_feature'] / 100.0
    
    # Momentum regime
    mom = df['close'].pct_change(24)
    df['momentum_regime_feature'] = pd.qcut(
        mom.rolling(72).mean(), 5, labels=False, duplicates='drop').astype(float)
    
    # Volume regime
    rel_vol = df['volume'] / df['volume'].rolling(72).mean()
    df['volume_regime_feature'] = pd.qcut(
        rel_vol, 5, labels=False, duplicates='drop').astype(float)
    
    # Combine regimes into a composite feature
    df['composite_regime_feature'] = (
        df['market_regime_feature'] * 0.4 + 
        df['momentum_regime_feature'] * 0.3 + 
        df['volume_regime_feature'] * 0.2 + 
        df['vol_regime_72h_feature'] * 0.1
    )
    
    return df

def clean_dataframe(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Clean a DataFrame by handling NaN, inf values, and outliers.
    
    Args:
        df: DataFrame to clean
        cols: List of column names to clean
        
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    for col in cols:
        if col in df.columns:
            # Replace inf/-inf with NaN first
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Fill NaNs with column median for numeric features
            if df[col].dtype.kind in 'ifc':  # integer, float or complex
                median = df[col].median()
                # If median is NaN, use 0
                if pd.isna(median):
                    median = 0
                df[col] = df[col].fillna(median)
    
    return df