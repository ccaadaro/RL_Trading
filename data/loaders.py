"""
Functions for loading market data from various sources.
"""
import pandas as pd
import numpy as np

from typing import Dict, List, Optional, Tuple

def load_candles(path: str) -> pd.DataFrame:
    """
    Load OHLCV candlestick data from JSON file.
    
    Args:
        path: Path to the JSON file with candle data
        
    Returns:
        DataFrame with OHLCV data, date as index
    """
    candles = pd.read_json(path, orient="records")
    candles.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    candles["date"] = pd.to_datetime(candles["timestamp"], unit="ms")
    candles.set_index("date", inplace=True)
    candles.drop(columns="timestamp", inplace=True)
    return candles

def load_trades(path: str) -> pd.DataFrame:
    """
    Load trade data from Feather file and aggregate to hourly.
    
    Args:
        path: Path to the Feather file with trade data
        
    Returns:
        DataFrame with processed trade data aggregated by hour
    """
    # Define columns and dtypes for efficient loading
    cols = ["timestamp", "side", "price", "amount", "cost"]
    dtypes = {"price": "float32", "amount": "float32", "cost": "float32"}

    # Load trades
    trades = pd.read_feather(path, columns=cols)
    trades = trades.astype(dtypes)
    trades["side"] = trades["side"].astype("category")
    trades["date"] = pd.to_datetime(trades["timestamp"], unit="ms")
    trades.set_index("date", inplace=True)

    # Add columns for buy/sell volumes and costs
    trades["buy_vol"] = np.where(trades["side"] == "buy", trades["amount"], 0.0)
    trades["sell_vol"] = np.where(trades["side"] == "sell", trades["amount"], 0.0)
    trades["buy_cost"] = np.where(trades["side"] == "buy", trades["cost"], 0.0)
    trades["sell_cost"] = np.where(trades["side"] == "sell", trades["cost"], 0.0)

    # Aggregate to 1h timeframe
    agg = {
        "amount": "sum",
        "buy_vol": "sum",
        "sell_vol": "sum", 
        "buy_cost": "sum",
        "sell_cost": "sum",
        "price": "mean",
        "side": "count"
    }

    trades_1h = (
        trades
        .groupby(pd.Grouper(freq="1h"))
        .agg(agg)
        .rename(columns={"side": "trades_n"})
    )
    
    # Add basic derived features
    trades_1h["vol_imbalance_feature"] = (
        trades_1h["buy_vol"] - trades_1h["sell_vol"]
    ) / (trades_1h["buy_vol"] + trades_1h["sell_vol"] + 1e-9)
    
    trades_1h["dollar_delta_feature"] = trades_1h["buy_cost"] - trades_1h["sell_cost"]
    
    return trades_1h


def merge_data(candles: pd.DataFrame, trades_1h: pd.DataFrame, fng_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge candles, trades and Fear & Greed data into a single DataFrame.
    
    Args:
        candles: OHLCV candle data
        trades_1h: Hourly aggregated trade data
        fng_df: Fear and Greed index data
        
    Returns:
        Combined DataFrame with all data sources
    """
    # Merge candles with trades
    df = candles.join(trades_1h, how="left")
    
    # Add Fear & Greed as a feature
    df["fng_feature"] = fng_df["value"].reindex(df.index, method="ffill").fillna(50)
    
    return df