"""
Performance metrics calculation for RL trading strategies.

This module provides functions to evaluate trading performance through metrics
like Sharpe ratio, max drawdown, win rate, and regime-specific analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union


def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 8760) -> float:
    """
    Calculate the annualized Sharpe ratio.
    
    Args:
        returns: Array of returns (not cumulative)
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods in a year (default: 8760 for hourly data)
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Convert annual risk-free rate to per-period rate
    rf_per_period = ((1 + risk_free_rate) ** (1 / periods_per_year)) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_per_period
    
    # Calculate annualized Sharpe ratio
    sharpe = np.mean(excess_returns) / (np.std(excess_returns, ddof=1) + 1e-9)
    sharpe_annualized = sharpe * np.sqrt(periods_per_year)
    
    return sharpe_annualized


def calculate_sortino(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 8760) -> float:
    """
    Calculate the annualized Sortino ratio.
    
    Args:
        returns: Array of returns (not cumulative)
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods in a year (default: 8760 for hourly data)
        
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Convert annual risk-free rate to per-period rate
    rf_per_period = ((1 + risk_free_rate) ** (1 / periods_per_year)) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_per_period
    
    # Calculate downside deviation (only negative excess returns)
    neg_returns = excess_returns[excess_returns < 0]
    
    if len(neg_returns) == 0 or np.std(neg_returns, ddof=1) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    # Calculate downside deviation
    downside_dev = np.std(neg_returns, ddof=1)
    
    # Calculate annualized Sortino ratio
    sortino = np.mean(excess_returns) / (downside_dev + 1e-9)
    sortino_annualized = sortino * np.sqrt(periods_per_year)
    
    return sortino_annualized


def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int]:
    """
    Calculate maximum drawdown and its duration.
    
    Args:
        equity_curve: Array of cumulative equity values
        
    Returns:
        Tuple of (max_drawdown, max_drawdown_duration)
    """
    if len(equity_curve) < 2:
        return 0.0, 0
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown in percent terms
    drawdown = (equity_curve - running_max) / running_max
    
    # Find the maximum drawdown and its index
    max_dd = np.min(drawdown)
    max_dd_idx = np.argmin(drawdown)
    
    # Find when the drawdown starts
    start_idx = np.where(equity_curve[:max_dd_idx] == running_max[max_dd_idx])[0][-1]
    
    # Find when the drawdown ends (recovers to previous peak)
    # If it doesn't recover, use the last index
    end_idx = max_dd_idx
    recovery_idxs = np.where(equity_curve[max_dd_idx:] >= running_max[max_dd_idx])[0]
    if len(recovery_idxs) > 0:
        end_idx = max_dd_idx + recovery_idxs[0]
    
    dd_duration = end_idx - start_idx
    
    return max_dd, dd_duration


def analyze_regime_performance(history: Dict, regime_col: str = 'market_regime_code') -> Dict:
    """
    Analyze performance across different market regimes.
    
    Args:
        history: Dictionary with historical trading data
        regime_col: Column name for market regime code
        
    Returns:
        Dictionary with performance metrics by regime
    """
    # Extract relevant data
    equity = np.array(history['portfolio_valuation', :])
    regimes = np.array(history[regime_col, :])
    
    # Calculate returns
    returns = np.diff(equity) / equity[:-1]
    regimes = regimes[1:]  # Align with returns
    
    # Find unique regimes
    unique_regimes = np.unique(regimes)
    regime_performance = {}
    
    # Analyze each regime
    for regime in unique_regimes:
        regime_returns = returns[regimes == regime]
        
        if len(regime_returns) > 0:
            total_return = np.prod(1 + regime_returns) - 1
            sharpe = calculate_sharpe(regime_returns) if len(regime_returns) > 1 else 0.0
            regime_performance[int(regime)] = {
                'total_return': total_return,
                'sharpe': sharpe,
                'win_rate': np.mean(regime_returns > 0),
                'count': len(regime_returns)
            }
    
    # Convert numerical regime codes to named regimes
    regime_names = {
        0: 'bear_low_vol',
        1: 'bear_high_vol',
        2: 'bull_low_vol',
        3: 'bull_high_vol',
        -1: 'neutral'
    }
    
    named_performance = {}
    for code, perf in regime_performance.items():
        named_performance[regime_names.get(code, f'regime_{code}')] = perf
    
    return named_performance


def calculate_metrics(history: Dict, risk_free_rate: float = 0.0) -> Dict:
    """
    Calculate comprehensive trading performance metrics.
    
    Args:
        history: Dictionary with historical trading data
        risk_free_rate: Annual risk-free rate (default: 0.0)
        
    Returns:
        Dictionary with performance metrics
    """
    if not history:
        return {"error": "No history data provided"}
    
    # Extract equity curve
    equity = np.array(history['portfolio_valuation', :])
    positions = np.array(history['position_index', :])
    
    # Calculate returns
    returns = np.diff(equity) / equity[:-1]
    
    # Basic return metrics
    total_return = np.prod(1 + returns) - 1
    
    # Calculate drawdown metrics
    max_dd, max_dd_duration = calculate_max_drawdown(equity)
    
    # Calculate risk-adjusted metrics
    sharpe = calculate_sharpe(returns, risk_free_rate)
    sortino = calculate_sortino(returns, risk_free_rate)
    
    # Calculate trade metrics
    position_changes = np.diff(positions) != 0
    trade_returns = returns[position_changes[:-1]]
    
    if len(trade_returns) > 0:
        win_rate = np.mean(trade_returns > 0)
        profit_factor = -np.sum(trade_returns[trade_returns > 0]) / np.sum(trade_returns[trade_returns < 0]) if np.sum(trade_returns[trade_returns < 0]) < 0 else 0
    else:
        win_rate = 0
        profit_factor = 0
    
    # Analyze performance by regime
    regime_performance = analyze_regime_performance(history) if 'market_regime_code' in history else {}
    
    # Combine metrics
    metrics = {
        "total_return": total_return,
        "annualized_return": (1 + total_return) ** (8760 / len(returns)) - 1,
        "annualized_sharpe": sharpe,
        "annualized_sortino": sortino,
        "max_drawdown": max_dd,
        "max_dd_duration": max_dd_duration,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": np.sum(position_changes),
        "regime_performance": regime_performance
    }
    
    return metrics