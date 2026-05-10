#!/usr/bin/env python3
"""
Evaluate Phase 12 Risk-Managed BTC Exposure on walk-forward (HARDENED).

Project Objective: Transform from "directional alpha" to "dynamic BTC exposure management".
Goal: Beat Buy & Hold on risk-adjusted metrics (Calmar, Max Drawdown).

Strategies:
0. Buy & Hold baseline
1. Fixed volatility targeting
2. Drawdown-aware volatility targeting
3. Volatility shock de-risking (rule-based)
4. Liquidity/stress overlay (rule-based)
5. Static 50% Exposure
6. Cash (0%)
7. Random Exposure (Matched Turnover)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add project root to sys.path to import utils
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
CACHE_DIR = ROOT / "cache"
REPORTS_DIR = ROOT / "reports"
EQUITY_CURVES_DIR = REPORTS_DIR / "phase12_equity_curves"

PHASE9_DATA = CACHE_DIR / "btc_1h_phase9.feather"
FUNDING_DATA = CACHE_DIR / "btc_1h_phase10_funding.feather"
BASIS_DATA = CACHE_DIR / "btc_1h_phase10_basis.feather"

def load_and_merge_data():
    """Load and merge Phase 9, Funding, and Basis data."""
    logger.info("Loading and merging datasets...")
    if not PHASE9_DATA.exists():
        logger.error(f"Base data missing: {PHASE9_DATA}")
        sys.exit(1)
        
    df = pd.read_feather(PHASE9_DATA)
    
    if FUNDING_DATA.exists():
        df_funding = pd.read_feather(FUNDING_DATA)
        funding_cols = [c for c in df_funding.columns if "funding" in c and c not in df.columns]
        df = pd.merge(df, df_funding[['date'] + funding_cols], on='date', how='left')
        
    if BASIS_DATA.exists():
        df_basis = pd.read_feather(BASIS_DATA)
        basis_cols = [c for c in df_basis.columns if "basis" in c and c not in df.columns]
        df = pd.merge(df, df_basis[['date'] + basis_cols], on='date', how='left')
        
    # Ensure chronological order
    df = df.sort_values('date').reset_index(drop=True)
    return df

def compute_metrics(returns_slice, full_positions, test_start, test_end, cost_bps=7.0):
    """Compute risk/return metrics with transaction costs and proper boundary handling."""
    ret_clean = returns_slice.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Position changes and costs
    # Action at close of t-1 applies to return of t
    # To get pos[test_start], we need full_pos[test_start - 1]
    
    # Extract positions including one bar of context for the shift
    # If test_start is 0, we can't shift from history, so fill with 0
    if test_start > 0:
        pos_with_context = full_positions[test_start-1 : test_end]
        pos_series = pd.Series(pos_with_context)
        pos = pos_series.shift(1).iloc[1:].reset_index(drop=True)
        # Turnover uses pos_with_context to calculate change at test_start
        pos_change = pos_series.diff().iloc[1:].reset_index(drop=True)
    else:
        pos_series = pd.Series(full_positions[test_start:test_end])
        pos = pos_series.shift(1).fillna(0)
        pos_change = pos_series.diff().fillna(0)
    
    pos.index = returns_slice.index
    pos_change.index = returns_slice.index
    
    strategy_returns = ret_clean * pos
    
    # Costs
    cost_per_side = cost_bps / 10000.0
    transaction_costs = pos_change.abs() * cost_per_side
    
    net_returns = strategy_returns - transaction_costs
    cumulative = (1 + net_returns).cumprod()
    
    total_return = cumulative.iloc[-1] - 1 if (len(cumulative) > 0 and not np.isnan(cumulative.iloc[-1])) else 0
    # Annualize based on hours
    hours = len(returns_slice)
    annualized_return = (1 + total_return) ** ((24 * 365) / hours) - 1 if hours > 0 else 0
    
    vol = net_returns.std() * np.sqrt(24 * 365)
    sharpe = annualized_return / vol if vol > 0 else 0
    
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative / rolling_max) - 1
    max_dd = drawdown.min()
    
    calmar = annualized_return / abs(max_dd) if (max_dd < 0 and not np.isnan(max_dd)) else 0
    
    # Turnover
    turnover = pos_change.abs().mean() * 24 * 30 # Approx monthly turnover
    
    return {
        'return': total_return,
        'ann_return': annualized_return,
        'vol': vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'turnover': turnover,
        'cumulative': cumulative,
        'drawdown': drawdown
    }

def strategy_buy_hold(df):
    """Strategy 0: Buy & Hold."""
    return np.ones(len(df))

def strategy_cash(df):
    """Strategy 6: Cash (0%)."""
    return np.zeros(len(df))

def strategy_static_50(df):
    """Strategy 5: Static 50% Exposure."""
    return np.ones(len(df)) * 0.5

def strategy_random_matched(df, target_turnover=3.0):
    """Strategy 7: Random Exposure with matched turnover."""
    # Monthly turnover = average absolute change * 24 * 30
    # Average hourly change = monthly / (24 * 30)
    avg_change = target_turnover / (24 * 30)
    
    np.random.seed(42)
    changes = np.random.normal(0, avg_change * 1.5, len(df)) # approx scaling
    exposure = np.cumsum(changes)
    exposure = np.clip(exposure, 0, 1)
    return exposure

def strategy_vol_targeting(df, target_vol=0.40, window=168):
    """Strategy 1: Fixed Volatility Targeting."""
    vol_col = f'realized_vol_{window}'
    if vol_col not in df.columns:
        returns = df['close'].pct_change().fillna(0)
        realized_vol = returns.rolling(window).std() * np.sqrt(24 * 365)
    else:
        # Values in dataset are hourly std, need to annualize
        realized_vol = df[vol_col] * np.sqrt(24 * 365)
        
    exposure = target_vol / realized_vol
    exposure = exposure.clip(0, 1).fillna(1.0)
    return exposure.values

def strategy_drawdown_aware(df, target_vol=0.40, vol_window=168, dd_window=720):
    """Strategy 2: Drawdown-aware Volatility Targeting."""
    base_exposure = strategy_vol_targeting(df, target_vol, vol_window)
    
    # Calculate drawdown from recent high (uses full history provided in df)
    rolling_max = df['close'].rolling(dd_window, min_periods=1).max()
    dd = (df['close'] / rolling_max) - 1
    
    multiplier = np.ones(len(df))
    multiplier[dd <= -0.10] = 0.70
    multiplier[dd <= -0.15] = 0.50
    multiplier[dd <= -0.20] = 0.30
    multiplier[dd <= -0.30] = 0.10
    
    return base_exposure * multiplier

def strategy_vol_shock(df, target_vol=0.40, vol_window=168, shock_threshold=2.0):
    """Strategy 3: Volatility shock de-risking."""
    base_exposure = strategy_vol_targeting(df, target_vol, vol_window)
    
    vol_col = f'realized_vol_{vol_window}'
    if vol_col not in df.columns:
        returns = df['close'].pct_change().fillna(0)
        realized_vol = returns.rolling(vol_window).std() * np.sqrt(24 * 365)
    else:
        realized_vol = df[vol_col] * np.sqrt(24 * 365) # Annualize
        
    vol_mean = realized_vol.rolling(720).mean() # 30d mean
    vol_std = realized_vol.rolling(720).std()
    vol_zscore = (realized_vol - vol_mean) / vol_std
    
    multiplier = np.ones(len(df))
    multiplier[vol_zscore > shock_threshold] = 0.5
    multiplier[vol_zscore > (shock_threshold + 1.0)] = 0.0
    
    return base_exposure * multiplier

def strategy_liquidity_stress(df, target_vol=0.40):
    """Strategy 4: Liquidity/stress overlay."""
    base_exposure = strategy_drawdown_aware(df, target_vol)
    
    multiplier = np.ones(len(df))
    
    # Funding stress
    if 'funding_8h_zscore_30d' in df.columns:
        multiplier[df['funding_8h_zscore_30d'] < -3.0] *= 0.5
        
    # Basis stress
    if 'basis_zscore_30d' in df.columns:
        multiplier[df['basis_zscore_30d'] < -3.0] *= 0.5
        
    # Turbulence overlay
    if 'turbulence_feature' in df.columns:
        multiplier[df['turbulence_feature'] > 2.0] *= 0.5
        multiplier[df['turbulence_feature'] > 2.5] *= 0.0
        
    return base_exposure * multiplier

def add_phase12_features(df):
    """Calculate additional features needed for Phase 12."""
    logger.info("Calculating Phase 12 features...")
    
    from utils.turbulence import add_turbulence_feature
    
    # Use log returns for turbulence
    df['log_ret_1h'] = np.log(df['close'] / df['close'].shift(1))
    df['log_ret_24h'] = np.log(df['close'] / df['close'].shift(24))
    
    # We need to handle nans for turbulence calculation
    df_turb = df.copy()
    df_turb = add_turbulence_feature(df_turb, return_cols=['log_ret_1h', 'log_ret_24h', 'atr_pct'], window=252)
    df['turbulence_feature'] = df_turb['turbulence_feature']
    
    return df

def run_evaluation():
    """Main evaluation loop with 4-fold walk-forward (HARDENED)."""
    df_raw = load_and_merge_data()
    df = add_phase12_features(df_raw)
    
    # Ensure directories exist
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EQUITY_CURVES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Chronological Split (4 folds)
    n_samples = len(df)
    fold_size = n_samples // 5
    
    all_results = []
    
    logger.info(f"Starting walk-forward evaluation on {n_samples} bars...")
    
    for fold in range(4):
        test_start = (fold + 1) * fold_size
        test_end = (fold + 2) * fold_size if fold < 3 else n_samples
        # CORRECT IMPLEMENTATION: Pass full history up to test_end to strategies
        # to ensure indicators (rolling vol, drawdown) have full context.
        df_context = df.iloc[:test_end].copy()
        df_context.set_index('date', inplace=True)
        
        raw_strategies = {
            'B&H': strategy_buy_hold(df_context),
            'Static_50%': strategy_static_50(df_context),
            'Cash': strategy_cash(df_context),
            'Random_Matched': strategy_random_matched(df_context),
            'VolTarget_30%': strategy_vol_targeting(df_context, target_vol=0.30),
            'DD_Aware_30%': strategy_drawdown_aware(df_context, target_vol=0.30),
            'VolShock_30%': strategy_vol_shock(df_context, target_vol=0.30),
            'Stress_Overlay_30%': strategy_liquidity_stress(df_context, target_vol=0.30),
        }
        
        # Apply Daily Rebalance (except for B&H, Static, Cash which are constant)
        strategies = {}
        for name, full_pos in raw_strategies.items():
            if name in ['B&H', 'Static_50%', 'Cash']:
                strategies[name] = full_pos
            else:
                # Update exposure only at 00:00 UTC each day
                pos_series = pd.Series(full_pos, index=df_context.index)
                daily_pos = pos_series.resample('24H').first()
                strategies[name] = daily_pos.reindex(pos_series.index).ffill().values
        
        # Test returns slice
        test_returns = df.iloc[test_start:test_end]['close'].pct_change().fillna(0)
        
        for name, full_pos in strategies.items():
            metrics = compute_metrics(test_returns, full_pos, test_start, test_end)
            all_results.append({
                'Fold': fold + 1,
                'Strategy': name,
                'Return': metrics['return'],
                'Ann_Return': metrics['ann_return'],
                'Max_DD': metrics['max_dd'],
                'Calmar': metrics['calmar'],
                'Sharpe': metrics['sharpe'],
                'Turnover': metrics['turnover']
            })
            
    results_df = pd.DataFrame(all_results)
    
    # Aggregate results
    summary = results_df.groupby('Strategy').agg({
        'Ann_Return': ['mean', 'std'],
        'Max_DD': ['mean', 'min'],
        'Calmar': ['mean', 'std'],
        'Sharpe': ['mean', 'std'],
        'Turnover': 'mean'
    }).round(4)
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 12 WALK-FORWARD SUMMARY (DAILY REBALANCE - HARDENED)")
    logger.info("="*80)
    logger.info("\n" + summary.to_string())
    
    # Check Gate
    bh_calmar = summary.loc['B&H', ('Calmar', 'mean')]
    best_strat = summary.drop(['B&H', 'Cash', 'Static_50%', 'Random_Matched']).index[np.argmax(summary.drop(['B&H', 'Cash', 'Static_50%', 'Random_Matched'])[('Calmar', 'mean')])]
    best_calmar = summary.loc[best_strat, ('Calmar', 'mean')]
    
    logger.info("\n" + "-"*40)
    logger.info(f"GATE CHECK: B&H Calmar = {bh_calmar:.4f}")
    logger.info(f"GATE CHECK: Best Managed Calmar ({best_strat}) = {best_calmar:.4f}")
    
    if best_calmar > bh_calmar:
        logger.info("[GATE] PASSED: Managed strategy exceeds B&H Calmar.")
    else:
        logger.warning("[GATE] REJECTED: No managed strategy beats B&H Calmar.")
        logger.warning("[KILL] Phase 12B rule-based strategies should be ARCHIVED.")
    
    # Save reports
    results_df.to_csv(REPORTS_DIR / "phase12_fold_metrics.csv", index=False)
    summary.to_csv(REPORTS_DIR / "phase12_risk_managed_summary.csv")
    
    # Final plot (on full dataset)
    full_returns = df['close'].pct_change().fillna(0)
    plt.figure(figsize=(15, 10))
    for name in ['B&H', 'Static_50%', 'VolTarget_30%', 'DD_Aware_30%', 'VolShock_30%']:
        if name == 'B&H': pos = strategy_buy_hold(df)
        elif name == 'Static_50%': pos = strategy_static_50(df)
        elif 'VolTarget' in name: pos = strategy_vol_targeting(df, 0.30)
        elif 'DD_Aware' in name: pos = strategy_drawdown_aware(df, 0.30)
        elif 'VolShock' in name: pos = strategy_vol_shock(df, 0.30)
        
        metrics = compute_metrics(full_returns, pos, 0, len(df))
        plt.plot(df['date'], metrics['cumulative'], label=f"{name} (Calmar: {metrics['calmar']:.2f})")
        
    plt.yscale('log')
    plt.legend()
    plt.title("Phase 12: Risk-Managed BTC Exposure (Full Period - Hardened)")
    plt.grid(True, alpha=0.3)
    plt.savefig(EQUITY_CURVES_DIR / "phase12_full_period.png")
    logger.info(f"Saved reports to {REPORTS_DIR}")
    logger.info(f"Saved equity curve plot to {EQUITY_CURVES_DIR / 'phase12_full_period.png'}")

if __name__ == "__main__":
    run_evaluation()
