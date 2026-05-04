#!/usr/bin/env python3
"""
Evaluate Phase 9 baselines on 4-fold walk-forward.

Baselines:
1. Buy-and-hold (expected return, Sharpe, Calmar)
2. EMA/HMA crossover (no ML)
3. Trend-only LightGBM
4. Trend+Vol LightGBM
5. Trend+Microstructure LightGBM (future: add CVD/aggressor)

Accept model only if:
- AUC > 0.55
- Calmar > 0.5
- Consistent across 4 folds (not just 1 lucky fold)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import lightgbm as lgb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset
DATA_PATH = Path('/home/nosferatu/freqtrade/user_data/strategies/RL_Trading/cache/btc_1h_phase9.feather')
df = pd.read_feather(DATA_PATH)

# Feature groups
TREND_FEATURES = [
    'return_3h', 'return_6h', 'return_12h', 'return_24h', 'return_48h',
    'ema_12_slope', 'ema_26_slope', 'ema_50_slope', 'hma_slope',
    'ma_bias_24', 'ma_bias_48', 'ma_bias_96', 'ma_bias_200',
]

VOL_FEATURES = [
    'realized_vol_24', 'realized_vol_72', 'realized_vol_168',
    'rsi_14', 'rsi_42', 'atr_pct',
    'drawdown_from_high_72', 'distance_to_high_72',
    'drawdown_from_high_168', 'distance_to_high_168',
]

MICRO_FEATURES = [
    'volume_zscore_24', 'volume_zscore_72',
    # Will add CVD, aggressor, whale features from micro dataset
]

def compute_pnl(returns, positions):
    """Compute PnL, Sharpe, Calmar from returns and positions."""
    strategy_returns = returns * positions.shift(1)
    cumulative = (1 + strategy_returns).cumprod()

    total_return = cumulative.iloc[-1] - 1
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 24)  # annualized
    max_dd = (cumulative / cumulative.expanding().max()).min() - 1
    calmar = total_return / abs(max_dd) if max_dd != 0 else 0

    return {
        'return': total_return,
        'sharpe': sharpe,
        'calmar': calmar,
        'max_dd': max_dd,
    }

def baseline_buy_hold(df_fold):
    """Buy-and-hold baseline."""
    returns = df_fold['close'].pct_change()
    positions = np.ones(len(returns))
    return compute_pnl(returns, pd.Series(positions))

def baseline_ema_crossover(df_fold):
    """EMA12 > EMA26 entry signal."""
    ema_12 = df_fold['close'].ewm(span=12).mean()
    ema_26 = df_fold['close'].ewm(span=26).mean()
    signal = (ema_12 > ema_26).astype(int)
    returns = df_fold['close'].pct_change()
    return compute_pnl(returns, pd.Series(signal))

def train_lgb_model(X_train, y_train, X_test):
    """Train LightGBM and predict on test set."""
    # Handle NaN in targets (convert -1/1 to 0/1 for binary classification)
    valid_idx = y_train.notna()
    X_train_clean = X_train[valid_idx]
    y_train_clean = ((y_train[valid_idx] + 1) / 2).astype(int)  # Convert {-1, 1} to {0, 1}

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test)

    # Train
    train_data = lgb.Dataset(X_train_scaled, label=y_train_clean)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'verbose': -1,
    }
    model = lgb.train(params, train_data, num_boost_round=100)

    # Predict
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = (y_pred_proba > 0.5).astype(int)

    return y_pred, y_pred_proba, model

def evaluate_model(df_test, y_pred, y_pred_proba):
    """Evaluate model on test set."""
    returns = df_test['close'].pct_change()

    # Positions: 1 if pred=1, -1 if pred=0, 0 if prediction uncertain
    positions = np.where(y_pred == 1, 1, 0)

    # Metrics
    y_true_binary = ((df_test['triple_barrier_48h'] + 1) / 2).astype(int)
    auc = roc_auc_score(y_true_binary, y_pred_proba) if len(np.unique(y_true_binary)) > 1 else np.nan
    accuracy = accuracy_score(y_true_binary, y_pred)

    # PnL
    pnl = compute_pnl(returns, pd.Series(positions))

    return {
        'auc': auc,
        'accuracy': accuracy,
        'positions': positions,
        **pnl,
    }

def run_4fold_walkforward():
    """Run 4-fold walk-forward validation."""

    # Prepare data
    df_clean = df.dropna(subset=['triple_barrier_48h']).reset_index(drop=True)
    n_samples = len(df_clean)
    fold_size = n_samples // 4

    logger.info(f"Running 4-fold walk-forward on {n_samples} samples (fold_size={fold_size})")

    results = {
        'buy_hold': [],
        'ema_crossover': [],
        'lgb_trend': [],
        'lgb_trend_vol': [],
    }

    for fold in range(4):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < 3 else n_samples
        train_end = test_start

        # Use all data before test set for training (expanding window)
        df_train = df_clean.iloc[:train_end]
        df_test = df_clean.iloc[test_start:test_end]

        logger.info(f"\nFold {fold+1}/4: train={len(df_train)}, test={len(df_test)}")

        # Baselines
        bh_result = baseline_buy_hold(df_test)
        results['buy_hold'].append(bh_result)
        logger.info(f"  B&H: return={bh_result['return']:.4f}, sharpe={bh_result['sharpe']:.4f}, calmar={bh_result['calmar']:.4f}")

        ema_result = baseline_ema_crossover(df_test)
        results['ema_crossover'].append(ema_result)
        logger.info(f"  EMA: return={ema_result['return']:.4f}, sharpe={ema_result['sharpe']:.4f}, calmar={ema_result['calmar']:.4f}")

        # LightGBM models
        X_train = df_train[TREND_FEATURES].fillna(0)
        X_test = df_test[TREND_FEATURES].fillna(0)
        y_train = df_train['triple_barrier_48h']
        y_test = df_test['triple_barrier_48h']

        if len(X_train) > 0 and len(X_test) > 0:
            y_pred, y_pred_proba, _ = train_lgb_model(X_train, y_train, X_test)
            trend_result = evaluate_model(df_test, y_pred, y_pred_proba)
            results['lgb_trend'].append(trend_result)
            logger.info(f"  LGB-Trend: AUC={trend_result['auc']:.4f}, Acc={trend_result['accuracy']:.4f}, "
                       f"return={trend_result['return']:.4f}, calmar={trend_result['calmar']:.4f}")

            # With vol features
            X_train_vol = df_train[TREND_FEATURES + VOL_FEATURES].fillna(0)
            X_test_vol = df_test[TREND_FEATURES + VOL_FEATURES].fillna(0)

            y_pred_vol, y_pred_proba_vol, _ = train_lgb_model(X_train_vol, y_train, X_test_vol)
            vol_result = evaluate_model(df_test, y_pred_vol, y_pred_proba_vol)
            results['lgb_trend_vol'].append(vol_result)
            logger.info(f"  LGB-Trend+Vol: AUC={vol_result['auc']:.4f}, Acc={vol_result['accuracy']:.4f}, "
                       f"return={vol_result['return']:.4f}, calmar={vol_result['calmar']:.4f}")

    # Summary stats
    logger.info("\n" + "="*80)
    logger.info("4-FOLD SUMMARY")
    logger.info("="*80)

    for model_name, fold_results in results.items():
        if fold_results:
            aucs = [r.get('auc', np.nan) for r in fold_results]
            rets = [r.get('return', np.nan) for r in fold_results]
            calmars = [r.get('calmar', np.nan) for r in fold_results]

            logger.info(f"\n{model_name}:")
            logger.info(f"  AUC:    {np.nanmean(aucs):.4f} ± {np.nanstd(aucs):.4f} (folds: {[f'{a:.4f}' for a in aucs]})")
            logger.info(f"  Return: {np.nanmean(rets):.4f} ± {np.nanstd(rets):.4f} (folds: {[f'{r:.4f}' for r in rets]})")
            logger.info(f"  Calmar: {np.nanmean(calmars):.4f} ± {np.nanstd(calmars):.4f} (folds: {[f'{c:.4f}' for c in calmars]})")

if __name__ == '__main__':
    run_4fold_walkforward()
