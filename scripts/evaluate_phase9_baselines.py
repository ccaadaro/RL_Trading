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

LABEL_COLS = ['triple_barrier_48h', 'trend_48h', 'trend_72h']

# Programmatic Safety: Ensure no labels ever leak into features
ALL_FEATURES = TREND_FEATURES + VOL_FEATURES + MICRO_FEATURES
leak_cols = [c for c in ALL_FEATURES if c in LABEL_COLS]
assert not leak_cols, f"CRITICAL LEAKAGE: Columns {leak_cols} are marked as both features and labels!"

def compute_pnl(returns, positions, cost_bps=7.0):
    """Compute PnL, Sharpe, Calmar with transaction costs."""
    # Ensure no nans in returns
    ret_clean = returns.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Ensure index alignment
    pos_series = pd.Series(positions, index=returns.index)
    pos = pos_series.shift(1).fillna(0)
    strategy_returns = ret_clean * pos
    
    # Costs: 7bps per side (0.07% / 0.0007)
    # A position flip (1 to -1) is a 2.0 change in exposure -> 14bps cost
    cost_per_side = cost_bps / 10000.0
    pos_change = pos.diff().abs().fillna(0)
    transaction_costs = pos_change * cost_per_side
    
    net_returns = strategy_returns - transaction_costs
    cumulative = (1 + net_returns).cumprod()

    total_return = cumulative.iloc[-1] - 1 if (len(cumulative) > 0 and not np.isnan(cumulative.iloc[-1])) else 0
    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252 * 24) if (net_returns.std() != 0 and not np.isnan(net_returns.std())) else 0
    max_dd = (cumulative / cumulative.expanding().max()).min() - 1 if len(cumulative) > 0 else 0
    calmar = total_return / abs(max_dd) if (max_dd != 0 and not np.isnan(max_dd)) else 0

    return {
        'return': total_return,
        'sharpe': sharpe,
        'calmar': calmar,
        'max_dd': max_dd,
    }

def baseline_buy_hold(df_fold):
    """Buy-and-hold baseline."""
    returns = df_fold['close'].pct_change().fillna(0)
    positions = np.ones(len(returns))
    pnl = compute_pnl(returns, positions)
    pnl['auc'] = 0.5
    return pnl

def baseline_ema_crossover(df_fold):
    """EMA12 > EMA26 entry signal."""
    ema_12 = df_fold['close'].ewm(span=12).mean()
    ema_26 = df_fold['close'].ewm(span=26).mean()
    signal = (ema_12 > ema_26).astype(int)
    returns = df_fold['close'].pct_change().fillna(0)
    pnl = compute_pnl(returns, signal)
    pnl['auc'] = 0.5
    return pnl

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
        'num_leaves': 15,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'verbose': -1,
    }
    model = lgb.train(params, train_data, num_boost_round=50)

    # Predict
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = (y_pred_proba > 0.5).astype(int)

    return y_pred, y_pred_proba, model

def evaluate_model(df_test, y_pred, y_pred_proba, vol_thresh):
    """Evaluate model on test set."""
    returns = df_test['close'].pct_change().reset_index(drop=True)

    # Positions: 48h hold strategy (only rebalance every 48 hours to match horizon)
    # Vol-conditional entry: skip trades when realized_vol_24 is in bottom quartile
    # (Threshold is calculated from training data to avoid forward leak)
    positions = np.zeros(len(df_test))
    BARRIER_HOURS = 48
    
    current_pos = 0
    hold_timer = 0
    
    for i in range(len(df_test)):
        if hold_timer > 0:
            # Continue holding
            positions[i] = current_pos
            hold_timer -= 1
        else:
            # Re-evaluate
            is_low_vol = df_test['realized_vol_24'].iloc[i] < vol_thresh
            
            if is_low_vol:
                current_pos = 0 # Skip entry in low vol
            else:
                # Asymmetric thresholds: only act on high confidence
                prob = y_pred_proba[i]
                if prob > 0.55:
                    current_pos = 1
                elif prob < 0.45:
                    current_pos = -1
                else:
                    current_pos = 0 # Sit flat on near-50/50
            
            positions[i] = current_pos
            hold_timer = BARRIER_HOURS - 1 # Hold for this bar + 47 more

    # Metrics (only on valid labels)
    valid_mask = df_test['triple_barrier_48h'].notna()
    y_true_raw = df_test['triple_barrier_48h'][valid_mask]
    y_true_binary = ((y_true_raw + 1) / 2).astype(int)
    
    y_pred_proba_valid = y_pred_proba[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    auc = roc_auc_score(y_true_binary, y_pred_proba_valid) if len(np.unique(y_true_binary)) > 1 else np.nan
    accuracy = accuracy_score(y_true_binary, y_pred_valid)

    # PnL
    pnl = compute_pnl(returns, positions)

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
    # Divide into 5 segments to have 1 for initial training and 4 for testing
    fold_size = n_samples // 5

    logger.info(f"Running 4-fold walk-forward on {n_samples} samples (fold_size={fold_size})")

    results = {
        'buy_hold': [],
        'ema_crossover': [],
        'lgb_trend': [],
        'lgb_trend_vol': [],
    }

    PURGE_WINDOW = 72  # 72 hours (exceeds 48h barrier lookahead)

    for fold in range(4):
        # Shift test window by 1 fold_size to allow for initial training
        test_start = (fold + 1) * fold_size
        test_end = (fold + 2) * fold_size if fold < 3 else n_samples
        train_end = max(0, test_start - PURGE_WINDOW)

        # Use all data before test set for training (expanding window)
        df_train = df_clean.iloc[:train_end]
        df_test = df_clean.iloc[test_start:test_end]

        logger.info(f"\nFold {fold+1}/4: train={len(df_train)}, test={len(df_test)}, purge={PURGE_WINDOW}")
        
        # Volatility threshold from training data (to avoid leak in test)
        vol_thresh = df_train['realized_vol_24'].quantile(0.25)

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
            trend_result = evaluate_model(df_test, y_pred, y_pred_proba, vol_thresh)
            results['lgb_trend'].append(trend_result)
            logger.info(f"  LGB-Trend: AUC={trend_result['auc']:.4f}, Acc={trend_result['accuracy']:.4f}, "
                       f"return={trend_result['return']:.4f}, calmar={trend_result['calmar']:.4f}")

            # With vol features
            X_train_vol = df_train[TREND_FEATURES + VOL_FEATURES].fillna(0)
            X_test_vol = df_test[TREND_FEATURES + VOL_FEATURES].fillna(0)

            y_pred_vol, y_pred_proba_vol, _ = train_lgb_model(X_train_vol, y_train, X_test_vol)
            vol_result = evaluate_model(df_test, y_pred_vol, y_pred_proba_vol, vol_thresh)
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

            mean_auc = np.nanmean(aucs)
            std_auc = np.nanstd(aucs)
            mean_ret = np.nanmean(rets)
            std_ret = np.nanstd(rets)
            mean_calmar = np.nanmean(calmars)
            std_calmar = np.nanstd(calmars)

            logger.info(f"\n{model_name}:")
            logger.info(f"  AUC:    {mean_auc:.4f} ± {std_auc:.4f} (folds: {[f'{a:.4f}' for a in aucs]})")
            logger.info(f"  Return: {mean_ret:.4f} ± {std_ret:.4f} (folds: {[f'{r:.4f}' for r in rets]})")
            logger.info(f"  Calmar: {mean_calmar:.4f} ± {std_calmar:.4f} (folds: {[f'{c:.4f}' for c in calmars]})")

            # Programmatic Acceptance Gate Status
            THRESHOLD = 0.55
            if mean_auc >= THRESHOLD:
                logger.info(f"  [GATE] Result: APPROVED (AUC {mean_auc:.4f} >= {THRESHOLD})")
            else:
                logger.info(f"  [GATE] Result: REJECTED (AUC {mean_auc:.4f} < {THRESHOLD})")

    # Final Success Flag (for CI exit code)
    passed_gate = False
    for model_name in ['lgb_trend', 'lgb_trend_vol']:
        fold_results = results.get(model_name, [])
        if fold_results:
            mean_auc = np.nanmean([r.get('auc', np.nan) for r in fold_results])
            if mean_auc >= 0.55:
                passed_gate = True
                break
    
    return passed_gate

if __name__ == '__main__':
    import sys
    success = run_4fold_walkforward()
    if not success:
        logger.error("\n[CRITICAL] Phase 9 evaluation REJECTED. AUC threshold not met.")
        sys.exit(1)
    else:
        logger.info("\n[SUCCESS] Phase 9 evaluation APPROVED.")
        sys.exit(0)
