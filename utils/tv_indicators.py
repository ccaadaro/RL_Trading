import numpy as np
import pandas as pd
import pandas_ta as ta

def tv_hma(series: pd.Series, length: int) -> pd.Series:
    """Hull Moving Average (TV Style)"""
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = ta.wma(series, length=half_length)
    wma_full = ta.wma(series, length=length)
    raw_hma = 2 * wma_half - wma_full
    return ta.wma(raw_hma, length=sqrt_length)

def tv_williams_vix_fix(df: pd.DataFrame, pd_len: int = 22, bbl_len: int = 20, mult: float = 2.0, lb_len: int = 50, ph: float = 0.85) -> pd.DataFrame:
    """Williams Vix Fix (Synthetic VIX) with TV-prefix names."""
    highest_close = df['close'].rolling(pd_len).max()
    wvf = ((highest_close - df['low']) / highest_close) * 100
    
    std_wvf = wvf.rolling(bbl_len).std()
    mid_wvf = wvf.rolling(bbl_len).mean()
    upper_band = mid_wvf + mult * std_wvf
    range_high = wvf.rolling(lb_len).max() * ph
    
    is_panic = (wvf >= upper_band) | (wvf >= range_high)
    
    return pd.DataFrame({
        'tv_wvf_val': wvf / 100.0,
        'tv_wvf_panic': is_panic.astype(float)
    }, index=df.index)

def tv_laguerre_bundle(series: pd.Series) -> pd.DataFrame:
    """Compact Laguerre bundle: Fast(0.2), Mid(0.5), Slow(0.8)."""
    l_fast = _laguerre_core(series, 0.2)
    l_mid  = _laguerre_core(series, 0.5)
    l_slow = _laguerre_core(series, 0.8)
    
    slope = (l_mid / l_mid.shift(1) - 1).fillna(0)
    dispersion = pd.concat([l_fast, l_mid, l_slow], axis=1).std(axis=1) / series.clip(1e-10)
    price_dist = (series / l_mid - 1).fillna(0)
    
    return pd.DataFrame({
        'tv_lag_fast': l_fast,
        'tv_lag_mid': l_mid,
        'tv_lag_slow': l_slow,
        'tv_lag_slope': slope,
        'tv_lag_dispersion': dispersion,
        'tv_price_dist_lag': price_dist
    }, index=series.index)

def _laguerre_core(series: pd.Series, gamma: float) -> pd.Series:
    """Recursive Laguerre Filter (Causal)."""
    vals = series.values
    l0, l1, l2, l3 = 0.0, 0.0, 0.0, 0.0
    out = np.zeros_like(vals)
    for i in range(len(vals)):
        l0_prev, l1_prev, l2_prev, l3_prev = l0, l1, l2, l3
        l0 = (1 - gamma) * vals[i] + gamma * l0_prev
        l1 = -gamma * l0 + l0_prev + gamma * l1_prev
        l2 = -gamma * l1 + l1_prev + gamma * l2_prev
        l3 = -gamma * l2 + l2_prev + gamma * l3_prev
        out[i] = (l0 + 2*l1 + 2*l2 + l3) / 6
    return pd.Series(out, index=series.index)

def tv_koncorde_selective(df: pd.DataFrame, m_len: int = 15) -> pd.DataFrame:
    """Selective Koncorde/TSV components: PVI/NVI spread and basic TSV."""
    pvi = ta.pvi(df['close'], df['volume']).ffill().fillna(100)
    nvi = ta.nvi(df['close'], df['volume']).ffill().fillna(100)
    
    pvi_ema = ta.ema(pvi, length=m_len)
    nvi_ema = ta.ema(nvi, length=m_len)
    
    # print(f"DEBUG: pvi_ema type={type(pvi_ema)}, nvi_ema type={type(nvi_ema)}")
    
    # Spread as a feature
    spread = (pvi_ema - nvi_ema) / nvi_ema.clip(1e-9)
    
    # TSV (Time Segmented Volume) - simplified causal version
    tsv_raw = (df['close'].diff() * df['volume']).rolling(13).sum()
    denom = (df['close'].rolling(13).mean() * df['volume'].rolling(13).mean()).clip(1e-9)
    tv_tsv = tsv_raw / denom

    return pd.DataFrame({
        'tv_pvi_nvi_spread': spread.fillna(0),
        'tv_tsv': tv_tsv.fillna(0)
    }, index=df.index)

def tv_microstructure_refined(df: pd.DataFrame) -> pd.DataFrame:
    """Clean microstructure features: CVD slope, aggr delta, imbalance."""
    # Assume buy_volume, volume, aggressor_ratio are already present in df
    buy_v = df.get('buy_volume', df.get('buy_vol', 0))
    sell_v = (df['volume'] - buy_v).clip(0)
    
    cvd = (buy_v - sell_v).cumsum()
    # Z-score of CVD over a window is more stationary than raw CVD
    cvd_ma = cvd.rolling(100).mean()
    cvd_std = cvd.rolling(100).std().clip(1e-9)
    cvd_zscore = (cvd - cvd_ma) / cvd_std
    
    cvd_slope = (cvd.diff() / df['volume'].clip(1e-9)).rolling(10).mean()
    
    aggr = df.get('aggressor_ratio', 0.5)
    aggr_delta = aggr - pd.Series(aggr).rolling(24).mean()
    
    imbalance = (buy_v - sell_v) / df['volume'].clip(1e-9)
    
    return pd.DataFrame({
        'tv_cvd_zscore': cvd_zscore.fillna(0),
        'tv_cvd_slope': cvd_slope.fillna(0),
        'tv_aggr_delta': aggr_delta.fillna(0),
        'tv_buy_sell_imbalance': imbalance.fillna(0)
    }, index=df.index)
