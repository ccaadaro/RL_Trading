import numpy as np
import pandas as pd
import pandas_ta as ta

def hma(series: pd.Series, length: int) -> pd.Series:
    """Hull Moving Average"""
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    
    wma_half = ta.wma(series, length=half_length)
    wma_full = ta.wma(series, length=length)
    
    raw_hma = 2 * wma_half - wma_full
    return ta.wma(raw_hma, length=sqrt_length)

def thma(series: pd.Series, length: int) -> pd.Series:
    """Triple Hull Moving Average"""
    # THMA = wma(wma(src, length / 3) * 3 - wma(src, length / 2) - wma(src, length), length)
    wma3 = ta.wma(series, length=int(length / 3))
    wma2 = ta.wma(series, length=int(length / 2))
    wma1 = ta.wma(series, length=length)
    
    raw_thma = 3 * wma3 - wma2 - wma1
    return ta.wma(raw_thma, length=length)

def ehma(series: pd.Series, length: int) -> pd.Series:
    """Exponential Hull Moving Average"""
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    
    ema_half = ta.ema(series, length=half_length)
    ema_full = ta.ema(series, length=length)
    
    raw_ehma = 2 * ema_half - ema_full
    return ta.ema(raw_ehma, length=sqrt_length)

def williams_vix_fix(df: pd.DataFrame, pd_len: int = 22, bbl_len: int = 20, mult: float = 2.0, lb_len: int = 50, ph: float = 0.85) -> pd.DataFrame:
    """
    Williams Vix Fix (Synthetic VIX) implementation.
    """
    highest_close = df['close'].rolling(pd_len).max()
    wvf = ((highest_close - df['low']) / highest_close) * 100
    
    # Bollinger Bands on WVF
    std_wvf = wvf.rolling(bbl_len).std()
    mid_wvf = wvf.rolling(bbl_len).mean()
    upper_band = mid_wvf + mult * std_wvf
    
    # Range High Percentile
    range_high = wvf.rolling(lb_len).max() * ph
    
    # Indicator is active (pánico) if wvf > upper_band or wvf > range_high
    is_panic = (wvf >= upper_band) | (wvf >= range_high)
    
    return pd.DataFrame({
        'wvf': wvf,
        'wvf_upper_band': upper_band,
        'wvf_range_high': range_high,
        'wvf_panic': is_panic.astype(float)
    }, index=df.index)

def laguerre_filter(series: pd.Series, gamma: float) -> pd.Series:
    """
    Laguerre Filter (4-pole) - Recursive implementation.
    """
    vals = series.values
    l0 = np.zeros_like(vals)
    l1 = np.zeros_like(vals)
    l2 = np.zeros_like(vals)
    l3 = np.zeros_like(vals)
    filt = np.zeros_like(vals)
    
    for i in range(len(vals)):
        if i == 0:
            l0[i] = (1 - gamma) * vals[i]
            l1[i] = -gamma * l0[i] + l0[i]
            l2[i] = -gamma * l1[i] + l1[i]
            l3[i] = -gamma * l2[i] + l2[i]
        else:
            l0[i] = (1 - gamma) * vals[i] + gamma * l0[i-1]
            l1[i] = -gamma * l0[i] + l0[i-1] + gamma * l1[i-1]
            l2[i] = -gamma * l1[i] + l1[i-1] + gamma * l2[i-1]
            l3[i] = -gamma * l2[i] + l2[i-1] + gamma * l3[i-1]
        
        filt[i] = (l0[i] + 2*l1[i] + 2*l2[i] + l3[i]) / 6
        
    return pd.Series(filt, index=series.index)

def koncorde_components(df: pd.DataFrame, m_len: int = 15, pvi_len: int = 90, nvi_len: int = 90) -> pd.DataFrame:
    """
    Koncorde components: MFI-based (osc_pos) and NVI-based (osc_neg).
    """
    # ta.pvi and ta.nvi expect volume
    pvi = ta.pvi(df['close'], df['volume']).fillna(method='ffill').fillna(100)
    nvi = ta.nvi(df['close'], df['volume']).fillna(method='ffill').fillna(100)
    
    pvi_ema = ta.ema(pvi, length=m_len)
    pvi_max = pvi_ema.rolling(pvi_len).max()
    pvi_min = pvi_ema.rolling(pvi_len).min()
    osc_pos = (pvi - pvi_ema) * 100 / (pvi_max - pvi_min).clip(lower=1e-9)
    
    nvi_ema = ta.ema(nvi, length=m_len)
    nvi_max = nvi_ema.rolling(nvi_len).max()
    nvi_min = nvi_ema.rolling(nvi_len).min()
    osc_neg = (nvi - nvi_ema) * 100 / (nvi_max - nvi_min).clip(lower=1e-9)
    
    return pd.DataFrame({
        'osc_pos': osc_pos.fillna(0),
        'osc_neg': osc_neg.fillna(0)
    }, index=df.index)
