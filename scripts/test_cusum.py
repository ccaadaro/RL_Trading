import pandas as pd
import numpy as np
from utils.filters import SymmetricCUSUMFilter

df = pd.read_feather('cache/dollar_bars_btc_2000000_sizing.feather')
close = df['close'].values
log_ret = df['log_return_feature'].values
sigma = pd.Series(log_ret).rolling(100).std().values
sigma_robust = pd.Series(log_ret).rolling(100).apply(lambda x: np.nanmedian(np.abs(x)) * 1.4826).values
sigma_eff = np.minimum(sigma, sigma_robust * 2.0)

cusum = SymmetricCUSUMFilter()
events = 0
for i in range(1, len(df)):
    lr = np.log(close[i] / close[i-1])
    se = sigma_eff[i]
    h_t = 3.5 * se if (not np.isnan(se) and se > 0) else 0.005
    if cusum.check(lr, h_t):
        events += 1
print(f"Total CUSUM events calculated: {events}")
