import pandas as pd

# Cargar feather entero solo una vez
trades = pd.read_feather("../../data/binance/BTC_USDT-trades.feather")

# Guardar como Parquet
trades.to_parquet("../../data/binance/BTC_USDT-trades.parquet", index=False)
