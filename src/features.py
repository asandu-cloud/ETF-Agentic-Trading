# src/features.py

import numpy as np
import pandas as pd
from src.data_loader import load_processed_prices

import numpy as np
import pandas as pd


def _add_features_single_ticker(g: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical features for a single ticker.
    Assumes g has columns: date, adj_close, volume, ret_1d, log_ret_1d.
    """
    g = g.sort_values("date").copy()

    price = g["adj_close"]
    vol = g["volume"]
    ret = g["ret_1d"]
    log_ret = g["log_ret_1d"]

    # --- TREND / MOMENTUM ---
    g["sma_10"] = price.rolling(10).mean()
    g["sma_20"] = price.rolling(20).mean()
    g["sma_50"] = price.rolling(50).mean()
    g["sma_200"] = price.rolling(200).mean()

    g["trend_20_50_up"] = (g["sma_20"] > g["sma_50"]).astype(int)
    g["trend_50_200_up"] = (g["sma_50"] > g["sma_200"]).astype(int)

    g["mom_5"] = price.pct_change(5)
    g["mom_20"] = price.pct_change(20)
    g["mom_60"] = price.pct_change(60)

    # --- VOLATILITY / RISK ---
    for w in (10, 20, 60):
        g[f"vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

    # 60d rolling max drawdown
    roll_max = price.rolling(60, min_periods=20).max()
    g["dd_60d"] = price / roll_max - 1.0

    # --- VOLUME FEATURES ---
    vol_ma_20 = vol.rolling(20, min_periods=5).mean()
    vol_ma_60 = vol.rolling(60, min_periods=10).mean()
    vol_std_20 = vol.rolling(20, min_periods=5).std()
    vol_std_60 = vol.rolling(60, min_periods=10).std()

    g["vol_ma_20"] = vol_ma_20
    g["vol_ma_60"] = vol_ma_60
    g["vol_z_20"] = (vol - vol_ma_20) / vol_std_20
    g["vol_z_60"] = (vol - vol_ma_60) / vol_std_60

    g["dollar_vol"] = price * vol

    # --- BREAKOUTS ---
    high_20 = price.rolling(20, min_periods=20).max()
    low_20 = price.rolling(20, min_periods=20).min()

    g["dist_to_20d_high"] = price / high_20 - 1.0
    g["dist_to_20d_low"] = price / low_20 - 1.0

    g["breakout_20d_high"] = (price >= high_20).astype(int)
    g["breakdown_20d_low"] = (price <= low_20).astype(int)

    return g


def add_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add a broad set of technical features per ticker.

    Expects long-format DataFrame with at least:
        ['date', 'ticker', 'adj_close', 'volume', 'ret_1d', 'log_ret_1d'].
    """
    df = prices.sort_values(["ticker", "date"]).copy()

    # Apply per-ticker to avoid any index alignment issues
    df = (
        df.groupby("ticker", group_keys=False)
          .apply(_add_features_single_ticker)
    )

    return df
