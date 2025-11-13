# -IMPORTS-

import os
from pathlib import Path
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime
from .config import TICKERS, START_DATE, END_DATE

# -PATHS

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RAW_PRICES_CSV = RAW_DIR / "prices_yahoo.csv"
PROCESSED_PRICES_PARQUET = PROCESSED_DIR / "prices_daily.parquet"


# -FUNCTIONS-
def download_prices(save=True):
    '''
    Download prices for tickers in config.
    Save to raw/prices_yahoo.csv
    Return pd df
    '''
    print('Downloading data for: {TICKERS}')
    data = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False
    )

    # drop empty rows (NaN)
    data = data.dropna(how='all')

    if save:
        data.to_csv(RAW_PRICES_CSV)
        print(f"Saved raw prices to {RAW_PRICES_CSV}")

    return data


def load_raw_prices() -> pd.DataFrame:
    """
    Load raw prices from csv (as saved by download_prices).
    """
    if not RAW_PRICES_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_PRICES_CSV} not found. Run download_prices() first."
        )

    # yfinance saves MultiIndex columns; header=[0,1] reconstructs them
    df = pd.read_csv(RAW_PRICES_CSV, header=[0, 1], index_col=0, parse_dates=True)
    return df


def _prices_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conver columns from yfinance base to: date, ticker, open, high, low, close, adj_close, volume
    """
    # expect MultiIndex columns: (field, ticker)
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns (field, ticker) from yfinance")

    # swap so level 0 = ticker, level 1 = field
    df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)

    # go to long format
    df_long = df.stack(level=0, future_stack=True).rename_axis(["date", "ticker"]).reset_index()

    # normalise column names
    df_long.columns = [c.lower().replace(" ", "_") for c in df_long.columns]

    # if there's no adj_close (because auto_adjust=True), use close as adj_close
    if "adj_close" not in df_long.columns and "close" in df_long.columns:
        df_long["adj_close"] = df_long["close"]

    expected = {"open", "high", "low", "close", "adj_close", "volume"}
    missing = expected.difference(df_long.columns)
    if missing:
        raise ValueError(f"Missing columns in long prices: {missing}")

    return df_long


def _add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic return features for agents 
    """
    df = df.sort_values(["ticker", "date"]).copy()

    # simple and log returns
    df["ret_1d"] = df.groupby("ticker")["adj_close"].pct_change()
    df["log_ret_1d"] = (
        np.log(df["adj_close"]).groupby(df["ticker"]).diff()
    )

    # forward returns (labels)
    df["ret_fwd_1d"] = df.groupby("ticker")["ret_1d"].shift(-1)
    df["ret_fwd_5d"] = (
        df.groupby("ticker")["adj_close"].pct_change(5).shift(-5)
    )

    # 20d rolling volatility of daily log returns (for risk agent)
    df["vol_20d"] = (
        df.groupby("ticker")["log_ret_1d"]
        .rolling(window=20)
        .std()
        .reset_index(level=0, drop=True)
        * np.sqrt(252)  # annualised
    )

    return df


def build_processed_prices(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Create dataset for agents to use
    """
    if PROCESSED_PRICES_PARQUET.exists() and not force_rebuild:
        print(f"Loading existing processed prices from {PROCESSED_PRICES_PARQUET}")
        return pd.read_parquet(PROCESSED_PRICES_PARQUET)

    # raw data check
    if RAW_PRICES_CSV.exists():
        raw = load_raw_prices()
    else:
        raw = download_prices(save=True)

    # long format + features
    long = _prices_to_long(raw)
    long = _add_return_features(long)

    # save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    long.to_parquet(PROCESSED_PRICES_PARQUET)
    print(f"Saved processed prices to {PROCESSED_PRICES_PARQUET}")

    return long


if __name__ == "__main__":
    print("Building processed price dataset…")
    df = build_processed_prices(force_rebuild=True)
    print(df.head())
    print("\nCompleted successfully.")
