# -IMPORTS-

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

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
PROCESSED_PRICES_CSV = PROCESSED_DIR / "prices_daily.csv"


# -FUNCTIONS-
def download_prices(
        save: bool =True, # if true, raw data saved to RAW_PRICES_CSV
        tickers: Optional[Sequence[str]] = None, # tickers to download, takes from config if none here 
        start: Optional[str] = None, # ''
        end: Optional[str] = None # ''
        ) -> pd.DataFrame:
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
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        data.to_csv(RAW_PRICES_CSV)
        print(f"Saved raw prices to {RAW_PRICES_CSV}")

    return data


def load_raw_prices(path: Path = RAW_PRICES_CSV) -> pd.DataFrame:
    """
    Load raw prices from csv (as saved by download_prices).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run download_prices() first."
        )

    # yfinance saves MultiIndex columns; header=[0,1] reconstructs them
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    return df


def _prices_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conver columns from yfinance base to: date, ticker, open, high, low, close, adj_close, volume
    """
    # expect MultiIndex columns: (field, ticker)
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns (field, ticker) from yfinance")

    # ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # swap so level 0 = ticker, level 1 = field
    df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)

    # go to long format
    df_long = (
        df.stack(level=0, future_stack=True)
        .rename_axis(["date", "ticker"])
        .reset_index()
    )

    # normalise column names
    df_long.columns = [c.lower().replace(" ", "_") for c in df_long.columns]

    # if there's no adj_close (because auto_adjust=True), use close as adj_close
    if "adj_close" not in df_long.columns and "close" in df_long.columns:
        df_long["adj_close"] = df_long["close"]

    expected = {"open", "high", "low", "close", "adj_close", "volume"}
    missing = expected.difference(df_long.columns)
    if missing:
        raise ValueError(f"Missing columns in long prices: {missing}")

    # enforce dtypes
    df_long["date"] = pd.to_datetime(df_long["date"])
    df_long["ticker"] = df_long["ticker"].astype(str)

    float_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    df_long[float_cols] = df_long[float_cols].astype(float)

    return df_long


def _add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic return features for agents 
    """
    df = df.sort_values(["ticker", "date"]).copy()

    # simple and log returns
    df["ret_1d"] = df.groupby("ticker", group_keys=False)["adj_close"].pct_change()
    df["log_ret_1d"] = (
        np.log(df["adj_close"]).groupby(df["ticker"], group_keys=False).diff()
    )

    # forward returns (labels)
    df["ret_fwd_1d"] = df.groupby("ticker", group_keys=False)["ret_1d"].shift(-1)
    df["ret_fwd_5d"] = (
        df.groupby("ticker", group_keys=False)["adj_close"].pct_change(5).shift(-5)
    )

    # 20d rolling volatility of daily log returns (for risk agent)
    df["vol_20d"] = (
        df.groupby("ticker", group_keys=False)["log_ret_1d"]
        .rolling(window=20, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
        * np.sqrt(252)  # annualised
    )

    # base columns for ordering
    base_cols = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "ret_1d",
        "log_ret_1d",
        "ret_fwd_1d",
        "ret_fwd_5d",
        "vol_20d",
    ]
    other_cols = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + other_cols]

    return df

def _run_quality_checks(df: pd.DataFrame) -> None:
    """
    Run basic data quality checks and print warnings.

    Checks:
        - duplicated (date, ticker) pairs
        - NaN counts per column
        - large calendar gaps (> 5 days) per ticker
    """
    print("Running data quality checks...")

    # duplicates
    dup_mask = df.duplicated(subset=["date", "ticker"])
    num_dups = dup_mask.sum()
    if num_dups > 0:
        raise ValueError(f"Found {num_dups} duplicated (date, ticker) rows.")

    # NaNs
    na_counts = df.isna().sum()
    na_counts = na_counts[na_counts > 0]
    if not na_counts.empty:
        print("[WARN] Missing values detected:")
        print(na_counts)

    # calendar gaps
    max_gap_info = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date")
        gaps = g["date"].diff().dt.days.dropna()
        if not gaps.empty:
            max_gap = gaps.max()
            if max_gap > 5:
                max_gap_info.append((ticker, int(max_gap)))

    if max_gap_info:
        print("[WARN] Large calendar gaps (>5 days) detected:")
        for ticker, gap in max_gap_info:
            print(f"  - {ticker}: max gap {gap} days")
    else:
        print("No large calendar gaps detected.")

    print("Quality checks completed.\n")


def build_processed_prices(force_rebuild: bool = False, run_checks: bool = True) -> pd.DataFrame:
    """
    Build prices used by all agents

    Params of func:
    force_rebuild: for redo of dataset, if set to True erases old file makes new 
    run_checks: if True, runs quality check
    """
    # reload from processed if available
    if PROCESSED_PRICES_PARQUET.exists() and not force_rebuild:
        print(f"Loading existing processed prices from {PROCESSED_PRICES_PARQUET}")
        return pd.read_parquet(PROCESSED_PRICES_PARQUET)

    if PROCESSED_PRICES_CSV.exists() and not force_rebuild:
        print(f"Loading existing processed prices from {PROCESSED_PRICES_CSV}")
        return pd.read_csv(PROCESSED_PRICES_CSV, parse_dates=["date"])

    # raw data
    if RAW_PRICES_CSV.exists():
        raw = load_raw_prices()
    else:
        raw = download_prices(save=True)

    # long format and features
    long = _prices_to_long(raw)
    long = _add_return_features(long)

    # quality check
    if run_checks:
        _run_quality_checks(long)

    # save to parquet or csv as backup
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        long.to_parquet(PROCESSED_PRICES_PARQUET)
        print(f"Saved processed prices to {PROCESSED_PRICES_PARQUET}")
    except ImportError:
        print(
            "[WARN] Could not write Parquet (missing pyarrow/fastparquet). "
            "Falling back to CSV."
        )
        long.to_csv(PROCESSED_PRICES_CSV, index=False)
        print(f"Saved processed prices to {PROCESSED_PRICES_CSV}")

    return long

def load_processed_prices(build_if_missing: bool = True) -> pd.DataFrame:
    """
    Load the processed prices dataset.

    If file is missing and build_if_missing=True, it will be built first.
    """
    if PROCESSED_PRICES_PARQUET.exists():
        return pd.read_parquet(PROCESSED_PRICES_PARQUET)

    if PROCESSED_PRICES_CSV.exists():
        return pd.read_csv(PROCESSED_PRICES_CSV, parse_dates=["date"])

    if build_if_missing:
        print("Processed prices not found. Building now...")
        return build_processed_prices(force_rebuild=True)

    raise FileNotFoundError("Processed prices file not found.")

if __name__ == "__main__":
    print("Building processed price dataset…")
    df = build_processed_prices(force_rebuild=True)
    print(df.head())
    print("\nCompleted successfully.")
