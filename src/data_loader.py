
import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from .config import TICKERS, START_DATE, END_DATE

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
        root = os.path.dirname(os.path.dirname(__file__))  # project root
        save_path = os.path.join(root, "data/raw/prices_yahoo.csv")
        data.to_csv(save_path)
        print(f"Saved to {save_path}")

    return data

def load_prices():
    """
    Loads prices from yahoo
    """
    df = pd.read_csv(save_path, index_col=0, parse_dates=True)
    return df