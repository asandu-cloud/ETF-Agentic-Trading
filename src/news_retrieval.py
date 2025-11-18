"""
ETF News Retrieval using NewsAPI
Retrieves news for ETFs: QQQ, SPXL, TLT, TQQQ
Only ETF-level queries are used to avoid rate limits.
Includes extraction of relevant keywords for sentiment analysis.
"""

from __future__ import annotations
from datetime import datetime
import time
import random
from typing import List, Tuple
import re

import pandas as pd
from newsapi import NewsApiClient
from pathlib import Path

# ===============================================
# 1. Configuration
# ===============================================

TICKERS = ["QQQ", "SPXL", "TLT", "TQQQ"]
START_DATE = "2010-01-01"

# ETF -> key companies (for keywords)
ETF_COMPANIES = {
    "QQQ": ["Apple", "Microsoft", "Amazon", "Nvidia", "Meta Platforms", "Alphabet", "Tesla"],
    "TQQQ": ["Apple", "Microsoft", "Amazon", "Nvidia", "Meta Platforms", "Alphabet", "Tesla"],
    "SPXL": ["Apple", "Microsoft", "Amazon", "Nvidia", "Alphabet", "Berkshire Hathaway", "JPMorgan Chase"],
    "TLT": ["U.S. Treasuries", "US Treasuries", "Treasury bonds", "Federal Reserve", "Fed"],
}

# ETF -> sector/macro terms (for keywords)
ETF_INDUSTRY_TERMS = {
    "QQQ": ["Nasdaq 100", "technology sector", "big tech", "semiconductor stocks", "growth stocks"],
    "TQQQ": ["Nasdaq 100", "technology sector", "big tech", "leveraged ETFs", "3x ETFs"],
    "SPXL": ["S&P 500", "US stock market", "US equities", "bull market", "stock market rally"],
    "TLT": ["long-term Treasury yields", "bond market", "interest rates", "yield curve", "rate hikes", "rate cuts", "inflation expectations"],
}

# Sentiment keywords (optional, for sentiment analysis)
SENTIMENT_POS = ["rally", "gain", "beat", "surge", "strong", "growth", "record"]
SENTIMENT_NEG = ["drop", "fall", "miss", "weak", "loss", "slump", "downgrade"]

ALL_KEYWORDS = (
    [item.lower() for sub in ETF_COMPANIES.values() for item in sub] +
    [item.lower() for sub in ETF_INDUSTRY_TERMS.values() for item in sub] +
    SENTIMENT_POS + SENTIMENT_NEG
)

# ===============================================
# 2. Initialize NewsAPI
# ===============================================

# IMPORTANT: replace with your actual API key
NEWSAPI_KEY = "bee94180b9994146a84784ba51bee662"
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# ===============================================
# 3. Helper functions
# ===============================================

def build_search_queries_for_etf(ticker: str) -> List[Tuple[str, str]]:
    """
    Return ETF-only search queries.
    """
    etf_only_queries = {
        "QQQ": ["QQQ ETF", "Invesco QQQ ETF", "Nasdaq 100 ETF"],
        "TQQQ": ["TQQQ ETF", "ProShares UltraPro QQQ", "3x Nasdaq 100 ETF"],
        "SPXL": ["SPXL ETF", "Direxion Daily S&P 500 Bull 3X Shares", "3x S&P 500 ETF"],
        "TLT": ["TLT ETF", "iShares 20+ Year Treasury Bond ETF", "long duration Treasury bond ETF"],
    }
    return [(q, "etf") for q in etf_only_queries.get(ticker, [f"{ticker} ETF"])]

def extract_relevant_keywords(text: str) -> list:
    """
    Extract relevant ETF/company/macro and sentiment keywords from text.
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    found = [kw for kw in ALL_KEYWORDS if kw in text]
    return list(set(found))

def _fetch_news_single_query(
    query: str,
    start_date: str,
    end_date: str,
    max_results: int = 50,
) -> pd.DataFrame:
    """
    Fetch news from NewsAPI for one query.
    """
    try:
        response = newsapi.get_everything(
            q=query,
            from_param=start_date,
            to=end_date,
            language="en",
            sort_by="relevancy",
            page_size=max_results,
        )
    except Exception as e:
        print(f"[ERROR] Request for '{query}' failed: {e}")
        return pd.DataFrame(columns=["title", "media", "date", "desc", "link"])

    if response.get("status") != "ok":
        print(f"[WARN] Query='{query}' failed: {response.get('message')}")
        return pd.DataFrame(columns=["title", "media", "date", "desc", "link"])

    articles = response.get("articles", [])
    if not articles:
        return pd.DataFrame(columns=["title", "media", "date", "desc", "link"])

    df = pd.DataFrame(articles)
    df.rename(columns={
        "source": "media",
        "publishedAt": "date",
        "url": "link",
        "description": "desc",
    }, inplace=True)
    df["media"] = df["media"].apply(lambda x: x.get("name") if isinstance(x, dict) else x)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df[["title", "media", "date", "desc", "link"]]

# ===============================================
# 4. Fetch news for a single ETF
# ===============================================

def fetch_news_for_etf(
    ticker: str,
    start_date: str | datetime,
    end_date: str | datetime | None = None,
) -> pd.DataFrame:

    if isinstance(start_date, str):
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start_dt = start_date

    if end_date is None:
        end_dt = datetime.today()
    elif isinstance(end_date, str):
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end_dt = end_date

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    queries = build_search_queries_for_etf(ticker)
    all_frames: List[pd.DataFrame] = []

    for q, qtype in queries:
        print(f"[{ticker}] Fetching news for '{q}' ({qtype})")

        df_q = _fetch_news_single_query(query=q, start_date=start_str, end_date=end_str)
        # delay to respect API limits
        time.sleep(random.uniform(1.5, 2.5))

        if df_q.empty:
            continue

        df_q["ticker"] = ticker
        df_q["query"] = q
        df_q["query_type"] = qtype
        # extract relevant keywords for sentiment
        df_q["relevant_keywords"] = (df_q["title"].fillna("") + " " + df_q["desc"].fillna("")).apply(extract_relevant_keywords)

        all_frames.append(df_q)

    if not all_frames:
        return pd.DataFrame(columns=["ticker","query","query_type","title","media","date","desc","link","relevant_keywords"])

    out = pd.concat(all_frames, ignore_index=True)
    out = out.drop_duplicates(subset=["ticker", "link"])
    return out

# ===============================================
# 5. Fetch news for entire ETF universe
# ===============================================

def fetch_news_for_universe(
    tickers: List[str] = None,
    start_date: str | datetime = START_DATE,
    end_date: str | datetime | None = None,
) -> pd.DataFrame:
    if tickers is None:
        tickers = TICKERS

    all_frames: List[pd.DataFrame] = []
    for t in tickers:
        df_t = fetch_news_for_etf(t, start_date, end_date)
        all_frames.append(df_t)

    if not all_frames:
        return pd.DataFrame(columns=["ticker","query","query_type","title","media","date","desc","link","relevant_keywords"])

    return pd.concat(all_frames, ignore_index=True)

# ===============================================
# 6. CLI / main
# ===============================================

if __name__ == "__main__":
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    news_df = fetch_news_for_universe(
        tickers=TICKERS,
        start_date="2025-11-01",
    )

    print(news_df.head())
    print(f"\nTotal articles downloaded: {len(news_df)}")

    out_path = "data/raw/newsapi_etf_universe.csv"
    news_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
