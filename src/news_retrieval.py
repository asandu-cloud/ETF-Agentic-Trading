"""
News retrieval for ETF universe:
    QQQ, SPXL, TLT, TQQQ

It pulls news about:
  - the ETF itself
  - major related companies
  - sector / macro themes that impact the ETF
using the GoogleNews package.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
from GoogleNews import GoogleNews
from .config import TICKERS, START_DATE

# === 1. Universe configuration ===

TICKERS = TICKERS
START_DATE = START_DATE


# --- ETF -> key companies (hand-picked; extend if you like) ---

ETF_COMPANIES: Dict[str, List[str]] = {
    "QQQ": [
        "Apple",
        "Microsoft",
        "Amazon",
        "Nvidia",
        "Meta Platforms",
        "Alphabet",
        "Tesla",
    ],
    "TQQQ": [
        # Same underlying as QQQ (leveraged)
        "Apple",
        "Microsoft",
        "Amazon",
        "Nvidia",
        "Meta Platforms",
        "Alphabet",
        "Tesla",
    ],
    "SPXL": [
        # Big S&P 500 names (strong impact on SPXL)
        "Apple",
        "Microsoft",
        "Amazon",
        "Nvidia",
        "Alphabet",
        "Berkshire Hathaway",
        "JPMorgan Chase",
    ],
    "TLT": [
        # For TLT we mostly care about rates / macro, but include big bond-related terms
        # plus some large bond funds / institutions that often appear in commentary.
        "U.S. Treasuries",
        "US Treasuries",
        "Treasury bonds",
        "Federal Reserve",
        "Fed",
    ],
}


# --- ETF -> sector / macro themes that drive them ---

ETF_INDUSTRY_TERMS: Dict[str, List[str]] = {
    "QQQ": [
        "Nasdaq 100",
        "technology sector",
        "big tech",
        "semiconductor stocks",
        "growth stocks",
    ],
    "TQQQ": [
        "Nasdaq 100",
        "technology sector",
        "big tech",
        "leveraged ETFs",
        "3x ETFs",
    ],
    "SPXL": [
        "S&P 500",
        "US stock market",
        "US equities",
        "bull market",
        "stock market rally",
    ],
    "TLT": [
        "long-term Treasury yields",
        "bond market",
        "interest rates",
        "yield curve",
        "rate hikes",
        "rate cuts",
        "inflation expectations",
    ],
}


# === 2. Helper: build queries for each ETF ===

def build_search_queries_for_etf(ticker: str) -> List[Tuple[str, str]]:
    """
    Build a list of (query_string, query_type) for a given ETF.

    query_type is one of: "etf", "company", "industry".
    """

    queries: List[Tuple[str, str]] = []

    # ETF-level queries
    if ticker == "QQQ":
        queries.extend(
            [
                ("QQQ ETF", "etf"),
                ("Invesco QQQ ETF", "etf"),
                ("Nasdaq 100 ETF", "etf"),
            ]
        )
    elif ticker == "TQQQ":
        queries.extend(
            [
                ("TQQQ ETF", "etf"),
                ("ProShares UltraPro QQQ", "etf"),
                ("3x Nasdaq 100 ETF", "etf"),
            ]
        )
    elif ticker == "SPXL":
        queries.extend(
            [
                ("SPXL ETF", "etf"),
                ("Direxion Daily S&P 500 Bull 3X Shares", "etf"),
                ("3x S&P 500 ETF", "etf"),
            ]
        )
    elif ticker == "TLT":
        queries.extend(
            [
                ("TLT ETF", "etf"),
                ("iShares 20+ Year Treasury Bond ETF", "etf"),
                ("long duration Treasury bond ETF", "etf"),
            ]
        )
    else:
        # Fallback generic ETF query
        queries.append((f"{ticker} ETF", "etf"))

    # Company-level queries
    for name in ETF_COMPANIES.get(ticker, []):
        queries.append((name, "company"))

    # Industry / macro-level queries
    for term in ETF_INDUSTRY_TERMS.get(ticker, []):
        queries.append((term, "industry"))

    return queries


# === 3. GoogleNews wrappers ===

def _fetch_news_single_query(
    query: str,
    start_date: datetime,
    end_date: datetime,
    lang: str = "en",
    max_pages: int = 3,
) -> pd.DataFrame:
    """
    Fetch news from GoogleNews for a single query and date range.

    Returns a DataFrame with columns at least:
      ['title', 'media', 'date', 'desc', 'link']
    """
    start_str = start_date.strftime("%m/%d/%Y")
    end_str = end_date.strftime("%m/%d/%Y")

    gn = GoogleNews(start=start_str, end=end_str, lang=lang)
    gn.search(query)

    all_results = []

    # First page (after search)
    results = gn.result()
    all_results.extend(results)

    for page in range(2, max_pages + 1):
        gn.getpage(page)
        res = gn.result()
        if not res:
            break
        all_results.extend(res)

    if not all_results:
        return pd.DataFrame(columns=["title", "media", "date", "desc", "link"])

    df = pd.DataFrame(all_results)
    df = df.drop_duplicates(subset="link")

    # Normalize date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def fetch_news_for_etf(
    ticker: str,
    start_date: str | datetime,
    end_date: str | datetime | None = None,
    lang: str = "en",
    max_pages_per_query: int = 3,
) -> pd.DataFrame:
    """
    Fetch news for a single ETF, including:
      - the ETF itself
      - related companies
      - industry/macro terms

    Returns a DataFrame with:
      ['ticker', 'query', 'query_type', 'title', 'media', 'date',
       'desc', 'link']
    """

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

    # You might not want to hit all years at once, but this is the simple
    # version: single big range. For larger backtests, chunk by year.
    queries = build_search_queries_for_etf(ticker)

    all_frames = []

    for q, qtype in queries:
        print(f"[{ticker}] Fetching news for query='{q}' ({qtype})")
        df_q = _fetch_news_single_query(
            query=q,
            start_date=start_dt,
            end_date=end_dt,
            lang=lang,
            max_pages=max_pages_per_query,
        )
        if df_q.empty:
            continue

        df_q["ticker"] = ticker
        df_q["query"] = q
        df_q["query_type"] = qtype

        all_frames.append(df_q)

    if not all_frames:
        return pd.DataFrame(
            columns=[
                "ticker",
                "query",
                "query_type",
                "title",
                "media",
                "date",
                "desc",
                "link",
            ]
        )

    out = pd.concat(all_frames, ignore_index=True)

    # Reorder columns for convenience
    cols = [
        "ticker",
        "query",
        "query_type",
        "title",
        "media",
        "date",
        "desc",
        "link",
    ]
    out = out[[c for c in cols if c in out.columns]]

    # De-duplicate across queries (same article might appear for multiple terms)
    out = out.drop_duplicates(subset=["ticker", "link"])

    return out


def fetch_news_for_universe(
    tickers: List[str] = TICKERS,
    start_date: str | datetime = START_DATE,
    end_date: str | datetime | None = None,
    lang: str = "en",
    max_pages_per_query: int = 3,
) -> pd.DataFrame:
    """
    Fetch news for the whole ETF universe.

    Returns a DataFrame with all ETFs stacked.
    """

    all_etf_frames = []

    for t in tickers:
        df_t = fetch_news_for_etf(
            ticker=t,
            start_date=start_date,
            end_date=end_date,
            lang=lang,
            max_pages_per_query=max_pages_per_query,
        )
        all_etf_frames.append(df_t)

    if not all_etf_frames:
        return pd.DataFrame(
            columns=[
                "ticker",
                "query",
                "query_type",
                "title",
                "media",
                "date",
                "desc",
                "link",
            ]
        )

    universe_df = pd.concat(all_etf_frames, ignore_index=True)

    return universe_df


# === 4. CLI entry point (optional) ===

if __name__ == "__main__":
    # Example: fetch from 2010-01-01 to today and save to disk
    news_df = fetch_news_for_universe(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=None,  # -> today
        lang="en",
        max_pages_per_query=3,  # tune this up/down
    )

    print(news_df.head())
    print(f"\nTotal articles downloaded: {len(news_df)}")

    # Save to parquet so you can reuse for sentiment, keywords, etc.
    news_df.to_parquet("data/raw/google_news_etf_universe.parquet", index=False)
    print("\nSaved to data/raw/google_news_etf_universe.parquet")
