# src/run_finbert_sentiment.py

import pandas as pd
from pathlib import Path

from src.news_finbert import add_finbert_sentiment, aggregate_daily_sentiment


def main():
    print("[1] Starting FinBERT sentiment pipeline...")

    # 1. Load news you already saved with your current script
    news_path = "data/raw/newsapi_etf_universe.csv"
    print(f"[2] Loading news from {news_path} ...")
    news_df = pd.read_csv(news_path)
    print(f"[2] Loaded {len(news_df)} news rows")

    # 2. Run FinBERT sentiment
    print("[3] Running FinBERT on title+desc...")
    news_with_sentiment = add_finbert_sentiment(news_df)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    news_with_sentiment.to_csv("data/processed/news_with_finbert.csv", index=False)
    print("[3] Saved per-article sentiment to data/processed/news_with_finbert.csv")

    # 3. Aggregate per (ticker, date)
    print("[4] Aggregating daily sentiment per ETF...")
    daily_sentiment = aggregate_daily_sentiment(news_with_sentiment)
    daily_sentiment.to_csv("data/processed/daily_finbert_sentiment.csv", index=False)
    print("[4] Saved daily sentiment to data/processed/daily_finbert_sentiment.csv")

    print("[5] Preview:")
    print(daily_sentiment.head())


if __name__ == "__main__":
    main()
