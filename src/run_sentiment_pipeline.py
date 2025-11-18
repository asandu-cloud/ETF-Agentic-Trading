# src/run_sentiment_pipeline.py

print(">>> run_sentiment_pipeline.py imported / executed")  # DEBUG

import pandas as pd
from src.news_sentiment import add_finbert_sentiment, aggregate_daily_sentiment


def main():
    print("[1] Starting sentiment pipeline...")

    print("[2] Loading news from data/raw/newsapi_etf_universe.csv ...")
    news_df = pd.read_csv("data/raw/newsapi_etf_universe.csv")
    print(f"[2] Loaded {len(news_df)} news rows")

    print("[3] Running FinBERT sentiment on articles...")
    news_with_sentiment = add_finbert_sentiment(news_df)
    print("[3] FinBERT finished, saving per-article sentiment...")
    news_with_sentiment.to_csv("data/processed/news_with_finbert.csv", index=False)
    print("[3] Saved: data/processed/news_with_finbert.csv")

    print("[4] Aggregating sentiment per (ticker, date)...")
    daily_sentiment = aggregate_daily_sentiment(news_with_sentiment)
    daily_sentiment.to_csv("data/processed/daily_finbert_sentiment.csv", index=False)
    print("[4] Saved: data/processed/daily_finbert_sentiment.csv")

    print("[5] Preview of daily sentiment:")
    print(daily_sentiment.head())


if __name__ == "__main__":
    print(">>> __main__ block reached")  # DEBUG
    main()
