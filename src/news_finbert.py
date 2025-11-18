# src/news_finbert.py

from __future__ import annotations
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# ============================================================
# 1. Load FinBERT model
# ============================================================

def load_finbert():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


# ============================================================
# 2. Single-text prediction
# ============================================================

def finbert_predict(text: str, tokenizer, model) -> dict:
    """Return dict with positive/negative/neutral probabilities."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).numpy()[0]

    return {
        "negative": float(probs[0]),
        "neutral": float(probs[1]),
        "positive": float(probs[2]),
    }


# ============================================================
# 3. Apply FinBERT to all articles (row-wise)
# ============================================================

def add_finbert_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds FinBERT sentiment scores for each article.

    Uses: title + desc  (your current column names)
    New columns:
      - finbert_positive
      - finbert_negative
      - finbert_neutral
      - finbert_label   (+1 / 0 / -1)
    """
    df = news_df.copy()
    tokenizer, model = load_finbert()

    pos_scores = []
    neg_scores = []
    neu_scores = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="FinBERT sentiment"):
        text = (str(row.get("title", "")) + " " + str(row.get("desc", ""))).strip()

        scores = finbert_predict(text, tokenizer, model)

        pos_scores.append(scores["positive"])
        neg_scores.append(scores["negative"])
        neu_scores.append(scores["neutral"])

        # Simple label: +1 positive, -1 negative, 0 neutral
        if scores["positive"] > scores["negative"]:
            labels.append(1)
        elif scores["negative"] > scores["positive"]:
            labels.append(-1)
        else:
            labels.append(0)

    df["finbert_positive"] = pos_scores
    df["finbert_negative"] = neg_scores
    df["finbert_neutral"] = neu_scores
    df["finbert_label"] = labels

    return df


# ============================================================
# 4. Aggregate daily sentiment per (ticker, date)
# ============================================================

def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates FinBERT sentiment per (ticker, date).

    Input: df with columns `ticker`, `date`, finbert_* columns
    Output: one row per (ticker, date) with:
      - finbert_pos_sum
      - finbert_neg_sum
      - finbert_neu_sum
      - finbert_label_sum (net)
      - n_articles
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    daily = (
        df.groupby(["ticker", "date"])
          .agg(
              finbert_pos_sum=("finbert_positive", "sum"),
              finbert_neg_sum=("finbert_negative", "sum"),
              finbert_neu_sum=("finbert_neutral", "sum"),
              finbert_label_sum=("finbert_label", "sum"),
              n_articles=("title", "count"),
          )
          .reset_index()
    )

    return daily
