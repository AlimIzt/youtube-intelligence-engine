"""Sentiment analysis.

Default: VADER (fast, social-media tuned, no downloads). Optionally switch to a
transformer model (cardiffnlp twitter-roberta) for higher accuracy.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from config import settings

LABELS = ("negative", "neutral", "positive")


@lru_cache(maxsize=1)
def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def vader_label(text: str) -> tuple[str, float]:
    score = _vader().polarity_scores(text or "")["compound"]
    if score >= 0.05:
        return "positive", score
    if score <= -0.05:
        return "negative", score
    return "neutral", score


@lru_cache(maxsize=1)
def _transformer():
    from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model=settings.sentiment_model,
        truncation=True,
        max_length=256,
    )


def add_sentiment(
    df: pd.DataFrame, text_col: str = "text", method: str = "vader"
) -> pd.DataFrame:
    df = df.copy()
    texts = df[text_col].fillna("").tolist()
    if method == "vader":
        pairs = [vader_label(t) for t in texts]
        df["sentiment"] = [p[0] for p in pairs]
        df["sentiment_score"] = [p[1] for p in pairs]
    elif method == "transformer":
        clf = _transformer()
        preds = clf(texts, batch_size=32)
        df["sentiment"] = [p["label"].lower() for p in preds]
        df["sentiment_score"] = [p["score"] for p in preds]
    else:
        raise ValueError("method must be 'vader' or 'transformer'")
    return df


def sentiment_distribution(df: pd.DataFrame) -> pd.Series:
    return df["sentiment"].value_counts(normalize=True).reindex(LABELS).fillna(0)
