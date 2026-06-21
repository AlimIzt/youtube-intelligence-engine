"""Topic modeling with BERTopic.

Supports a general model over all comments and per-sentiment models, as the
deliverables require ("topic modeling (general, per sentiment)").
"""
from __future__ import annotations

import pandas as pd


def fit_topics(texts: list[str], min_topic_size: int = 30):
    """Fit a BERTopic model. Returns (model, topics, probs)."""
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    model = BERTopic(
        vectorizer_model=vectorizer,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, probs = model.fit_transform(texts)
    return model, topics, probs


def topic_table(model) -> pd.DataFrame:
    """Readable topic -> top words table (drops the -1 outlier topic)."""
    info = model.get_topic_info()
    info = info[info.Topic != -1].copy()
    info["top_words"] = info.Topic.map(
        lambda t: ", ".join(w for w, _ in model.get_topic(t)[:8])
    )
    return info[["Topic", "Count", "top_words"]].reset_index(drop=True)


def add_topics(df: pd.DataFrame, text_col: str = "text", min_topic_size: int = 30):
    """Attach a general topic id to each row. Returns (df, model)."""
    df = df.copy()
    model, topics, _ = fit_topics(df[text_col].fillna("").tolist(), min_topic_size)
    df["topic"] = topics
    return df, model


def topics_per_sentiment(df: pd.DataFrame, text_col: str = "text") -> dict:
    """Fit a separate topic model per sentiment class."""
    results = {}
    for sent, group in df.groupby("sentiment"):
        if len(group) < 50:
            continue
        model, _, _ = fit_topics(group[text_col].fillna("").tolist())
        results[sent] = topic_table(model)
    return results
