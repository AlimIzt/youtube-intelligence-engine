"""Keyword extraction.

Two strategies:
- corpus-level keywords with KeyBERT (semantic, uses the Ollama-free MiniLM).
- fast per-row keywords with YAKE (unsupervised, no model download).
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=1)
def _yake():
    import yake

    return yake.KeywordExtractor(lan="en", n=2, top=8, dedupLim=0.9)


def yake_keywords(text: str) -> list[str]:
    if not text:
        return []
    return [kw for kw, _score in _yake().extract_keywords(text)]


def add_keywords(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    df = df.copy()
    df["keywords"] = df[text_col].fillna("").map(yake_keywords)
    return df


def corpus_keywords(texts: list[str], top_n: int = 20) -> list[tuple[str, float]]:
    """KeyBERT keywords over the joined corpus (semantic, higher quality)."""
    from keybert import KeyBERT

    kb = KeyBERT()  # default all-MiniLM-L6-v2
    joined = " ".join(texts)[:200_000]  # cap for memory
    return kb.extract_keywords(
        joined, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n
    )


def top_keywords(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    counter: Counter = Counter()
    for row in df["keywords"]:
        counter.update(row)
    return pd.DataFrame(counter.most_common(n), columns=["keyword", "count"])
