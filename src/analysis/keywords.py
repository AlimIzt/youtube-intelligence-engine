"""Per-row keyword extraction with YAKE (unsupervised, no model download)."""
from __future__ import annotations

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
