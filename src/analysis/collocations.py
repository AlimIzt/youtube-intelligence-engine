"""Lab 1 lexical analysis: word frequency, n-grams, and PMI collocations.

Ports the Lab 1 NLTK collocation work: FreqDist for top words, n-grams, and
BigramCollocationFinder / TrigramCollocationFinder ranked by PMI with a minimum
frequency filter.
"""
from __future__ import annotations

import pandas as pd

from src.preprocessing.nltk_clean import clean_tokens
from src.utils.nltk_setup import ensure_nltk


def _all_tokens(texts: list[str]) -> list[str]:
    tokens: list[str] = []
    for t in texts:
        tokens.extend(clean_tokens(t))
    return tokens


def word_frequencies(texts: list[str], n: int = 30) -> pd.DataFrame:
    """Top-n most frequent tokens (NLTK FreqDist)."""
    ensure_nltk()
    from nltk.probability import FreqDist

    fd = FreqDist(_all_tokens(texts))
    return pd.DataFrame(fd.most_common(n), columns=["word", "count"])


def top_ngrams(texts: list[str], n: int = 2, top: int = 20) -> pd.DataFrame:
    """Most frequent raw n-grams (NLTK ngrams + FreqDist)."""
    ensure_nltk()
    from nltk.probability import FreqDist
    from nltk.util import ngrams

    grams = list(ngrams(_all_tokens(texts), n))
    fd = FreqDist(grams)
    rows = [(" ".join(g), c) for g, c in fd.most_common(top)]
    return pd.DataFrame(rows, columns=[f"{n}-gram", "count"])


def pmi_collocations(
    texts: list[str], n: int = 2, top: int = 20, min_freq: int = 3
) -> pd.DataFrame:
    """Top PMI-scored bigrams (n=2) or trigrams (n=3)."""
    ensure_nltk()
    from nltk.collocations import (
        BigramAssocMeasures,
        BigramCollocationFinder,
        TrigramAssocMeasures,
        TrigramCollocationFinder,
    )

    words = _all_tokens(texts)
    if n == 2:
        measures = BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(words)
    elif n == 3:
        measures = TrigramAssocMeasures()
        finder = TrigramCollocationFinder.from_words(words)
    else:
        raise ValueError("n must be 2 or 3")

    finder.apply_freq_filter(min_freq)
    best = finder.nbest(measures.pmi, top)
    rows = [(" ".join(g), finder.ngram_fd[g]) for g in best]
    return pd.DataFrame(rows, columns=[f"{n}-gram", "freq"])
