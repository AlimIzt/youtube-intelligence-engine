"""Lab 1 cleaning pipeline (NLTK).

Faithful port of the Lab 1 `clean_text`: lowercase, strip URLs/HTML, drop
emojis/non-ASCII, normalise repeated chars/punctuation, remove punctuation,
tokenize, remove NLTK + custom stopwords, keep alphabetic tokens, lemmatize.

This complements the lightweight regex cleaner in `clean.py`. Use this when you
want tokenized, lemmatized output (e.g. for collocations / topic modeling).
"""
from __future__ import annotations

import re
import string
from functools import lru_cache

from src.utils.nltk_setup import ensure_nltk

# Custom domain stopwords. The Lab 1 list was restaurant-specific; this one is
# tuned for YouTube/GTA-trailer comment noise. Add tokens after inspecting
# FreqDist.most_common (see analysis/collocations.py).
CUSTOM_STOP_WORDS = {
    "game", "video", "trailer", "youtube", "comment", "like", "watch",
    "im", "dont", "thats", "got", "get", "go", "one", "also", "would",
    "really", "back", "us", "u", "ur", "lol", "yeah", "gonna", "wanna",
}


@lru_cache(maxsize=1)
def _resources():
    ensure_nltk()
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    nltk_stop = set(stopwords.words("english"))
    all_stop = nltk_stop | CUSTOM_STOP_WORDS
    return all_stop, WordNetLemmatizer()


def clean_tokens(text: str) -> list[str]:
    """Full Lab 1 cleaning pipeline → list of clean lemmatized tokens."""
    from nltk.tokenize import word_tokenize

    all_stop, lemmatizer = _resources()
    if not isinstance(text, str):
        return []

    # 1. lowercase
    text = text.lower()
    # 2. remove URLs / HTML
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    # 3. remove emojis / non-ASCII
    text = text.encode("ascii", "ignore").decode("ascii")
    # 4. normalise repeated chars / punctuation
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # "loooove" -> "loove"
    text = re.sub(r"[!?]{2,}", "!", text)
    text = re.sub(r"\.{2,}", ".", text)
    # 5. remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 6. tokenize
    tokens = word_tokenize(text)
    # 7. remove stopwords (NLTK + custom)
    tokens = [t for t in tokens if t not in all_stop]
    # 8. keep only alphabetic tokens
    tokens = [t for t in tokens if t.isalpha()]
    # 9. lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def clean_join(text: str) -> str:
    """Same pipeline but returns a single cleaned string."""
    return " ".join(clean_tokens(text))
