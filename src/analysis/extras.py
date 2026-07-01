"""Extra NLP methods surfaced in the dashboard.

Emotion detection (NRC lexicon), subjectivity (TextBlob), language detection
(langdetect), emoji frequency, readability/text stats (textstat), distinctive
TF-IDF terms per sentiment, and question detection.
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

EMOTIONS = ("joy", "trust", "anticipation", "surprise",
            "sadness", "fear", "anger", "disgust")


_NLTK_READY = False


def _nrc(text: str):
    from nrclex import NRCLex

    global _NLTK_READY
    if not _NLTK_READY:
        from src.utils.nltk_setup import ensure_nltk
        ensure_nltk()
        _NLTK_READY = True
    try:
        n = NRCLex()
        n.load_raw_text(text or "")
    except TypeError:
        n = NRCLex(text or "")
    return n


def emotion_scores(texts: list[str]) -> pd.DataFrame:
    counter: Counter = Counter()
    for t in texts:
        counter.update({k: v for k, v in _nrc(t).raw_emotion_scores.items()
                        if k in EMOTIONS})
    rows = [(e, counter.get(e, 0)) for e in EMOTIONS]
    return pd.DataFrame(rows, columns=["emotion", "count"])


def top_emotion_comments(texts: list[str], emotion: str, top: int = 10) -> pd.DataFrame:
    scored = []
    for t in texts:
        s = _nrc(t).affect_frequencies.get(emotion, 0.0)
        if s > 0:
            scored.append((t, round(s, 3)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return pd.DataFrame(scored[:top], columns=["comment", f"{emotion} intensity"])


def subjectivity(texts: list[str]) -> pd.DataFrame:
    from textblob import TextBlob

    rows = [(t, TextBlob(t or "").sentiment.polarity,
             TextBlob(t or "").sentiment.subjectivity) for t in texts]
    return pd.DataFrame(rows, columns=["text", "polarity", "subjectivity"])


def detect_languages(texts: list[str]) -> pd.DataFrame:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    counter: Counter = Counter()
    for t in texts:
        try:
            counter[detect(t)] += 1
        except Exception:
            counter["unknown"] += 1
    return pd.DataFrame(counter.most_common(), columns=["language", "count"])


def emoji_counts(texts: list[str], top: int = 20) -> pd.DataFrame:
    import emoji

    counter: Counter = Counter()
    for t in texts:
        counter.update(m["emoji"] for m in emoji.emoji_list(t or ""))
    return pd.DataFrame(counter.most_common(top), columns=["emoji", "count"])


def text_stats(texts: list[str]) -> dict:
    import textstat

    joined = " ".join(t for t in texts if t)
    lengths = [len((t or "").split()) for t in texts]
    return {
        "avg_words": sum(lengths) / max(len(lengths), 1),
        "reading_ease": textstat.flesch_reading_ease(joined),
        "grade_level": textstat.flesch_kincaid_grade(joined),
        "lengths": lengths,
    }


def distinctive_terms(df: pd.DataFrame, label_col: str = "sentiment",
                      text_col: str = "text", top: int = 12) -> dict[str, list[str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    out: dict[str, list[str]] = {}
    grouped = {lab: " ".join(g[text_col].dropna().astype(str))
               for lab, g in df.groupby(label_col)}
    if len(grouped) < 2:
        return out
    vec = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    m = vec.fit_transform(grouped.values())
    terms = vec.get_feature_names_out()
    for i, lab in enumerate(grouped):
        row = m[i].toarray().ravel()
        out[lab] = [terms[j] for j in row.argsort()[::-1][:top]]
    return out


_Q = re.compile(r"\?|^(who|what|when|where|why|how|is|are|will|does|did|can)\b",
                re.IGNORECASE)


def questions(texts: list[str]) -> list[str]:
    return [t for t in texts if t and "?" in t]


def question_share(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return len(questions(texts)) / len(texts)
