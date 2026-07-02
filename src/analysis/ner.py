"""Named Entity Recognition with spaCy (Lab 2 style, batched)."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from config import settings

# Entity types worth keeping for game/social-media discussion.
KEEP = {"PERSON", "ORG", "PRODUCT", "GPE", "LOC", "WORK_OF_ART", "EVENT", "NORP"}


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    try:
        nlp = spacy.load(settings.spacy_model, disable=["lemmatizer", "tagger"])
    except OSError as e:  # pragma: no cover
        raise RuntimeError(
            f"spaCy model '{settings.spacy_model}' missing. Run:\n"
            f"  python -m spacy download {settings.spacy_model}"
        ) from e
    return nlp


def extract_entities(texts: list[str], batch_size: int = 256) -> list[list[tuple[str, str]]]:
    """Return [(entity_text, label), ...] per input text."""
    nlp = _nlp()
    out: list[list[tuple[str, str]]] = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        out.append([(e.text.strip(), e.label_) for e in doc.ents if e.label_ in KEEP])
    return out


def add_entities(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    df = df.copy()
    ents = extract_entities(df[text_col].fillna("").tolist())
    df["entities"] = [[t for t, _ in row] for row in ents]
    df["entity_labels"] = [[f"{t}:{l}" for t, l in row] for row in ents]
    return df
