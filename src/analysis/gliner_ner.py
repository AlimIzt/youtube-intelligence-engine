"""Zero-shot domain-specific NER with GLiNER.

This module adds a flexible NER option for the dashboard. Unlike spaCy NER,
which predicts a fixed set of entity types such as PERSON, ORG, and GPE,
GLiNER can search for custom labels provided at inference time.

That makes it useful for YouTube comments about games, where we may want
domain-specific labels such as "game feature", "platform", "release date",
or "company/studio" without training a new model.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

import pandas as pd

GLINER_MODEL = "urchade/gliner_mediumv2.1"
DEFAULT_LABELS = (
    "game feature",
    "character name",
    "platform",
    "release date",
    "company/studio",
    "price",
)


@lru_cache(maxsize=1)
def _model():
    """Load and cache the GLiNER model.

    The model is relatively expensive to load, so lru_cache keeps one shared
    instance in memory for repeated dashboard calls.
    """
    from gliner import GLiNER

    return GLiNER.from_pretrained(GLINER_MODEL)


def extract_domain_entities(
    texts: list[str],
    labels: tuple[str, ...] | list[str] = DEFAULT_LABELS,
    threshold: float = 0.4,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Extract and aggregate custom-domain entities from text.

    Args:
        texts: List of comments or other text snippets to analyze.
        labels: Custom entity labels that GLiNER should search for.
        threshold: Minimum confidence score for keeping a prediction.
        batch_size: Number of texts processed at once.

    Returns:
        A dataframe with three columns:
        - entity: normalized entity text
        - label: predicted custom entity label
        - count: how often that entity-label pair appears
    """
    model = _model()
    labels = list(labels)
    counter: Counter = Counter()

    for i in range(0, len(texts), batch_size):
        batch = [(t or "")[:1500] for t in texts[i:i + batch_size]]

        if hasattr(model, "batch_predict_entities"):
            batches = model.batch_predict_entities(batch, labels, threshold=threshold)
        else:  # pragma: no cover - older gliner versions
            batches = [
                model.predict_entities(t, labels, threshold=threshold)
                for t in batch
            ]

        for preds in batches:
            counter.update((p["text"].strip().lower(), p["label"]) for p in preds)

    rows = [(ent, lab, c) for (ent, lab), c in counter.most_common()]
    return pd.DataFrame(rows, columns=["entity", "label", "count"])


def main() -> None:
    """Run a small GLiNER demo from the command line."""
    demo = [
        "The graphics on PS5 look insane, Lucia is such a great protagonist",
        "Rockstar better not delay this past 2026, I'm not paying 80 dollars",
    ]

    print(extract_domain_entities(demo))


if __name__ == "__main__":
    main()