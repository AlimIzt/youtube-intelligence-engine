"""Zero-shot NER with GLiNER (LLM_vs_REBEL_KG_Comparison notebook).

Unlike spaCy's fixed tag set in src/analysis/ner.py (PERSON/ORG/GPE/...),
GLiNER matches ARBITRARY label strings supplied at inference time, so it can
find domain concepts like "game feature" or "platform" without any training.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

import pandas as pd

GLINER_MODEL = "urchade/gliner_mediumv2.1"
DEFAULT_LABELS = ("game feature", "character name", "platform",
                  "release date", "company/studio", "price")


@lru_cache(maxsize=1)
def _model():
    from gliner import GLiNER

    return GLiNER.from_pretrained(GLINER_MODEL)


def extract_domain_entities(
    texts: list[str],
    labels: tuple[str, ...] | list[str] = DEFAULT_LABELS,
    threshold: float = 0.4,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Aggregate GLiNER predictions into (entity, label, count) rows."""
    model = _model()
    labels = list(labels)
    counter: Counter = Counter()
    for i in range(0, len(texts), batch_size):
        batch = [(t or "")[:1500] for t in texts[i:i + batch_size]]
        if hasattr(model, "batch_predict_entities"):
            batches = model.batch_predict_entities(batch, labels, threshold=threshold)
        else:  # pragma: no cover - older gliner versions
            batches = [model.predict_entities(t, labels, threshold=threshold)
                       for t in batch]
        for preds in batches:
            counter.update((p["text"].strip().lower(), p["label"]) for p in preds)
    rows = [(ent, lab, c) for (ent, lab), c in counter.most_common()]
    return pd.DataFrame(rows, columns=["entity", "label", "count"])


def main() -> None:
    demo = ["The graphics on PS5 look insane, Lucia is such a great protagonist",
            "Rockstar better not delay this past 2026, I'm not paying 80 dollars"]
    print(extract_domain_entities(demo))


if __name__ == "__main__":
    main()
