"""Run the full enrichment pipeline: clean -> NER -> keywords -> sentiment -> topics."""
from __future__ import annotations

import pandas as pd

from config import settings
from src.analysis.keywords import add_keywords
from src.analysis.ner import add_entities
from src.analysis.sentiment import add_sentiment
# from src.analysis.topics import add_topics  # Temporarily disabled because BERTopic crashes on Mac
from src.preprocessing.clean import clean_dataframe


def run(spell: bool = False, sentiment_method: str = "vader") -> pd.DataFrame:
    print("1/5  loading raw comments ...")
    df = pd.read_csv(settings.raw_csv)

    print("2/5  cleaning ...")
    df = clean_dataframe(df, spell=spell)

    print("3/5  NER + keywords ...")
    df = add_entities(df)
    df = add_keywords(df)

    print("4/5  sentiment ...")
    df = add_sentiment(df, method=sentiment_method)

    print("5/5  topic modeling ...")
    # Original BERTopic call:
    # df, _model = add_topics(df)
    #
    # Temporary fallback:
    # -1 means "topic not assigned".
    # This keeps the rest of the pipeline working, especially vector index building.
    df["topic"] = -1

    # Parquet preserves list columns (entities/keywords); csv would stringify them.
    df.to_parquet(settings.enriched_parquet, index=False)
    df.to_csv(settings.enriched_parquet.with_suffix(".csv"), index=False)

    print(f"\nDone. Enriched {len(df):,} comments → {settings.enriched_parquet}")
    return df


if __name__ == "__main__":
    run()