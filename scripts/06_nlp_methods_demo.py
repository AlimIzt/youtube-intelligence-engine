"""Step 6 (optional): demonstrate every lab NLP method on the comment data.

Runs the ported lab techniques on a sample of comments and prints their output,
so each method is reproducible and gradeable in one place:

  Lab 1  : NLTK cleaning, FreqDist, n-grams, PMI collocations, TextBlob
  Lab 2  : POS tagger comparison, chunking, Noun-Verb-Noun relations
  Lab 5  : VADER / TextBlob / transformer sentiment, TF-IDF+LogReg classifier
  (Lab 5 Part 2 pyABSA runs separately — see src/analysis/absa.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import settings


def banner(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main(sample: int = 500) -> None:
    df = pd.read_csv(settings.clean_csv)
    texts = df["text"].dropna().astype(str).tolist()
    sample_texts = texts[:sample]
    demo_sentence = "The graphics look amazing but the release date disappointed fans."

    # ---- Lab 1: lexical analysis ----
    banner("LAB 1 — Word frequency, n-grams, PMI collocations")
    from src.analysis.collocations import pmi_collocations, top_ngrams, word_frequencies

    print("Top words:\n", word_frequencies(sample_texts, 15).to_string(index=False))
    print("\nTop bigrams:\n", top_ngrams(sample_texts, 2, 10).to_string(index=False))
    print("\nPMI bigrams:\n", pmi_collocations(sample_texts, 2, 10).to_string(index=False))

    # ---- Lab 2: POS + relations ----
    banner("LAB 2 — POS tagger comparison")
    from src.analysis.pos import compare_taggers

    print(compare_taggers(demo_sentence).to_string(index=False))

    banner("LAB 2 — Chunking + Noun-Verb-Noun relations")
    from src.analysis.relations import noun_chunks, relations_for_corpus

    print("Noun chunks:", noun_chunks(demo_sentence))
    print("Relations (corpus sample):", relations_for_corpus(sample_texts, limit=300)[:15])

    # ---- Lab 5: sentiment tools ----
    banner("LAB 5 — Sentiment tool comparison (VADER vs TextBlob)")
    from src.analysis.sentiment import textblob_label, vader_label

    for t in sample_texts[:5]:
        print(f"  VADER={vader_label(t)[0]:>8}  TextBlob={textblob_label(t)[0]:>8}  | {t[:60]}")

    banner("LAB 5 — TF-IDF + LogisticRegression classifier (trained on VADER labels)")
    from src.analysis.sentiment import vader_label as _vl
    from src.analysis.text_classifier import train

    labels = [_vl(t)[0] for t in sample_texts]
    _model, report = train(sample_texts, labels)
    print(report)

    print("\nAll lab methods ran successfully.")


if __name__ == "__main__":
    main()
