"""Download the NLTK corpora the lab methods need (run once).

Lab 1 used: punkt, stopwords, wordnet, omw-1.4, averaged_perceptron_tagger.
Lab 2 also used: treebank (training data for the n-gram/HMM taggers).
"""
from __future__ import annotations

import nltk

RESOURCES = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "treebank",
]


def ensure_nltk() -> None:
    for res in RESOURCES:
        try:
            nltk.download(res, quiet=True)
        except Exception as e:  # pragma: no cover
            print(f"[nltk_setup] could not download {res}: {e}")


if __name__ == "__main__":
    ensure_nltk()
    print("NLTK resources ready.")
