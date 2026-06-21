"""Lab 2 Part-of-Speech tagging with multiple taggers.

Trains Unigram/Bigram/Trigram taggers (with backoff) and an HMM tagger on the
NLTK treebank corpus, plus the default Perceptron tagger (nltk.pos_tag) and
spaCy, then compares their tags token-by-token — exactly the Lab 2 comparison.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.utils.nltk_setup import ensure_nltk


@lru_cache(maxsize=1)
def _taggers():
    ensure_nltk()
    import nltk
    from nltk.corpus import treebank
    from nltk.tag import hmm

    train_data = treebank.tagged_sents()

    # Backoff chain: default -> regexp -> unigram -> bigram -> trigram (Lab 2 T2)
    default_tagger = nltk.DefaultTagger("NN")
    patterns = [
        (r".*ing$", "VBG"),
        (r".*ed$", "VBD"),
        (r".*es$", "VBZ"),
        (r".*ould$", "MD"),
        (r".*\'s$", "NN$"),
        (r".*s$", "NNS"),
        (r"^-?[0-9]+(.[0-9]+)?$", "CD"),
        (r".*", "NN"),
    ]
    regexp_tagger = nltk.RegexpTagger(patterns, backoff=default_tagger)
    unigram = nltk.UnigramTagger(train_data, backoff=regexp_tagger)
    bigram = nltk.BigramTagger(train_data, backoff=unigram)
    trigram = nltk.TrigramTagger(train_data, backoff=bigram)

    hmm_tagger = hmm.HiddenMarkovModelTrainer().train_supervised(train_data)
    return unigram, bigram, trigram, hmm_tagger


@lru_cache(maxsize=1)
def _spacy():
    import spacy

    from config import settings

    return spacy.load(settings.spacy_model)


def compare_taggers(text: str) -> pd.DataFrame:
    """Tag `text` with every tagger and return a side-by-side comparison."""
    import nltk

    unigram, bigram, trigram, hmm_tagger = _taggers()
    tokens = nltk.word_tokenize(text)

    nltk_tags = [t for _, t in nltk.pos_tag(tokens)]          # Perceptron
    uni_tags = [t for _, t in unigram.tag(tokens)]
    bi_tags = [t for _, t in bigram.tag(tokens)]
    tri_tags = [t for _, t in trigram.tag(tokens)]
    hmm_tags = [t for _, t in hmm_tagger.tag(tokens)]

    nlp = _spacy()
    spacy_tags = [tok.tag_ for tok in nlp(" ".join(tokens))][: len(tokens)]
    spacy_tags += [""] * (len(tokens) - len(spacy_tags))

    return pd.DataFrame(
        {
            "Token": tokens,
            "Perceptron": nltk_tags,
            "Unigram": uni_tags,
            "Bigram": bi_tags,
            "Trigram": tri_tags,
            "HMM": hmm_tags,
            "spaCy": spacy_tags,
        }
    )


def pos_tags(text: str) -> list[tuple[str, str]]:
    """Quick (token, tag) list using spaCy — used by relation extraction."""
    return [(t.text, t.tag_) for t in _spacy()(text)]
