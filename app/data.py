"""Data loading, the sidebar sample scope, and all cached analytics.

Every function that computes something over the comments is keyed on
(n, strategy) so the whole dashboard respects the sidebar's sample-scope
controls, and cached with @st.cache_data so tab switches stay instant.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from config import settings


# ----------------------------------------------------------------------------
# loading + scope
# ----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    if settings.enriched_parquet.exists():
        df = pd.read_parquet(settings.enriched_parquet)
    elif settings.clean_csv.exists():
        df = pd.read_csv(settings.clean_csv)
    else:
        return pd.DataFrame()
    if "published_at" in df:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    return df


@dataclass(frozen=True)
class Scope:
    """The sidebar-selected slice of the dataset, passed to every tab."""

    full: pd.DataFrame
    df: pd.DataFrame
    n: int
    strategy: str
    use_all: bool
    texts: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return "all" if self.use_all else self.strategy.lower()


def sidebar_scope(full: pd.DataFrame) -> Scope:
    """Render the sidebar controls and return the selected Scope."""
    with st.sidebar:
        st.header("Analysis scope")
        st.caption("These settings control how many comments every chart and "
                   "NLP method below is computed on.")
        use_all = st.toggle(f"Use all {len(full):,} comments", value=False)
        n = len(full) if use_all else st.slider(
            "Comments to analyze", 100, len(full),
            min(1000, len(full)), step=100)
        strategy = st.selectbox(
            "Which comments to pick",
            ["First in dataset", "Random sample", "Most liked", "Newest"],
            disabled=use_all)
        st.caption("Tip: heavier methods (relations, emotions, languages) get "
                   "slow above ~3,000 comments.")
    df = scoped(n, strategy)
    texts = df["text"].dropna().astype(str).tolist()
    return Scope(full=full, df=df, n=n, strategy=strategy,
                 use_all=use_all, texts=texts)


@st.cache_data
def scoped(n: int, strategy: str) -> pd.DataFrame:
    df = load_data()
    if n >= len(df):
        return df
    if strategy == "Random sample":
        return df.sample(n, random_state=42)
    if strategy == "Most liked" and "likes" in df:
        return df.sort_values("likes", ascending=False).head(n)
    if strategy == "Newest" and "published_at" in df:
        return df.sort_values("published_at", ascending=False).head(n)
    return df.head(n)


def _texts(n: int, strategy: str) -> list[str]:
    return scoped(n, strategy)["text"].dropna().astype(str).tolist()


# ----------------------------------------------------------------------------
# cached analytics (all keyed on the sidebar scope)
# ----------------------------------------------------------------------------
@st.cache_data
def list_column_counts(n: int, strategy: str, col: str, top: int) -> pd.DataFrame:
    d = scoped(n, strategy)
    counter: Counter = Counter()
    if col not in d:
        return pd.DataFrame(columns=[col, "count"])
    for row in d[col]:
        if row is not None:
            counter.update([str(x) for x in row])
    return pd.DataFrame(counter.most_common(top), columns=[col, "count"])


@st.cache_data
def collocations(n: int, strategy: str, gram: int, top: int) -> pd.DataFrame:
    from src.analysis.collocations import pmi_collocations
    return pmi_collocations(_texts(n, strategy),
                            n=gram, top=top)


@st.cache_data
def ngrams(n: int, strategy: str, gram: int, top: int) -> pd.DataFrame:
    from src.analysis.collocations import top_ngrams
    return top_ngrams(_texts(n, strategy),
                      n=gram, top=top)


@st.cache_data
def word_freq(n: int, strategy: str, top: int) -> pd.DataFrame:
    from src.analysis.collocations import word_frequencies
    return word_frequencies(_texts(n, strategy), top)


@st.cache_resource
def _spacy():
    import spacy
    return spacy.load(settings.spacy_model, disable=["ner", "lemmatizer"])


@st.cache_data
def pos_distribution(n: int, strategy: str) -> pd.DataFrame:
    nlp = _spacy()
    t = _texts(n, strategy)
    counter: Counter = Counter()
    for doc in nlp.pipe(t, batch_size=128):
        counter.update(tok.pos_ for tok in doc if not tok.is_space)
    return pd.DataFrame(counter.most_common(), columns=["POS", "count"])


@st.cache_data
def relations(n: int, strategy: str) -> list[str]:
    from src.analysis.relations import relations_for_corpus
    t = _texts(n, strategy)
    return relations_for_corpus(t, limit=len(t))


@st.cache_data
def kg_dot(n: int, strategy: str, max_edges: int) -> str:
    from src.analysis.relations import knowledge_graph_dot
    return knowledge_graph_dot(_texts(n, strategy),
                               max_edges=max_edges)


@st.cache_data
def wordcloud_png(n: int, strategy: str) -> bytes:
    import io
    from wordcloud import WordCloud
    from src.preprocessing.nltk_clean import clean_join

    t = _texts(n, strategy)
    text = " ".join(clean_join(x) for x in t) or "none"
    wc = WordCloud(width=900, height=400, background_color="#0b0b12",
                   colormap="plasma", max_words=120).generate(text)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data
def emotions(n: int, strategy: str) -> pd.DataFrame:
    from src.analysis.extras import emotion_scores
    return emotion_scores(_texts(n, strategy))


@st.cache_data
def languages(n: int, strategy: str) -> pd.DataFrame:
    from src.analysis.extras import detect_languages
    return detect_languages(_texts(n, strategy))


@st.cache_data
def emojis(n: int, strategy: str) -> pd.DataFrame:
    from src.analysis.extras import emoji_counts
    return emoji_counts(_texts(n, strategy))


@st.cache_data
def subjectivity_df(n: int, strategy: str) -> pd.DataFrame:
    from src.analysis.extras import subjectivity
    return subjectivity(_texts(n, strategy))


@st.cache_data
def stats(n: int, strategy: str) -> dict:
    from src.analysis.extras import text_stats
    return text_stats(_texts(n, strategy))


@st.cache_data
def tfidf_terms(n: int, strategy: str) -> dict:
    from src.analysis.extras import distinctive_terms
    return distinctive_terms(scoped(n, strategy))


@st.cache_data
def gliner_entities(n: int, strategy: str, labels: tuple[str, ...]) -> pd.DataFrame:
    from src.analysis.gliner_ner import extract_domain_entities
    return extract_domain_entities(
        _texts(n, strategy), labels)
