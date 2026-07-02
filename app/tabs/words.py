"""Word cloud, frequencies, emojis, phrases, TF-IDF, and POS."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.data import Scope, collocations, ngrams, word_freq, pos_distribution, wordcloud_png, emojis, tfidf_terms
from app.theme import explain, style


def render(scope: Scope) -> None:
    df = scope.df
    n = scope.n
    strategy = scope.strategy

    st.subheader("Word cloud & most frequent words")
    explain("The bigger the word, the more often people use it. Stop-words "
            "('the', 'and'…) are removed first so only meaningful words remain.")
    col5, col6 = st.columns([2, 1])
    with col5:
        st.image(wordcloud_png(n, strategy), width="stretch")
    with col6:
        st.dataframe(word_freq(n, strategy, 20), width="stretch", hide_index=True)

    st.subheader("Top emojis")
    explain("Emojis carry a lot of sentiment on YouTube and tell you as "
            "much as words do.")
    em = emojis(n, strategy)
    if em.empty:
        st.caption("No emojis found in this sample.")
    else:
        st.plotly_chart(style(px.bar(em, x="emoji", y="count",
                                     color_discrete_sequence=["#fbbf24"])),
                        width="stretch")

    st.subheader("Phrases people repeat")
    explain("Left/middle: word pairs and triples that appear together far more "
            "often than chance (PMI collocations) — these are set phrases like "
            "game titles. Right: simply the most repeated two-word sequences.")
    col7, col8, col9 = st.columns(3)
    with col7:
        st.caption("Strongest word pairs (PMI bigrams)")
        st.dataframe(collocations(n, strategy, 2, 15), width="stretch", hide_index=True)
    with col8:
        st.caption("Strongest word triples (PMI trigrams)")
        st.dataframe(collocations(n, strategy, 3, 15), width="stretch", hide_index=True)
    with col9:
        st.caption("Most frequent bigrams")
        st.dataframe(ngrams(n, strategy, 2, 15), width="stretch", hide_index=True)

    if "sentiment" in df:
        st.subheader("What fans vs critics say differently")
        explain("TF-IDF finds the words most characteristic of each sentiment "
                "group — the vocabulary that separates praise from complaints.")
        terms = tfidf_terms(n, strategy)
        cols = st.columns(max(len(terms), 1))
        for col, (lab, ws) in zip(cols, terms.items()):
            with col:
                st.caption(lab.capitalize())
                st.write("  \n".join(f"• {w}" for w in ws))

    st.subheader("Grammar profile (part-of-speech)")
    explain("Breaks all words into grammar categories. Many adjectives = a "
            "descriptive, opinionated crowd; many verbs = people describing "
            "actions and gameplay.")
    st.plotly_chart(style(px.bar(pos_distribution(n, strategy), x="POS", y="count",
                                 color_discrete_sequence=["#818cf8"])),
                    width="stretch")
