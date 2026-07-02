"""spaCy NER, GLiNER zero-shot domain NER, keywords, relations, KG."""
from __future__ import annotations

import plotly.express as px
import pandas as pd
import streamlit as st

from app.data import Scope, list_column_counts, relations, kg_dot, gliner_entities
from app.theme import explain, takeaway, style


def render(scope: Scope) -> None:
    n = scope.n
    strategy = scope.strategy

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Names mentioned (NER)")
        explain("Named-entity recognition automatically spots names of people, "
                "companies, games and places inside the comments.")
        ents = list_column_counts(n, strategy, "entities", 20)
        if not ents.empty:
            st.plotly_chart(style(px.bar(ents, x="count", y="entities",
                                         orientation="h",
                                         color_discrete_sequence=["#a5b4fc"])),
                            width="stretch")
            takeaway(f"Most mentioned: **{ents.iloc[0].entities}**.")
    with col4:
        st.subheader("Keywords")
        explain("The most informative words/phrases per comment (YAKE), "
                "aggregated — a quick summary of what the comments are about.")
        kws = list_column_counts(n, strategy, "keywords", 20)
        if not kws.empty:
            st.plotly_chart(style(px.bar(kws, x="count", y="keywords",
                                         orientation="h",
                                         color_discrete_sequence=["#fda4af"])),
                            width="stretch")

    st.subheader("Domain entities (GLiNER zero-shot NER)")
    explain("Classic NER above only knows fixed categories like PERSON or ORG. "
            "GLiNER matches any label you type at inference time, so it can "
            "find things spaCy has no tag for — game features, platforms, "
            "prices. Edit the labels and re-run to ask for anything else.")
    from src.analysis.gliner_ner import DEFAULT_LABELS
    labels_raw = st.text_input("Labels (comma-separated)",
                               ", ".join(DEFAULT_LABELS))
    st.caption("First run downloads the GLiNER model (~500 MB); inference on "
               "large samples takes a while — lower the sidebar sample size "
               "if it's slow.")
    if st.button("Extract domain entities"):
        with st.spinner("Running GLiNER ..."):
            try:
                labels = tuple(x.strip() for x in labels_raw.split(",") if x.strip())
                gl = gliner_entities(n, strategy, labels)
                if gl.empty:
                    st.caption("No entities found for these labels in this sample.")
                else:
                    st.plotly_chart(style(px.bar(
                        gl.head(25), x="count", y="entity", color="label",
                        orientation="h",
                        color_discrete_sequence=px.colors.qualitative.Bold)),
                        width="stretch")
                    top = gl.iloc[0]
                    takeaway(f"Most mentioned: **{top.entity}** "
                             f"({top.label}, {top['count']} mentions) across "
                             f"{gl.label.nunique()} label types.")
                    with st.expander("Full entity table"):
                        st.dataframe(gl, width="stretch", hide_index=True)
            except Exception as e:
                st.error(f"GLiNER error: {e}")

    st.subheader("Who does what — relations & knowledge graph")
    explain("Extracts simple 'subject → verb → object' statements from comments "
            "(e.g. 'Rockstar delayed game') and draws them as a graph, so you "
            "can see claims, not just words.")
    col10, col11 = st.columns([1, 2])
    with col10:
        rels = relations(n, strategy)
        st.caption(f"{len(rels)} unique relations (showing 25)")
        st.dataframe(pd.DataFrame(rels[:25], columns=["verb(subject, object)"]),
                     width="stretch", hide_index=True)
    with col11:
        st.caption("Knowledge graph (top relations)")
        st.graphviz_chart(kg_dot(n, strategy, 25))
