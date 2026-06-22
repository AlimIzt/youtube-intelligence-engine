"""Streamlit dashboard: insights + chat with the agent.

Surfaces every analytical method from the labs (NER, keywords, collocations,
n-grams, POS, relations/knowledge-graph, sentiment, topics) plus the RAG agent.

Run:  streamlit run app/dashboard.py   ->  http://localhost:5000
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from config import settings

st.set_page_config(page_title="YouTube Intelligence Engine", layout="wide")
st.title("YouTube Intelligence Engine")


# ----------------------------------------------------------------------------
# data + cached analytics (sampled for the heavier NLP so the UI stays snappy)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    if settings.enriched_parquet.exists():
        return pd.read_parquet(settings.enriched_parquet)
    if settings.clean_csv.exists():
        return pd.read_csv(settings.clean_csv)
    return pd.DataFrame()


@st.cache_data
def list_column_counts(col: str, top: int) -> pd.DataFrame:
    """Aggregate a list-valued column (entities/keywords) into top counts."""
    df = load_data()
    counter: Counter = Counter()
    if col not in df:
        return pd.DataFrame(columns=[col, "count"])
    for row in df[col]:
        if row is None:
            continue
        counter.update([str(x) for x in row])  # row may be a numpy array
    return pd.DataFrame(counter.most_common(top), columns=[col, "count"])


@st.cache_data
def sample_texts(n: int) -> list[str]:
    df = load_data()
    return df["text"].dropna().astype(str).head(n).tolist()


@st.cache_data
def collocations(n: int, gram: int, top: int) -> pd.DataFrame:
    from src.analysis.collocations import pmi_collocations

    return pmi_collocations(sample_texts(n), n=gram, top=top)


@st.cache_data
def ngrams(n: int, gram: int, top: int) -> pd.DataFrame:
    from src.analysis.collocations import top_ngrams

    return top_ngrams(sample_texts(n), n=gram, top=top)


@st.cache_data
def word_freq(n: int, top: int) -> pd.DataFrame:
    from src.analysis.collocations import word_frequencies

    return word_frequencies(sample_texts(n), top)


@st.cache_resource
def _spacy():
    import spacy

    return spacy.load(settings.spacy_model, disable=["ner", "lemmatizer"])


@st.cache_data
def pos_distribution(n: int) -> pd.DataFrame:
    nlp = _spacy()
    counter: Counter = Counter()
    for doc in nlp.pipe(sample_texts(n), batch_size=128):
        counter.update(t.pos_ for t in doc if not t.is_space)
    return pd.DataFrame(counter.most_common(), columns=["POS", "count"])


@st.cache_data
def relations(n: int) -> list[str]:
    from src.analysis.relations import relations_for_corpus

    return relations_for_corpus(sample_texts(n), limit=n)


@st.cache_data
def kg_dot(n: int, max_edges: int) -> str:
    from src.analysis.relations import knowledge_graph_dot

    return knowledge_graph_dot(sample_texts(n), max_edges=max_edges)


@st.cache_data
def wordcloud_png(n: int) -> bytes:
    import io

    from wordcloud import WordCloud

    from src.preprocessing.nltk_clean import clean_join

    text = " ".join(clean_join(t) for t in sample_texts(n)) or "none"
    wc = WordCloud(width=900, height=400, background_color="white",
                   colormap="viridis", max_words=120).generate(text)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return buf.getvalue()


df = load_data()
if df.empty:
    st.warning("No data yet. Run the pipeline scripts (02–04) first.")
    st.stop()

tab_overview, tab_insights, tab_chat, tab_rag = st.tabs(
    ["Overview", "Insights", "Ask the Agent", "RAG Lab (W8L7)"]
)

# ----------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Comments", f"{len(df):,}")
    if "sentiment" in df:
        c2.metric("Positive", f"{(df.sentiment == 'positive').mean() * 100:.0f}%")
    if "likes" in df:
        c3.metric("Total likes", f"{int(df.likes.sum()):,}")
    st.dataframe(df.head(50), width="stretch")

# ----------------------------------------------------------------------------
with tab_insights:
    st.caption(
        "Heavier NLP (collocations, POS, relations) runs on a sample for speed."
    )
    n = st.slider("Sample size for text analytics", 200, 5000, 800, step=200)

    # --- sentiment + topics (precomputed columns) ---
    col1, col2 = st.columns(2)
    if "sentiment" in df:
        with col1:
            st.subheader("Sentiment distribution")
            counts = df.sentiment.value_counts().reset_index()
            counts.columns = ["sentiment", "count"]
            st.plotly_chart(
                px.bar(counts, x="sentiment", y="count", color="sentiment",
                       color_discrete_map={"positive": "green", "neutral": "gray",
                                           "negative": "red"}),
                width="stretch",
            )
    if "topic" in df:
        with col2:
            st.subheader("Top topics by volume")
            tc = df[df.topic != -1].topic.value_counts().head(10).reset_index()
            tc.columns = ["topic", "count"]
            st.plotly_chart(px.bar(tc, x="topic", y="count"),
                            width="stretch")

    # --- NER + keywords (precomputed list columns) ---
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Top named entities (NER)")
        ents = list_column_counts("entities", 20)
        if not ents.empty:
            st.plotly_chart(px.bar(ents, x="count", y="entities", orientation="h"),
                            width="stretch")
    with col4:
        st.subheader("Top keywords")
        kws = list_column_counts("keywords", 20)
        if not kws.empty:
            st.plotly_chart(px.bar(kws, x="count", y="keywords", orientation="h"),
                            width="stretch")

    # --- word cloud + word frequency ---
    st.subheader("Word cloud & frequency (Lab 1/5)del")
    col5, col6 = st.columns([2, 1])
    with col5:
        st.image(wordcloud_png(n), width="stretch")
    with col6:
        st.dataframe(word_freq(n, 20), width="stretch", hide_index=True)

    # --- collocations + n-grams ---
    st.subheader("Collocations & n-grams (Lab 1)del")
    col7, col8, col9 = st.columns(3)
    with col7:
        st.caption("PMI bigrams")
        st.dataframe(collocations(n, 2, 15), width="stretch", hide_index=True)
    with col8:
        st.caption("PMI trigrams")
        st.dataframe(collocations(n, 3, 15), width="stretch", hide_index=True)
    with col9:
        st.caption("Top bigrams (frequency)")
        st.dataframe(ngrams(n, 2, 15), width="stretch", hide_index=True)

    # --- POS distribution ---
    st.subheader("Part-of-speech distribution (Lab 2)del")
    st.plotly_chart(px.bar(pos_distribution(n), x="POS", y="count"),
                    width="stretch")

    # --- relations + knowledge graph ---
    st.subheader("Noun-Verb-Noun relations & knowledge graph (Lab 2)del")
    col10, col11 = st.columns([1, 2])
    with col10:
        rels = relations(n)
        st.caption(f"{len(rels)} unique relations (showing 25)")
        st.dataframe(pd.DataFrame(rels[:25], columns=["verb(subject, object)"]),
                     width="stretch", hide_index=True)
    with col11:
        st.caption("Knowledge graph (top relations)")
        st.graphviz_chart(kg_dot(n, 25))

# ----------------------------------------------------------------------------
with tab_chat:
    st.subheader("Ask about the comments")
    st.caption("RAG grounded in the comments (needs Ollama running).")

    c1, c2 = st.columns(2)
    mode = c1.radio(
        "Agent",
        [
            "Tool-calling (single)",
            "Multi-agent (router)",
            "Supervisor (full report)",
            "Swarm (reflection)",
            "DSPy (Chain-of-Thought)",
        ],
    )
    retrieval = c2.radio("Retrieval", ["semantic", "mmr"], horizontal=True)
    rerank = st.checkbox("Cross-encoder rerank (W8L7)", value=False)

    q = st.text_input("Your question", "What do people think about the graphics?")
    if st.button("Ask"):
        with st.spinner("Thinking ..."):
            try:
                if mode.startswith("Multi"):
                    from src.agents.multi_agent import ask_multi

                    cat, resp = ask_multi(q)
                    st.info(f"Routed to: **{cat}**")
                    st.write(resp)
                elif mode.startswith("Supervisor"):
                    from src.agents.supervisor import generate_report

                    st.write(generate_report(q))
                elif mode.startswith("Swarm"):
                    from src.agents.swarm import ask_swarm

                    st.write(ask_swarm(q))
                elif mode.startswith("DSPy"):
                    from src.rag.dspy_qa import answer_question_dspy

                    r = answer_question_dspy(q)
                    st.write(r.answer)
                    st.caption(f"Self-rated confidence: {r.confidence}")
                    with st.expander("Reasoning (Chain-of-Thought)"):
                        st.write(getattr(r, "reasoning", ""))
                else:
                    from src.rag.generation import answer_question
                    from src.rag.retrieval import mmr_retriever, semantic_retriever

                    retr = mmr_retriever() if retrieval == "mmr" else semantic_retriever()
                    if rerank:
                        from src.rag.postretrieval import rerank as rr
                        docs = rr(q, retr.invoke(q), top_k=settings.top_k)
                        from src.rag.generation import QA_PROMPT, get_llm
                        from src.rag.retrieval import format_docs
                        from langchain_core.output_parsers import StrOutputParser

                        chain = QA_PROMPT | get_llm() | StrOutputParser()
                        st.write(chain.invoke({"question": q,
                                               "context": format_docs(docs)}))
                    else:
                        st.write(answer_question(q, retriever=retr))
            except Exception as e:
                st.error(f"Agent error: {e}")

# ----------------------------------------------------------------------------
with tab_rag:
    st.subheader("RAG components (W8L7)")
    st.caption("Chunking strategies, retrieval comparison, and reranking.")

    # --- 1. Chunking strategies ---
    st.markdown("### 1 · Chunking strategies")
    join_n = st.slider("Comments to join into one document", 5, 100, 20, step=5)
    strategy = st.radio(
        "Strategy", ["recursive", "token", "semantic"], horizontal=True
    )
    if st.button("Chunk"):
        with st.spinner("Chunking ..."):
            try:
                from src.rag import chunking

                doc = "\n".join(sample_texts(join_n))
                fn = {
                    "recursive": chunking.recursive_chunks,
                    "token": chunking.token_chunks,
                    "semantic": chunking.semantic_chunks,
                }[strategy]
                chunks = fn(doc)
                st.success(f"{strategy} → {len(chunks)} chunks")
                for i, c in enumerate(chunks[:8]):
                    st.text(f"[chunk {i}] {c[:200]}")
            except Exception as e:
                st.error(f"Chunking error: {e}")

    st.divider()

    # --- 2. Retrieval strategy comparison ---
    st.markdown("### 2 · Retrieval strategy comparison")
    rq = st.text_input("Query", "graphics look amazing", key="rag_query")
    if st.button("Compare retrieval"):
        with st.spinner("Retrieving ..."):
            try:
                from src.rag.retrieval import (
                    hybrid_retriever,
                    lexical_retriever,
                    mmr_retriever,
                    semantic_retriever,
                )

                strategies = {
                    "Semantic": semantic_retriever(k=4),
                    "Lexical (BM25)": lexical_retriever(df, k=4),
                    "Hybrid": hybrid_retriever(df, k=4),
                    "MMR": mmr_retriever(k=4),
                }
                cols = st.columns(len(strategies))
                for col, (name, retr) in zip(cols, strategies.items()):
                    with col:
                        st.caption(name)
                        for d in retr.invoke(rq):
                            st.text(f"• {d.page_content[:80]}")
            except Exception as e:
                st.error(f"Retrieval error: {e}")

    st.divider()

    # --- 3. Cross-encoder reranking ---
    st.markdown("### 3 · Cross-encoder reranking")
    rrq = st.text_input("Query", "is the game realistic", key="rerank_query")
    if st.button("Rerank"):
        with st.spinner("Reranking ..."):
            try:
                from src.rag.postretrieval import rerank
                from src.rag.retrieval import semantic_retriever

                docs = semantic_retriever(k=10).invoke(rrq)
                reranked = rerank(rrq, docs, top_k=5)
                c_before, c_after = st.columns(2)
                with c_before:
                    st.caption("Before (semantic top 5)")
                    for d in docs[:5]:
                        st.text(f"• {d.page_content[:80]}")
                with c_after:
                    st.caption("After (cross-encoder top 5)")
                    for d in reranked:
                        st.text(f"• {d.page_content[:80]}")
            except Exception as e:
                st.error(f"Rerank error: {e}")
