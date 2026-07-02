"""RAG internals: chunking, retrieval comparison, reranking, HyDE."""
from __future__ import annotations

import streamlit as st

from app.data import Scope
from app.theme import explain


def render(scope: Scope) -> None:
    df = scope.df
    texts = scope.texts

    st.subheader("RAG components")
    explain("A peek under the hood of the 'Ask the Agent' tab: how documents are "
            "split into chunks, how different retrieval strategies compare, and "
            "how reranking improves results.")

    st.markdown("### 1 · Chunking strategies")
    join_n = st.slider("Comments to join into one document", 5, 100, 20, step=5)
    chunk_strategy = st.radio(
        "Strategy", ["recursive", "token", "semantic"], horizontal=True,
        help="recursive = split on paragraphs/sentences; token = fixed token "
             "windows; semantic = split where the meaning shifts.")
    if st.button("Chunk"):
        with st.spinner("Chunking ..."):
            try:
                from src.rag import chunking

                doc = "\n".join(texts[:join_n])
                fn = {"recursive": chunking.recursive_chunks,
                      "token": chunking.token_chunks,
                      "semantic": chunking.semantic_chunks}[chunk_strategy]
                chunks = fn(doc)
                st.success(f"{chunk_strategy} → {len(chunks)} chunks")
                for i, c in enumerate(chunks[:8]):
                    st.text(f"[chunk {i}] {c[:200]}")
            except Exception as e:
                st.error(f"Chunking error: {e}")

    st.divider()

    st.markdown("### 2 · Retrieval strategy comparison")
    rq = st.text_input("Query", "graphics look amazing", key="rag_query")
    if st.button("Compare retrieval"):
        with st.spinner("Retrieving ..."):
            try:
                from src.rag.retrieval import (hybrid_retriever, lexical_retriever,
                                               mmr_retriever, semantic_retriever)

                strategies = {"Semantic": semantic_retriever(k=4),
                              "Lexical (BM25)": lexical_retriever(df, k=4),
                              "Hybrid": hybrid_retriever(df, k=4),
                              "MMR": mmr_retriever(k=4)}
                cols = st.columns(len(strategies))
                for col, (name, retr) in zip(cols, strategies.items()):
                    with col:
                        st.caption(name)
                        for d in retr.invoke(rq):
                            st.text(f"• {d.page_content[:80]}")
            except Exception as e:
                st.error(f"Retrieval error: {e}")

    st.divider()

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
    st.divider()

    st.markdown("### 4 · HyDE (hypothetical document embeddings)")
    explain("Short questions embed poorly. HyDE first asks the LLM to invent a "
            "plausible comment that WOULD answer the question, then searches "
            "with that richer fake comment instead — it usually lands closer "
            "to real matches. Needs Ollama running.")
    hq = st.text_input("Question", "Do people complain about the release date?",
                       key="hyde_query")
    n_hypo = st.slider("Hypothetical comments to generate", 1, 3, 1,
                       help="With more than one, the pooled results are "
                            "reranked against the original question with the "
                            "cross-encoder.")
    if st.button("Run HyDE"):
        with st.spinner("Generating hypothetical comment and retrieving ..."):
            try:
                from src.rag.hyde import hyde_retrieve, hypothetical_answers
                from src.rag.retrieval import semantic_retriever

                hypos = hypothetical_answers(hq, n_hypo)
                for h in hypos:
                    st.info(h)
                c_plain, c_hyde = st.columns(2)
                with c_plain:
                    st.caption("Plain semantic search (raw question)")
                    for d in semantic_retriever(k=5).invoke(hq):
                        st.text(f"• {d.page_content[:80]}")
                with c_hyde:
                    st.caption("HyDE (searched with the hypothetical comment)")
                    for d in hyde_retrieve(hq, k=5, hypotheticals=hypos):
                        st.text(f"• {d.page_content[:80]}")
            except Exception as e:
                st.error(f"HyDE error: {e}")
