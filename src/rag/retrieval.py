"""Retrieval strategies: semantic, lexical, hybrid, and metadata-filtered.

Implements the four options listed in the deliverables so they can be compared
in evaluation.
"""
from __future__ import annotations

import pandas as pd
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever

from config import settings
from src.rag.representation import df_to_documents
from src.rag.vectorstore import load_vectorstore


def semantic_retriever(k: int | None = None):
    """Dense vector similarity over Chroma."""
    k = k or settings.top_k
    return load_vectorstore().as_retriever(search_kwargs={"k": k})


def lexical_retriever(df: pd.DataFrame, k: int | None = None) -> BM25Retriever:
    """Sparse BM25 keyword search (built in-memory from the corpus)."""
    k = k or settings.top_k
    retr = BM25Retriever.from_documents(df_to_documents(df))
    retr.k = k
    return retr


def hybrid_retriever(df: pd.DataFrame, k: int | None = None, weights=(0.5, 0.5)):
    """Ensemble of semantic + lexical (reciprocal-rank fusion)."""
    k = k or settings.top_k
    return EnsembleRetriever(
        retrievers=[semantic_retriever(k), lexical_retriever(df, k)],
        weights=list(weights),
    )


def metadata_retriever(filters: dict, k: int | None = None):
    """Semantic search constrained by metadata, e.g. {'sentiment': 'negative'}.

    Chroma expects a where-filter; for a single key we pass it directly, for
    multiple we wrap with $and.
    """
    k = k or settings.top_k
    if len(filters) == 1:
        where = filters
    else:
        where = {"$and": [{key: val} for key, val in filters.items()]}
    return load_vectorstore().as_retriever(
        search_kwargs={"k": k, "filter": where}
    )


def get_retriever(strategy: str, df: pd.DataFrame | None = None, **kwargs):
    """Factory used by the agent/generation layer."""
    if strategy == "semantic":
        return semantic_retriever(**kwargs)
    if strategy == "lexical":
        if df is None:
            raise ValueError("lexical retrieval needs the dataframe")
        return lexical_retriever(df, **kwargs)
    if strategy == "hybrid":
        if df is None:
            raise ValueError("hybrid retrieval needs the dataframe")
        return hybrid_retriever(df, **kwargs)
    if strategy == "metadata":
        return metadata_retriever(**kwargs)
    raise ValueError(f"unknown strategy: {strategy}")


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i+1}] ({d.metadata.get('sentiment','?')}, "
        f"likes={d.metadata.get('likes',0)}) {d.page_content}"
        for i, d in enumerate(docs)
    )
