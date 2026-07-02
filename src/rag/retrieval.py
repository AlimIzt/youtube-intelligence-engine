"""Retrieval strategies: semantic, lexical, hybrid, and metadata-filtered.

Implements the four options listed in the deliverables so they can be compared
in evaluation.
"""
from __future__ import annotations

import pandas as pd
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever

from config import settings
from src.rag.representation import df_to_documents
from src.rag.vectorstore import load_vectorstore


def semantic_retriever(k: int | None = None):
    """Dense vector similarity over Chroma."""
    k = k or settings.top_k
    return load_vectorstore().as_retriever(search_kwargs={"k": k})


def mmr_retriever(k: int | None = None, fetch_k: int = 20, lambda_mult: float = 0.5):
    """Maximal Marginal Relevance: balances relevance with diversity (W8L7).

    Avoids returning many near-duplicate comments (which are common on YouTube).
    """
    k = k or settings.top_k
    return load_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


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


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i+1}] ({d.metadata.get('sentiment','?')}, "
        f"likes={d.metadata.get('likes',0)}) {d.page_content}"
        for i, d in enumerate(docs)
    )
