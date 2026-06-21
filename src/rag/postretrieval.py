"""W8L7 post-retrieval: cross-encoder reranking.

Research shows LLMs are sensitive to the order of retrieved context. A
cross-encoder rescores (question, document) pairs more accurately than the
bi-encoder used for first-stage retrieval, so we rerank before generation.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _reranker(model_name: str = RERANK_MODEL):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank(question: str, docs: list[Document], top_k: int = 5) -> list[Document]:
    """Reorder retrieved documents by cross-encoder relevance (W8L7)."""
    if not docs:
        return docs
    pairs = [[question, d.page_content] for d in docs]
    scores = _reranker().predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]
