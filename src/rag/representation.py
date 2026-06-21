"""Turn enriched comments into LangChain Documents + embeddings.

Representation = text (the comment) + metadata (sentiment, topic, likes,
entities, keywords) + vector (Ollama embedding). Metadata enables metadata and
hybrid retrieval downstream.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    """Local Ollama embedding model (matches Lab 6/7 Ollama usage)."""
    return OllamaEmbeddings(model=settings.embed_model)


def _join(values) -> str:
    if isinstance(values, (list, tuple)):
        return ", ".join(map(str, values))
    return str(values) if pd.notna(values) else ""


def df_to_documents(df: pd.DataFrame, text_col: str = "text") -> list[Document]:
    """One Document per comment. Comments are short, so no chunking needed."""
    docs: list[Document] = []
    for _, r in df.iterrows():
        meta = {
            "comment_id": r.get("comment_id", ""),
            "video_id": r.get("video_id", ""),
            "author": r.get("author", ""),
            "likes": int(r.get("likes", 0) or 0),
            "published_at": str(r.get("published_at", "")),
            "sentiment": r.get("sentiment", "unknown"),
            "topic": int(r.get("topic", -1)) if pd.notna(r.get("topic", -1)) else -1,
            "entities": _join(r.get("entities", "")),
            "keywords": _join(r.get("keywords", "")),
        }
        docs.append(Document(page_content=str(r[text_col]), metadata=meta))
    return docs
