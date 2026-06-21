"""Persisted Chroma vector store."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings
from src.rag.representation import get_embeddings


def build_vectorstore(
    docs: list[Document], reset: bool = True, batch_size: int = 128
) -> Chroma:
    """Build (or rebuild) the Chroma collection from documents.

    Documents are embedded in small batches: sending thousands of texts in a
    single Ollama embed call crashes the model runner, so we chunk the work.
    """
    if reset and settings.chroma_dir.exists():
        import shutil

        shutil.rmtree(settings.chroma_dir, ignore_errors=True)

    store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )

    total = len(docs)
    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        store.add_documents(batch)
        done = min(i + batch_size, total)
        print(f"  indexed {done:,} / {total:,}", flush=True)

    print(f"Indexed {total:,} documents → {settings.chroma_dir}")
    return store


def load_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )
