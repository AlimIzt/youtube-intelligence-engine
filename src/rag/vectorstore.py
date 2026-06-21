"""Persisted Chroma vector store."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings
from src.rag.representation import get_embeddings


def build_vectorstore(docs: list[Document], reset: bool = True) -> Chroma:
    """Build (or rebuild) the Chroma collection from documents."""
    if reset and settings.chroma_dir.exists():
        import shutil

        shutil.rmtree(settings.chroma_dir, ignore_errors=True)

    store = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
    )
    print(f"Indexed {len(docs):,} documents → {settings.chroma_dir}")
    return store


def load_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )
