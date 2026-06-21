"""W8L7 chunking strategies.

Comments are short so the main pipeline indexes one comment per document, but
these strategies are ported from the W8L7 RAG lab for completeness and for
chunking longer text (e.g. concatenated threads) in the report's representation
section: sliding/recursive, token-based, and semantic chunking.
"""
from __future__ import annotations

from config import settings


def recursive_chunks(text: str, chunk_size: int = 250, overlap: int = 25) -> list[str]:
    """Sliding-window chunks (RecursiveCharacterTextSplitter)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def token_chunks(text: str, chunk_size: int = 100, overlap: int = 10) -> list[str]:
    """Token-based chunks using the gpt-2 tokenizer (needs tiktoken)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt2", chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_text(text)


def semantic_chunks(text: str, threshold: int = 95) -> list[str]:
    """Semantic chunks: split where embedding similarity drops (SemanticChunker).

    Needs `langchain-experimental`. Uses the same Ollama embedding model as the
    vector store.
    """
    from langchain_experimental.text_splitter import SemanticChunker

    from src.rag.representation import get_embeddings

    splitter = SemanticChunker(
        get_embeddings(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold,
    )
    return splitter.split_text(text)
