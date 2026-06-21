"""QA and summarization chains grounded in retrieved comments (Ollama)."""
from __future__ import annotations

from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import settings
from src.rag.retrieval import format_docs, semantic_retriever


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    return ChatOllama(model=settings.llm_model, temperature=0)


QA_PROMPT = ChatPromptTemplate.from_template(
    """You are an analyst answering questions about YouTube comments on a video.
Use ONLY the comments in the context. If the context doesn't answer the
question, say so. Cite comment numbers like [1], [2].

Question: {question}

Context (retrieved comments):
{context}

Answer:"""
)

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """Summarize what these YouTube comments express about: {topic}
Give 3-5 bullet points covering the main opinions, praise, and complaints.
Base the summary ONLY on the comments below.

Comments:
{context}

Summary:"""
)


def answer_question(question: str, retriever=None) -> str:
    retriever = retriever or semantic_retriever()
    docs = retriever.invoke(question)
    chain = QA_PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({"question": question, "context": format_docs(docs)})


def summarize(topic: str, retriever=None, k: int = 12) -> str:
    retriever = retriever or semantic_retriever(k=k)
    docs = retriever.invoke(topic)
    chain = SUMMARY_PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({"topic": topic, "context": format_docs(docs)})
