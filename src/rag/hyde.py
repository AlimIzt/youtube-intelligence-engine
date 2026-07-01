"""HyDE: Hypothetical Document Embeddings (W8L7 5_Retrieval.py, adapted).

Short questions embed poorly. HyDE first asks the LLM to write a plausible
hypothetical comment answering the question, then uses THAT richer text as the
retrieval query — it usually lands closer to real matching comments than the
bare question does. Reuses the project's DSPy/Ollama setup, semantic retriever,
cross-encoder reranker, and grounded QA chain.
"""
from __future__ import annotations

import dspy
from langchain_core.documents import Document

from config import settings
from src.rag.dspy_qa import _configure


class GetHyDE(dspy.Signature):
    """Write a short, realistic YouTube comment that would answer the question.
    Sound like a real commenter, not an analyst. 1-3 sentences."""

    question = dspy.InputField(desc="a question about YouTube comments on a game trailer")
    hypothetical_answer = dspy.OutputField(desc="a plausible comment answering the question")


def hypothetical_answers(question: str, n: int = 1) -> list[str]:
    """Generate n hypothetical comments (temperature varied for diversity)."""
    _configure()
    outs: list[str] = []
    for i in range(max(n, 1)):
        pred = dspy.Predict(GetHyDE, temperature=min(0.2 + 0.4 * i, 1.0))
        outs.append(str(pred(question=question).hypothetical_answer).strip())
    return outs


def hyde_retrieve(
    question: str,
    k: int | None = None,
    n_hypothetical: int = 1,
    hypotheticals: list[str] | None = None,
) -> list[Document]:
    """Retrieve real comments using hypothetical answer(s) as the query.

    With multiple hypotheticals the pooled results are deduplicated and
    reranked against the ORIGINAL question with the existing cross-encoder.
    """
    from src.rag.retrieval import semantic_retriever

    k = k or settings.top_k
    hypos = hypotheticals or hypothetical_answers(question, n_hypothetical)
    retriever = semantic_retriever(k=k)
    seen: set[str] = set()
    pool: list[Document] = []
    for h in hypos:
        for d in retriever.invoke(h):
            if d.page_content not in seen:
                seen.add(d.page_content)
                pool.append(d)
    if len(hypos) > 1:
        from src.rag.postretrieval import rerank

        return rerank(question, pool, top_k=k)
    return pool[:k]


def hyde_answer(question: str, k: int | None = None,
                hypotheticals: list[str] | None = None) -> str:
    """Grounded final answer over HyDE-retrieved comments (existing QA chain)."""
    from langchain_core.output_parsers import StrOutputParser

    from src.rag.generation import QA_PROMPT, get_llm
    from src.rag.retrieval import format_docs

    docs = hyde_retrieve(question, k, hypotheticals=hypotheticals)
    chain = QA_PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({"question": question, "context": format_docs(docs)})


def main() -> None:
    q = "Do people complain about the release date?"
    hypos = hypothetical_answers(q, 2)
    for h in hypos:
        print(f"hypothetical: {h}")
    for d in hyde_retrieve(q, hypotheticals=hypos):
        print(f"retrieved: {d.page_content[:90]}")
    print(f"\nanswer:\n{hyde_answer(q, hypotheticals=hypos)}")


if __name__ == "__main__":
    main()
