"""Lab 6/7 DSPy: declarative LLM programming for grounded QA.

Instead of hand-writing prompts, DSPy uses a typed Signature (inputs -> outputs)
and a Module (here ChainOfThought, which adds a reasoning step). This mirrors the
Lab 6 `QA_CoT` example: the module answers grounded in retrieved comments and
rates its own confidence. Uses the same local Ollama model via dspy.LM.
"""
from __future__ import annotations

from functools import lru_cache

import dspy

from config import settings
from src.rag.retrieval import format_docs, semantic_retriever


@lru_cache(maxsize=1)
def _configure():
    lm = dspy.LM(
        f"ollama/{settings.llm_model}",
        api_base="http://localhost:11434",
        api_key="ollama",
        temperature=0,
    )
    dspy.configure(lm=lm)
    return lm


class GroundedQA(dspy.Signature):
    """Answer the question using ONLY the provided YouTube comments.
    Cite comment numbers like [1]. If the comments don't answer it, say so."""

    context = dspy.InputField(desc="retrieved YouTube comments")
    question = dspy.InputField(desc="a question about the comments")
    answer = dspy.OutputField(desc="answer grounded in the comments, with citations")
    confidence = dspy.OutputField(desc="self-rated confidence from 1 (low) to 5 (high)")


class CommentCoT(dspy.Module):
    """Chain-of-Thought QA module (adds an automatic reasoning field)."""

    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(GroundedQA)

    def forward(self, context: str, question: str):
        return self.prog(context=context, question=question)


@lru_cache(maxsize=1)
def _module() -> CommentCoT:
    _configure()
    return CommentCoT()


def answer_question_dspy(question: str, k: int | None = None):
    """Run the DSPy CoT module over retrieved comments.

    Returns a dspy.Prediction with .reasoning, .answer, .confidence.
    """
    docs = semantic_retriever(k=k or settings.top_k).invoke(question)
    return _module()(context=format_docs(docs), question=question)


def main() -> None:
    r = answer_question_dspy("What do people think about the graphics?")
    print(f"reasoning:\n{getattr(r, 'reasoning', '')}\n")
    print(f"answer:\n{r.answer}\n")
    print(f"confidence: {r.confidence}")


if __name__ == "__main__":
    main()
