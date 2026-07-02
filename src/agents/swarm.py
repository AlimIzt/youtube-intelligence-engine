"""Lab 8 Swarm pattern (homework).

Swarm = peer agents that hand control to each other until the task is done.
Here a `researcher` drafts a grounded answer from the comments and a `reviewer`
critiques it, handing control back for a revision if needed (a reflection loop),
otherwise finalizing. Implemented as an explicit handoff graph so it stays
reliable on small models.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

import mlflow
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from config import settings
from src.rag.generation import answer_question
from src.rag.retrieval import format_docs, semantic_retriever

MAX_TURNS = 2


class SwarmState(TypedDict, total=False):
    """Shared state passed between swarm agents.

    Attributes:
        query: User question about the YouTube comments.
        draft: Current answer drafted by the researcher.
        feedback: Reviewer feedback used for revision.
        approved: Whether the reviewer accepted the draft.
        turns: Number of researcher attempts already made.
    """

    query: str
    draft: str
    feedback: str
    approved: bool
    turns: int


@lru_cache(maxsize=1)
def _llm() -> ChatOllama:
    """Return a cached local Ollama chat model for swarm agents."""
    return ChatOllama(model=settings.llm_model, temperature=0)


def researcher(state: SwarmState) -> dict:
    """Draft or revise a grounded answer, then hand off to the reviewer.

    If reviewer feedback exists, the researcher retrieves relevant comments and
    revises the previous answer according to that feedback. Otherwise, it uses
    the standard RAG question-answering chain to create the first draft.
    """
    query = state["query"]

    if state.get("feedback"):
        docs = semantic_retriever().invoke(query)
        prompt = ChatPromptTemplate.from_template(
            "Revise the answer to address the reviewer's feedback. Use only the "
            "comments.\n\nQuestion: {q}\nFeedback: {fb}\nComments:\n{ctx}\n\nRevised answer:"
        )
        chain = prompt | _llm() | StrOutputParser()
        draft = chain.invoke({"q": query, "fb": state["feedback"], "ctx": format_docs(docs)})
    else:
        draft = answer_question(query)

    return {"draft": draft, "turns": state.get("turns", 0) + 1}


REVIEW_PROMPT = ChatPromptTemplate.from_template(
    """You are a reviewer. Does this answer address the question using the comments,
with at least one citation like [1]? Reply 'APPROVE' if good, otherwise reply
'REVISE: <one specific improvement>'.

Question: {q}
Answer: {a}"""
)


def reviewer(state: SwarmState) -> dict:
    """Review the draft and decide whether revision is needed.

    The reviewer approves answers that address the question using retrieved
    comment evidence. If the answer is not strong enough, the reviewer returns
    specific feedback and hands control back to the researcher. The loop is
    capped by MAX_TURNS to avoid infinite revision.
    """
    verdict = (REVIEW_PROMPT | _llm() | StrOutputParser()).invoke(
        {"q": state["query"], "a": state["draft"]}
    )

    if verdict.strip().upper().startswith("APPROVE") or state.get("turns", 0) >= MAX_TURNS:
        return {"approved": True, "feedback": ""}

    return {"approved": False, "feedback": verdict}


def handoff(state: SwarmState) -> str:
    """Route control after reviewer feedback.

    Returns END if the draft is approved. Otherwise, routes back to the
    researcher for another revision.
    """
    return END if state.get("approved") else "researcher"


@lru_cache(maxsize=1)
def build_swarm():
    """Build and cache the Lab 8 swarm handoff graph.

    The graph starts with the researcher, sends the draft to the reviewer, and
    either ends or loops back to the researcher depending on reviewer approval.
    """
    builder = StateGraph(SwarmState)
    builder.add_node("researcher", researcher)
    builder.add_node("reviewer", reviewer)

    builder.set_entry_point("researcher")
    builder.add_edge("researcher", "reviewer")
    builder.add_conditional_edges("reviewer", handoff, {"researcher": "researcher", END: END})

    return builder.compile()


def ask_swarm(query: str) -> str:
    """Answer a user question using the swarm reflection workflow."""
    result = build_swarm().invoke({"query": query})
    return result["draft"]


def main() -> None:
    """Run a demo swarm session with MLflow logging enabled."""
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    mlflow.langchain.autolog()

    with mlflow.start_run(run_name="swarm_session"):
        print(ask_swarm("What do people think about the graphics?"))


if __name__ == "__main__":
    main()