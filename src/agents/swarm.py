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
    query: str
    draft: str
    feedback: str
    approved: bool
    turns: int


@lru_cache(maxsize=1)
def _llm() -> ChatOllama:
    return ChatOllama(model=settings.llm_model, temperature=0)


def researcher(state: SwarmState) -> dict:
    """Draft (or revise) a grounded answer, then hand off to the reviewer."""
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
    """Critique the draft; hand control back to the researcher if it needs work."""
    verdict = (REVIEW_PROMPT | _llm() | StrOutputParser()).invoke(
        {"q": state["query"], "a": state["draft"]}
    )
    if verdict.strip().upper().startswith("APPROVE") or state.get("turns", 0) >= MAX_TURNS:
        return {"approved": True, "feedback": ""}
    return {"approved": False, "feedback": verdict}


def handoff(state: SwarmState) -> str:
    return END if state.get("approved") else "researcher"


@lru_cache(maxsize=1)
def build_swarm():
    builder = StateGraph(SwarmState)
    builder.add_node("researcher", researcher)
    builder.add_node("reviewer", reviewer)
    builder.set_entry_point("researcher")
    builder.add_edge("researcher", "reviewer")
    builder.add_conditional_edges("reviewer", handoff, {"researcher": "researcher", END: END})
    return builder.compile()


def ask_swarm(query: str) -> str:
    result = build_swarm().invoke({"query": query})
    return result["draft"]


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    mlflow.langchain.autolog()
    with mlflow.start_run(run_name="swarm_session"):
        print(ask_swarm("What do people think about the graphics?"))


if __name__ == "__main__":
    main()
