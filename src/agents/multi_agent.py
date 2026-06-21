"""Lab 8 multi-agent orchestration: classifier -> router -> specialist.

An alternative to the tool-calling agent in orchestrator.py. A classifier node
labels the query, a router sends it to a specialist node, and each specialist
produces the answer. Deterministic routing is more reliable on small models
(llama3.2:3b) than tool-calling, and mirrors the Lab 8 customer-service graph.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

import mlflow
import pandas as pd
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from config import settings
from src.rag.generation import answer_question, summarize


class QueryState(TypedDict):
    query: str
    category: str
    response: str


@lru_cache(maxsize=1)
def _llm() -> ChatOllama:
    return ChatOllama(model=settings.llm_model, temperature=0)


def classify(state: QueryState) -> dict:
    """Classify the query into one of four categories (Lab 8 classifier agent)."""
    prompt = f"""Classify this question about YouTube comments into ONE category:
- qa         : asks what people said about something specific
- summary    : asks for an overall summary of opinions
- sentiment  : asks whether people are positive/negative
- topics     : asks what topics/themes are discussed

Question: {state['query']}
Return only the category name."""
    category = _llm().invoke(prompt).content.strip().lower()
    for cat in ("sentiment", "summary", "topics", "qa"):
        if cat in category:
            return {"category": cat}
    return {"category": "qa"}


def route(state: QueryState) -> str:
    return {
        "qa": "qa_agent",
        "summary": "summary_agent",
        "sentiment": "sentiment_agent",
        "topics": "topic_agent",
    }.get(state["category"], "qa_agent")


def qa_agent(state: QueryState) -> dict:
    return {"response": answer_question(state["query"])}


def summary_agent(state: QueryState) -> dict:
    return {"response": summarize(state["query"])}


def sentiment_agent(state: QueryState) -> dict:
    df = pd.read_parquet(settings.enriched_parquet)
    dist = df["sentiment"].value_counts(normalize=True).mul(100).round(1)
    lines = [f"Sentiment across {len(df):,} comments:"]
    lines += [f"  - {k}: {v}%" for k, v in dist.items()]
    return {"response": "\n".join(lines)}


def topic_agent(state: QueryState) -> dict:
    df = pd.read_parquet(settings.enriched_parquet)
    counts = df[df.topic != -1]["topic"].value_counts().head(8)
    body = "\n".join(f"  - topic {t}: {c} comments" for t, c in counts.items())
    return {"response": "Main discussion topics by volume:\n" + body}


@lru_cache(maxsize=1)
def build_multi_agent():
    builder = StateGraph(QueryState)
    builder.add_node("classifier", classify)
    builder.add_node("qa_agent", qa_agent)
    builder.add_node("summary_agent", summary_agent)
    builder.add_node("sentiment_agent", sentiment_agent)
    builder.add_node("topic_agent", topic_agent)

    builder.set_entry_point("classifier")
    builder.add_conditional_edges(
        "classifier",
        route,
        {
            "qa_agent": "qa_agent",
            "summary_agent": "summary_agent",
            "sentiment_agent": "sentiment_agent",
            "topic_agent": "topic_agent",
        },
    )
    for node in ("qa_agent", "summary_agent", "sentiment_agent", "topic_agent"):
        builder.add_edge(node, END)
    return builder.compile()


def ask_multi(query: str) -> tuple[str, str]:
    """Returns (category, response)."""
    graph = build_multi_agent()
    result = graph.invoke({"query": query})
    return result["category"], result["response"]


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    mlflow.langchain.autolog()
    print("\nMulti-agent (classifier -> router -> specialist). 'exit' to quit.\n")
    with mlflow.start_run(run_name="multi_agent_session"):
        while True:
            q = input("User: ")
            if q.lower() in {"exit", "quit"}:
                break
            cat, resp = ask_multi(q)
            print(f"\n[routed to: {cat}]\n{resp}\n")


if __name__ == "__main__":
    main()
