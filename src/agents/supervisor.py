"""Lab 8 Supervisor pattern.

A supervisor decides which specialist agent runs next, sequencing them until a
complete "comment intelligence report" is assembled, then a final agent composes
the report. Mirrors the Lab 8 travel-planner supervisor (conditional entry point
+ each specialist routing back through the supervisor), adapted to YouTube
comment analysis.

Flow:  sentiment -> topics -> entities -> summary -> final
"""
from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

import mlflow
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from config import settings
from src.rag.generation import summarize


class ReportState(TypedDict, total=False):
    request: str
    sentiment_report: str
    topic_report: str
    entity_report: str
    summary_report: str
    final_report: str


@lru_cache(maxsize=1)
def _llm() -> ChatOllama:
    return ChatOllama(model=settings.llm_model, temperature=0)


@lru_cache(maxsize=1)
def _data() -> pd.DataFrame:
    return pd.read_parquet(settings.enriched_parquet)


# ----------------------------------------------------------------- supervisor
def supervisor(state: ReportState) -> str:
    """Decide which specialist runs next based on what's still missing."""
    if not state.get("sentiment_report"):
        return "sentiment_agent"
    if not state.get("topic_report"):
        return "topic_agent"
    if not state.get("entity_report"):
        return "entity_agent"
    if not state.get("summary_report"):
        return "summary_agent"
    return "final_agent"


# ----------------------------------------------------------------- specialists
def sentiment_agent(state: ReportState) -> dict:
    df = _data()
    dist = df["sentiment"].value_counts(normalize=True).mul(100).round(1)
    report = "; ".join(f"{k}: {v}%" for k, v in dist.items())
    return {"sentiment_report": report}


def topic_agent(state: ReportState) -> dict:
    df = _data()
    counts = df[df.topic != -1]["topic"].value_counts().head(6)
    report = ", ".join(f"topic {t} ({c})" for t, c in counts.items())
    return {"topic_report": report}


def entity_agent(state: ReportState) -> dict:
    from collections import Counter

    df = _data()
    ents: Counter = Counter()
    kws: Counter = Counter()
    for row in df.get("entities", []):
        if row is not None:
            ents.update(map(str, row))
    for row in df.get("keywords", []):
        if row is not None:
            kws.update(map(str, row))
    top_e = ", ".join(e for e, _ in ents.most_common(10))
    top_k = ", ".join(k for k, _ in kws.most_common(10))
    return {"entity_report": f"Entities: {top_e}\nKeywords: {top_k}"}


def summary_agent(state: ReportState) -> dict:
    return {"summary_report": summarize(state.get("request", "the video"))}


FINAL_PROMPT = ChatPromptTemplate.from_template(
    """You are an analyst. Write a concise intelligence report on the YouTube
comments about: {request}

Use these findings:
- Sentiment: {sentiment_report}
- Main topics: {topic_report}
- {entity_report}
- Summary of opinions:
{summary_report}

Produce a short report with sections: Overall Sentiment, Key Topics,
Notable Entities, and Takeaways."""
)


def final_agent(state: ReportState) -> dict:
    chain = FINAL_PROMPT | _llm() | StrOutputParser()
    return {"final_report": chain.invoke(dict(state))}


# ----------------------------------------------------------------- graph
ROUTES = {
    "sentiment_agent": "sentiment_agent",
    "topic_agent": "topic_agent",
    "entity_agent": "entity_agent",
    "summary_agent": "summary_agent",
    "final_agent": "final_agent",
}


@lru_cache(maxsize=1)
def build_supervisor():
    builder = StateGraph(ReportState)
    for name, fn in [
        ("sentiment_agent", sentiment_agent),
        ("topic_agent", topic_agent),
        ("entity_agent", entity_agent),
        ("summary_agent", summary_agent),
        ("final_agent", final_agent),
    ]:
        builder.add_node(name, fn)

    builder.set_conditional_entry_point(supervisor, ROUTES)
    for node in ("sentiment_agent", "topic_agent", "entity_agent", "summary_agent"):
        builder.add_conditional_edges(node, supervisor, ROUTES)
    builder.add_edge("final_agent", END)
    return builder.compile()


def generate_report(request: str = "the video") -> str:
    result = build_supervisor().invoke({"request": request})
    return result["final_report"]


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    mlflow.langchain.autolog()
    with mlflow.start_run(run_name="supervisor_report"):
        print(generate_report("the GTA VI trailer"))


if __name__ == "__main__":
    main()
