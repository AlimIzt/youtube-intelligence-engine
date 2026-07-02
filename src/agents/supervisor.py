"""Lab 8 supervisor pattern for YouTube comment analysis.

This module keeps the same supervisor workflow from Lab 8: a central
supervisor routes the state to specialist agents, each specialist writes one
part of the report, and control returns to the supervisor until the final
report can be generated.
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
    """Shared graph state for the supervisor workflow.

    Attributes:
        request: User request or analysis topic.
        sentiment_report: Summary produced by the sentiment specialist.
        topic_report: Summary produced by the topic specialist.
        entity_report: Summary produced by the entity/keyword specialist.
        summary_report: General summary of audience opinions.
        final_report: Final combined report produced by the final agent.
    """

    request: str
    sentiment_report: str
    topic_report: str
    entity_report: str
    summary_report: str
    final_report: str


@lru_cache(maxsize=1)
def _llm() -> ChatOllama:
    """Return a cached local Ollama chat model for report generation."""
    return ChatOllama(model=settings.llm_model, temperature=0)


@lru_cache(maxsize=1)
def _data() -> pd.DataFrame:
    """Load and cache the enriched comments dataframe."""
    return pd.read_parquet(settings.enriched_parquet)


# ----------------------------------------------------------------- supervisor
def supervisor(state: ReportState) -> str:
    """Choose the next specialist based on missing report sections.

    The supervisor checks the shared state in a fixed order. If a section is
    missing, it routes execution to the specialist responsible for that section.
    Once all specialist outputs exist, it routes to the final report agent.

    Args:
        state: Current LangGraph state.

    Returns:
        Name of the next graph node to execute.
    """
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
    """Create a sentiment distribution summary from enriched comments."""
    df = _data()
    dist = df["sentiment"].value_counts(normalize=True).mul(100).round(1)
    report = "; ".join(f"{k}: {v}%" for k, v in dist.items())
    return {"sentiment_report": report}


def topic_agent(state: ReportState) -> dict:
    """Create a short topic-count summary from assigned topic labels."""
    df = _data()
    counts = df[df.topic != -1]["topic"].value_counts().head(6)
    report = ", ".join(f"topic {t} ({c})" for t, c in counts.items())
    return {"topic_report": report}


def entity_agent(state: ReportState) -> dict:
    """Aggregate the most common entities and keywords from the dataset."""
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
    """Generate a general natural-language summary for the request."""
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
    """Combine specialist outputs into the final intelligence report."""
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
    """Build and cache the LangGraph supervisor workflow.

    Returns:
        A compiled LangGraph application that routes between specialist agents
        until the final report is produced.
    """
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
    """Generate a full comment intelligence report.

    Args:
        request: Topic or user request to analyze.

    Returns:
        Final report text produced by the supervisor workflow.
    """
    result = build_supervisor().invoke({"request": request})
    return result["final_report"]


def main() -> None:
    """Run the supervisor workflow with MLflow logging enabled."""
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    mlflow.langchain.autolog()

    with mlflow.start_run(run_name="supervisor_report"):
        print(generate_report("the GTA VI trailer"))


if __name__ == "__main__":
    main()