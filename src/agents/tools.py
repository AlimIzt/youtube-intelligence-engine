"""Tools the agent can route to. Each is traced for MLflow monitoring (Lab 8)."""
from __future__ import annotations

import mlflow
import pandas as pd
from langchain_core.tools import tool

from config import settings
from src.rag.generation import answer_question, summarize
from src.rag.retrieval import metadata_retriever


def _load_df() -> pd.DataFrame:
    return pd.read_parquet(settings.enriched_parquet)


@tool
@mlflow.trace(span_type="TOOL")
def comment_qa(question: str) -> str:
    """Answer a factual question about the video's comments using RAG retrieval.
    Use for questions like 'what do people say about the graphics?'."""
    return answer_question(question)


@tool
@mlflow.trace(span_type="TOOL")
def comment_summary(topic: str) -> str:
    """Summarize overall opinions about a topic mentioned in the comments.
    Use for requests like 'summarize what people think about the release date'."""
    return summarize(topic)


@tool
@mlflow.trace(span_type="TOOL")
def sentiment_insight(aspect: str = "") -> str:
    """Report the sentiment breakdown (and optionally for a specific aspect).
    Use for 'are people positive or negative?' style questions."""
    df = _load_df()
    dist = df["sentiment"].value_counts(normalize=True).mul(100).round(1)
    lines = [f"Overall sentiment across {len(df):,} comments:"]
    lines += [f"  - {k}: {v}%" for k, v in dist.items()]
    if aspect:
        retr = metadata_retriever({"sentiment": "negative"}, k=5)
        neg = retr.invoke(aspect)
        lines.append(f"\nSample negative comments about '{aspect}':")
        lines += [f"  • {d.page_content}" for d in neg]
    return "\n".join(lines)


@tool
@mlflow.trace(span_type="TOOL")
def topic_insight(_: str = "") -> str:
    """List the main discussion topics found in the comments."""
    df = _load_df()
    if "topic" not in df.columns:
        return "Topic modeling has not been run yet."
    counts = df[df.topic != -1]["topic"].value_counts().head(8)
    return "Top topics by comment volume:\n" + "\n".join(
        f"  - topic {t}: {c} comments" for t, c in counts.items()
    )


ALL_TOOLS = [comment_qa, comment_summary, sentiment_insight, topic_insight]
