"""Evaluate RAG retrieval + generation and log everything to MLflow.

Metrics:
- Retrieval: hit-rate / MRR against a small hand-built question→keyword set.
- Generation: answer length, latency, and a simple groundedness heuristic
  (fraction of answers that cite at least one retrieved comment).

Extend `EVAL_SET` with your own gold questions for the report.
"""
from __future__ import annotations

import time

import mlflow

from config import settings
from src.rag.generation import answer_question
from src.rag.retrieval import semantic_retriever

# Each item: a question and keywords we expect to appear in good retrieved docs.
EVAL_SET = [
    {"q": "What do people think about the graphics?", "expect": ["graphic", "look", "detail"]},
    {"q": "Are people excited about the release date?", "expect": ["2026", "wait", "release", "date"]},
    {"q": "What complaints do viewers have?", "expect": ["no", "bad", "disappoint", "yapping"]},
    {"q": "What is mentioned about the characters?", "expect": ["lucia", "character", "protagonist"]},
]


def evaluate_retrieval(k: int | None = None) -> dict:
    retr = semantic_retriever(k=k or settings.top_k)
    hits, rr = 0, 0.0
    for item in EVAL_SET:
        docs = retr.invoke(item["q"])
        rank = next(
            (
                i
                for i, d in enumerate(docs, 1)
                if any(e in d.page_content.lower() for e in item["expect"])
            ),
            None,
        )
        if rank:
            hits += 1
            rr += 1 / rank
    n = len(EVAL_SET)
    return {"hit_rate": hits / n, "mrr": rr / n}


def evaluate_generation() -> dict:
    latencies, lengths, grounded = [], [], 0
    for item in EVAL_SET:
        t0 = time.perf_counter()
        ans = answer_question(item["q"])
        latencies.append(time.perf_counter() - t0)
        lengths.append(len(ans.split()))
        if "[" in ans:  # cited at least one comment
            grounded += 1
    n = len(EVAL_SET)
    return {
        "avg_latency_s": sum(latencies) / n,
        "avg_answer_words": sum(lengths) / n,
        "groundedness": grounded / n,
    }


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_params(
            {
                "llm_model": settings.llm_model,
                "embed_model": settings.embed_model,
                "top_k": settings.top_k,
            }
        )
        retr_metrics = evaluate_retrieval()
        gen_metrics = evaluate_generation()
        mlflow.log_metrics({**retr_metrics, **gen_metrics})
        print("Retrieval:", retr_metrics)
        print("Generation:", gen_metrics)
        print("Logged to MLflow. Run `mlflow ui` to inspect.")


if __name__ == "__main__":
    main()
