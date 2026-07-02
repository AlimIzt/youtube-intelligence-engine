"""Tab registry: (label, render) pairs in display order.

Each tab lives in its own module and exposes `render(scope: Scope) -> None`.
To add a tab, create a module here and append it to TABS.
"""
from __future__ import annotations

from app.tabs import (agent, comments, entities, faq, overview, rag_pipeline,
                      sentiment, words)

TABS = [
    ("Overview", overview.render),
    ("Comments", comments.render),
    ("Sentiment & Emotion", sentiment.render),
    ("Words & Phrases", words.render),
    ("Entities & Graph", entities.render),
    ("Ask the Agent", agent.render),
    ("RAG Pipeline", rag_pipeline.render),
    ("FAQ", faq.render),
]
