"""Audience-level metrics, sentiment over time, and comment length."""
from __future__ import annotations

import plotly.express as px
import pandas as pd
import streamlit as st

from app.data import Scope, stats
from app.theme import explain, style


def render(scope: Scope) -> None:
    df = scope.df
    texts = scope.texts
    n = scope.n
    strategy = scope.strategy

    s = stats(n, strategy)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Comments analyzed", f"{len(df):,}")
    if "sentiment" in df:
        c2.metric("Positive", f"{(df.sentiment == 'positive').mean() * 100:.0f}%")
    if "likes" in df:
        c3.metric("Total likes", f"{int(df.likes.sum()):,}")
    c4.metric("Avg words / comment", f"{s['avg_words']:.1f}")
    from src.analysis.extras import question_share
    c5.metric("Questions", f"{question_share(texts) * 100:.0f}%")

    if "sentiment" in df and "published_at" in df:
        st.subheader("Sentiment over time")
        explain("Each line shows how many positive, neutral and negative comments "
                "were posted per month — spikes usually match trailers, "
                "announcements or news.")
        tl = (df.dropna(subset=["published_at"])
                .assign(month=lambda d: d.published_at.dt.strftime("%Y-%m"))
                .groupby(["month", "sentiment"]).size().reset_index(name="count"))
        st.plotly_chart(style(px.line(
            tl, x="month", y="count", color="sentiment",
            color_discrete_map={"positive": "#4ade80", "neutral": "#94a3b8",
                                "negative": "#f87171"})), width="stretch")

    st.subheader("Comment length")
    explain("How long people write. Mostly short comments = quick reactions; "
            "long tails = engaged fans writing mini-reviews.")
    st.plotly_chart(style(px.histogram(
        pd.DataFrame({"words": s["lengths"]}), x="words", nbins=40,
        color_discrete_sequence=["#818cf8"])), width="stretch")
    st.caption(f"Overall readability: Flesch reading ease {s['reading_ease']:.0f} "
               f"(higher = simpler), ≈ grade {s['grade_level']:.1f} level.")
