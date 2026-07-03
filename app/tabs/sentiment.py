"""Sentiment, emotions, subjectivity, languages, and topics."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.data import Scope, emotions, languages, subjectivity_df, topic_labels
from app.theme import explain, takeaway, style


def render(scope: Scope) -> None:
    df = scope.df
    texts = scope.texts
    n = scope.n
    strategy = scope.strategy

    col1, col2 = st.columns(2)

    if "sentiment" in df:
        with col1:
            st.subheader("Sentiment")
            explain("Every comment is automatically labeled positive, neutral or "
                    "negative based on the words and emojis it uses (VADER).")
            counts = df.sentiment.value_counts().reset_index()
            counts.columns = ["sentiment", "count"]
            st.plotly_chart(style(px.bar(
                counts, x="sentiment", y="count", color="sentiment",
                color_discrete_map={"positive": "#4ade80", "neutral": "#94a3b8",
                                    "negative": "#f87171"})), width="stretch")
            pos = (df.sentiment == "positive").mean() * 100
            neg = (df.sentiment == "negative").mean() * 100
            mood = ("overwhelmingly positive" if pos > 60 else
                    "mostly positive" if pos > neg else
                    "mostly negative" if neg > pos else "mixed")
            takeaway(f"The audience mood is {mood}: {pos:.0f}% positive vs "
                     f"{neg:.0f}% negative in this sample.")

    with col2:
        st.subheader("Emotions")
        explain("Goes deeper than positive/negative: counts words linked to eight "
                "basic emotions (NRC emotion lexicon). 'Anticipation' being high "
                "on a trailer means people are hyped for release.")
        emo = emotions(n, strategy)
        st.plotly_chart(style(px.bar(
            emo, x="emotion", y="count", color="emotion",
            color_discrete_sequence=px.colors.qualitative.Bold)), width="stretch")
        if not emo.empty:
            top_emo = emo.sort_values("count", ascending=False).iloc[0]
            takeaway(f"The dominant emotion is **{top_emo.emotion}**.")
        pick = st.selectbox("Show the most emotional comments for", emo.emotion)
        if st.button("Find comments"):
            from src.analysis.extras import top_emotion_comments

            st.dataframe(top_emotion_comments(texts, pick), width="stretch",
                         hide_index=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Opinion vs fact (subjectivity)")
        explain("Each dot is a comment. Right = positive, left = negative, "
                "high = pure opinion ('best game ever'), low = factual "
                "('releases in 2026'). TextBlob.")
        sub = subjectivity_df(min(n, 2000), strategy)
        st.plotly_chart(style(px.scatter(
            sub, x="polarity", y="subjectivity", hover_data=["text"],
            opacity=.45, color_discrete_sequence=["#a5b4fc"])), width="stretch")
        takeaway(f"{(sub.subjectivity > .5).mean() * 100:.0f}% of comments are "
                 "more opinion than fact.")

    with col4:
        st.subheader("Languages")
        explain("Automatic language detection — shows how international the "
                "audience is.")
        lang = languages(min(n, 2000), strategy)
        st.plotly_chart(style(px.pie(lang.head(8), names="language",
                                     values="count", hole=.45)), width="stretch")
        if len(lang):
            takeaway(f"{len(lang)} languages detected; "
                     f"'{lang.iloc[0].language}' leads.")

    if "topic" in df:
        st.subheader("What people talk about (topics)")
        explain("Comments were automatically grouped by subject using BERTopic — "
                "each bar is a cluster of comments about the same thing "
                "(topic -1 = uncategorized noise, hidden here). Hover a bar to "
                "see the topic's top keywords.")
        topic_df = df[df.topic != -1].copy()
        labels = topic_labels(n, strategy)
        tc = topic_df.topic.value_counts().head(10).reset_index()
        tc.columns = ["topic", "count"]
        tc["keywords"] = tc["topic"].map(lambda t: labels.get(int(t), ""))

        st.plotly_chart(style(px.bar(tc, x="topic", y="count",
                                     hover_data={"keywords": True},
                                     color_discrete_sequence=["#fb7185"])),
                        width="stretch")

        if not tc.empty:
            top = tc.iloc[0]
            kw = f" ({top.keywords})" if top.keywords else ""
            takeaway(f"The largest visible topic is **topic {top.topic}**{kw} "
                     f"with {top['count']} comments.")

        if "sentiment" in df and not topic_df.empty:
            st.subheader("Topics by sentiment")
            explain("This breaks the same topic clusters down by sentiment. It "
                    "helps show whether positive, neutral and negative comments "
                    "are talking about the same subjects or different ones. Hover "
                    "a bar to see the topic's top keywords.")

            top_topics = topic_df.topic.value_counts().head(8).index
            topic_sent = (
                topic_df[topic_df.topic.isin(top_topics)]
                .groupby(["topic", "sentiment"])
                .size()
                .reset_index(name="count")
            )
            topic_sent["keywords"] = topic_sent["topic"].map(
                lambda t: labels.get(int(t), ""))

            st.plotly_chart(style(px.bar(
                topic_sent,
                x="topic",
                y="count",
                color="sentiment",
                barmode="group",
                hover_data={"keywords": True},
                color_discrete_map={
                    "positive": "#4ade80",
                    "neutral": "#94a3b8",
                    "negative": "#f87171",
                },
            )), width="stretch")

            dominant = (
                topic_sent.sort_values("count", ascending=False)
                .head(1)
            )

            if not dominant.empty:
                row = dominant.iloc[0]
                kw = f" ({row.keywords})" if row.keywords else ""
                takeaway(f"The strongest topic-sentiment pair is **topic "
                         f"{row.topic}**{kw} with **{row.sentiment}** comments "
                         f"({row['count']} comments in this sample).")