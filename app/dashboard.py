"""Streamlit dashboard: insights + chat with the agent.

Run:  streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from config import settings

st.set_page_config(page_title="YouTube Intelligence Engine", layout="wide")
st.title("🎬 YouTube Intelligence Engine")


@st.cache_data
def load_data() -> pd.DataFrame:
    if settings.enriched_parquet.exists():
        return pd.read_parquet(settings.enriched_parquet)
    if settings.clean_csv.exists():
        return pd.read_csv(settings.clean_csv)
    return pd.DataFrame()


df = load_data()

if df.empty:
    st.warning("No data yet. Run the pipeline scripts (01–04) first.")
    st.stop()

tab_overview, tab_insights, tab_chat = st.tabs(["Overview", "Insights", "Ask the Agent"])

with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Comments", f"{len(df):,}")
    if "sentiment" in df:
        pos = (df.sentiment == "positive").mean() * 100
        c2.metric("Positive", f"{pos:.0f}%")
    if "likes" in df:
        c3.metric("Total likes", f"{int(df.likes.sum()):,}")
    st.dataframe(df.head(50), use_container_width=True)

with tab_insights:
    if "sentiment" in df:
        st.subheader("Sentiment distribution")
        counts = df.sentiment.value_counts().reset_index()
        counts.columns = ["sentiment", "count"]
        st.plotly_chart(
            px.bar(counts, x="sentiment", y="count", color="sentiment"),
            use_container_width=True,
        )
    if "topic" in df:
        st.subheader("Top topics by volume")
        tc = df[df.topic != -1].topic.value_counts().head(10).reset_index()
        tc.columns = ["topic", "count"]
        st.plotly_chart(px.bar(tc, x="topic", y="count"), use_container_width=True)

with tab_chat:
    st.subheader("Ask about the comments")
    st.caption("Powered by the LangGraph agent + RAG (needs Ollama running).")
    q = st.text_input("Your question", "What do people think about the graphics?")
    if st.button("Ask"):
        with st.spinner("Thinking ..."):
            try:
                from src.agents.orchestrator import ask

                st.write(ask(q))
            except Exception as e:
                st.error(f"Agent error: {e}")
