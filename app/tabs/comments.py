"""Browse the full dataset with search, filters, and sorting."""
from __future__ import annotations

import streamlit as st

from app.data import Scope
from app.theme import explain


def render(scope: Scope) -> None:
    full_df = scope.full

    st.subheader("Browse every comment")
    explain("The full dataset with filters and sorting — search any word, "
            "keep only one sentiment, or find the most-liked comments.")

    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    q = f1.text_input("Search text or author", "")
    sent_opts = sorted(full_df.sentiment.dropna().unique()) if "sentiment" in full_df else []
    sel_sent = f2.multiselect("Sentiment", sent_opts, default=sent_opts)
    min_likes = f3.number_input("Min likes", 0, step=1)
    sort_by = f4.selectbox("Sort by", ["Newest", "Oldest", "Most liked", "Longest"])

    view = full_df.copy()
    if q:
        mask = view["text"].astype(str).str.contains(q, case=False, na=False)
        if "author" in view:
            mask |= view["author"].astype(str).str.contains(q, case=False, na=False)
        view = view[mask]
    if sent_opts and "sentiment" in view:
        view = view[view.sentiment.isin(sel_sent)]
    if "likes" in view:
        view = view[view.likes >= min_likes]
    if "published_at" in view and view.published_at.notna().any():
        dmin, dmax = view.published_at.min(), view.published_at.max()
        d1, d2 = st.slider("Date range", dmin.to_pydatetime(), dmax.to_pydatetime(),
                           (dmin.to_pydatetime(), dmax.to_pydatetime()))
        view = view[(view.published_at >= d1) & (view.published_at <= d2)]

    if sort_by == "Most liked" and "likes" in view:
        view = view.sort_values("likes", ascending=False)
    elif sort_by == "Longest":
        view = view.assign(_len=view.text.astype(str).str.len()).sort_values(
            "_len", ascending=False).drop(columns="_len")
    elif "published_at" in view:
        view = view.sort_values("published_at", ascending=(sort_by == "Oldest"))

    cols = [c for c in ["author", "text", "sentiment", "likes", "published_at",
                        "keywords", "topic"] if c in view]
    st.caption(f"{len(view):,} comments match")
    st.dataframe(view[cols], width="stretch", height=520,
                 column_config={"text": st.column_config.TextColumn(width="large")})
    st.download_button("Download filtered CSV", view[cols].to_csv(index=False),
                       "comments_filtered.csv", "text/csv")
