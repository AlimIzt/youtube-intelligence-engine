"""Streamlit dashboard entry point: insights + chat with the agent.

Run:  streamlit run app/dashboard.py   ->  http://localhost:5000

Layout of the app package:
    app/theme.py    global CSS, animated hero, explain/takeaway/style helpers
    app/data.py     data loading, sidebar sample scope, cached analytics
    app/tabs/       one module per tab, each exposing render(scope)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app import theme
from app.data import load_data, sidebar_scope
from app.tabs import TABS

st.set_page_config(page_title="YouTube Intelligence Engine", layout="wide")
theme.inject()
theme.hero()

full_df = load_data()
if full_df.empty:
    st.warning("No data yet. Run the pipeline scripts (02–04) first.")
    st.stop()

scope = sidebar_scope(full_df)
st.caption(f"Analyzing **{len(scope.df):,}** of {len(full_df):,} comments "
           f"({scope.label}).")

for tab, (label, render) in zip(st.tabs([label for label, _ in TABS]), TABS):
    with tab:
        render(scope)
