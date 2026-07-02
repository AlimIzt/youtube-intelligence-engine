"""Streamlit dashboard: insights + chat with the agent.

Run:  streamlit run app/dashboard.py   ->  http://localhost:5000
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from config import settings

st.set_page_config(page_title="YouTube Intelligence Engine", layout="wide")

# App-level theming (metric cards, explainer/takeaway callouts, background).
st.markdown("""
<style>
.stApp { background: radial-gradient(ellipse at top left, rgba(99,102,241,.08), transparent 50%),
                     radial-gradient(ellipse at bottom right, rgba(244,63,94,.06), transparent 50%),
                     #060609; }
div[data-testid="stMetric"] { background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08);
  border-radius:14px; padding:1rem 1.2rem; }
.explain { border-left:3px solid #818cf8; background:rgba(129,140,248,.07);
  padding:.65rem .9rem; border-radius:0 10px 10px 0; color:rgba(255,255,255,.75);
  font-size:.9rem; margin-bottom:.8rem; }
.takeaway { border-left:3px solid #fb7185; background:rgba(251,113,133,.07);
  padding:.65rem .9rem; border-radius:0 10px 10px 0; color:rgba(255,255,255,.85);
  font-size:.92rem; margin:.4rem 0 .8rem; }
hr { border-color: rgba(255,255,255,.08); }
</style>
""", unsafe_allow_html=True)

# Animated "Ethereal Shadow" hero — SVG turbulence + displacement filter with an
# animated hue-rotate (native SMIL, no framer-motion), rendered in a self-contained
# frame so Streamlit's HTML sanitizer doesn't strip the SVG filter primitives.
# Ported from the etheral-shadow React/shadcn component to pure SVG/CSS.
_MASK = "https://framerusercontent.com/images/ceBGguIpUU8luwByxuQz79t7To.png"
_NOISE = "https://framerusercontent.com/images/g0QcWrxr87K0ufOxIUFBakwYA8.png"

components.html(f"""
<div class="wrap">
  <svg width="0" height="0" style="position:absolute">
    <defs>
      <filter id="eshadow">
        <feTurbulence result="undulation" numOctaves="2" baseFrequency="0.0005,0.002"
                      seed="0" type="turbulence"/>
        <feColorMatrix in="undulation" type="hueRotate" values="180">
          <animate attributeName="values" from="0" to="360" dur="8s"
                   repeatCount="indefinite"/>
        </feColorMatrix>
        <feColorMatrix in="dist" result="circulation" type="matrix"
                       values="4 0 0 0 1  4 0 0 0 1  4 0 0 0 1  1 0 0 0 0"/>
        <feDisplacementMap in="SourceGraphic" in2="circulation" scale="45" result="dist"/>
        <feDisplacementMap in="dist" in2="undulation" scale="45" result="output"/>
      </filter>
    </defs>
  </svg>
  <div class="fx"><div class="shape"></div></div>
  <div class="noise"></div>
  <div class="content">
    <div class="badge"><span class="dot"></span>NLP · RAG · LLM AGENTS</div>
    <h1>YouTube Intelligence <span>Engine</span></h1>
    <p>Turning thousands of raw comments into insights anyone can read.</p>
  </div>
</div>
<style>
  html, body {{ margin:0; background:transparent; overflow:hidden; }}
  .wrap {{ position:relative; height:290px; border-radius:18px; overflow:hidden;
    background:#08080d; border:1px solid rgba(255,255,255,.08);
    font-family:'Source Sans Pro', system-ui, sans-serif; }}
  .fx {{ position:absolute; inset:-45px; filter:url(#eshadow) blur(4px); }}
  .shape {{ width:100%; height:100%; background-color:rgba(150,160,225,.85);
    -webkit-mask-image:url('{_MASK}'); mask-image:url('{_MASK}');
    -webkit-mask-size:cover; mask-size:cover;
    -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat;
    -webkit-mask-position:center; mask-position:center; }}
  .noise {{ position:absolute; inset:0; background-image:url('{_NOISE}');
    background-size:240px; background-repeat:repeat; opacity:.5; }}
  .content {{ position:absolute; inset:0; display:flex; flex-direction:column;
    align-items:center; justify-content:center; text-align:center; z-index:10; }}
  .badge {{ display:inline-flex; align-items:center; gap:.5rem; padding:.3rem .9rem;
    border-radius:999px; background:rgba(0,0,0,.35); border:1px solid rgba(255,255,255,.12);
    color:rgba(255,255,255,.7); font-size:.8rem; letter-spacing:.05em; margin-bottom:1rem;
    backdrop-filter:blur(6px); }}
  .dot {{ width:8px; height:8px; border-radius:50%; background:#fb7185; }}
  h1 {{ font-size:3rem; font-weight:800; letter-spacing:-.02em; margin:0; line-height:1.1;
    background:linear-gradient(180deg,#fff,rgba(255,255,255,.75));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  h1 span {{ background:linear-gradient(90deg,#a5b4fc,#e2e8f0,#fda4af);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  p {{ color:rgba(255,255,255,.55); font-weight:300; margin-top:.7rem; }}
</style>
""", height=300)


def explain(text: str):
    st.markdown(f'<div class="explain">{text}</div>', unsafe_allow_html=True)


def takeaway(text: str):
    st.markdown(f'<div class="takeaway"><b>Takeaway:</b> {text}</div>',
                unsafe_allow_html=True)


PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)")


def style(fig):
    fig.update_layout(**PLOT, margin=dict(t=30, b=10))
    return fig


# ----------------------------------------------------------------------------
# data + global scope (every analysis respects the sidebar sample settings)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    if settings.enriched_parquet.exists():
        df = pd.read_parquet(settings.enriched_parquet)
    elif settings.clean_csv.exists():
        df = pd.read_csv(settings.clean_csv)
    else:
        return pd.DataFrame()
    if "published_at" in df:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    return df


full_df = load_data()
if full_df.empty:
    st.warning("No data yet. Run the pipeline scripts (02–04) first.")
    st.stop()

with st.sidebar:
    st.header("Analysis scope")
    st.caption("These settings control how many comments every chart and "
               "NLP method below is computed on.")
    use_all = st.toggle(f"Use all {len(full_df):,} comments", value=False)
    n = len(full_df) if use_all else st.slider(
        "Comments to analyze", 100, len(full_df),
        min(1000, len(full_df)), step=100)
    strategy = st.selectbox(
        "Which comments to pick",
        ["First in dataset", "Random sample", "Most liked", "Newest"],
        disabled=use_all)
    st.caption("Tip: heavier methods (relations, emotions, languages) get "
               "slow above ~3,000 comments.")


@st.cache_data
def scoped(n: int, strategy: str) -> pd.DataFrame:
    df = load_data()
    if n >= len(df):
        return df
    if strategy == "Random sample":
        return df.sample(n, random_state=42)
    if strategy == "Most liked" and "likes" in df:
        return df.sort_values("likes", ascending=False).head(n)
    if strategy == "Newest" and "published_at" in df:
        return df.sort_values("published_at", ascending=False).head(n)
    return df.head(n)


df = scoped(n, strategy)
texts = df["text"].dropna().astype(str).tolist()
st.caption(f"Analyzing **{len(df):,}** of {len(full_df):,} comments "
           f"({strategy.lower() if not use_all else 'all'}).")


@st.cache_data
def list_column_counts(n: int, strategy: str, col: str, top: int) -> pd.DataFrame:
    d = scoped(n, strategy)
    counter: Counter = Counter()
    if col not in d:
        return pd.DataFrame(columns=[col, "count"])
    for row in d[col]:
        if row is not None:
            counter.update([str(x) for x in row])
    return pd.DataFrame(counter.most_common(top), columns=[col, "count"])


@st.cache_data
def collocations(n: int, strategy: str, gram: int, top: int) -> pd.DataFrame:
    from src.analysis.collocations import pmi_collocations
    return pmi_collocations(scoped(n, strategy)["text"].dropna().astype(str).tolist(),
                            n=gram, top=top)


@st.cache_data
def ngrams(n: int, strategy: str, gram: int, top: int) -> pd.DataFrame:
    from src.analysis.collocations import top_ngrams
    return top_ngrams(scoped(n, strategy)["text"].dropna().astype(str).tolist(),
                      n=gram, top=top)


@st.cache_data
def word_freq(n: int, strategy: str, top: int) -> pd.DataFrame:
    from src.analysis.collocations import word_frequencies
    return word_frequencies(scoped(n, strategy)["text"].dropna().astype(str).tolist(), top)


@st.cache_resource
def _spacy():
    import spacy
    return spacy.load(settings.spacy_model, disable=["ner", "lemmatizer"])


@st.cache_data
def pos_distribution(n: int, strategy: str) -> pd.DataFrame:
    nlp = _spacy()
    t = scoped(n, strategy)["text"].dropna().astype(str).tolist()
    counter: Counter = Counter()
    for doc in nlp.pipe(t, batch_size=128):
        counter.update(tok.pos_ for tok in doc if not tok.is_space)
    return pd.DataFrame(counter.most_common(), columns=["POS", "count"])


@st.cache_data
def relations(n: int, strategy: str) -> list[str]:
    from src.analysis.relations import relations_for_corpus
    t = scoped(n, strategy)["text"].dropna().astype(str).tolist()
    return relations_for_corpus(t, limit=len(t))


@st.cache_data
def kg_dot(n: int, strategy: str, max_edges: int) -> str:
    from src.analysis.relations import knowledge_graph_dot
    return knowledge_graph_dot(scoped(n, strategy)["text"].dropna().astype(str).tolist(),
                               max_edges=max_edges)


@st.cache_data
def wordcloud_png(n: int, strategy: str) -> bytes:
    import io
    from wordcloud import WordCloud
    from src.preprocessing.nltk_clean import clean_join

    t = scoped(n, strategy)["text"].dropna().astype(str).tolist()
    text = " ".join(clean_join(x) for x in t) or "none"
    wc = WordCloud(width=900, height=400, background_color="#0b0b12",
                   colormap="plasma", max_words=120).generate(text)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data
def emotions(n: int, strategy: str) -> pd.DataFrame:
    from src.analysis.extras import emotion_scores
    return emotion_scores(scoped(n, strategy)["text"].dropna().astype(str).tolist())


@st.cache_data
def languages(n: int, strategy: str) -> pd.DataFrame:
    from src.analysis.extras import detect_languages
    return detect_languages(scoped(n, strategy)["text"].dropna().astype(str).tolist())


@st.cache_data
def emojis(n: int, strategy: str) -> pd.DataFrame:
    from src.analysis.extras import emoji_counts
    return emoji_counts(scoped(n, strategy)["text"].dropna().astype(str).tolist())


@st.cache_data
def subjectivity_df(n: int, strategy: str) -> pd.DataFrame:
    from src.analysis.extras import subjectivity
    return subjectivity(scoped(n, strategy)["text"].dropna().astype(str).tolist())


@st.cache_data
def stats(n: int, strategy: str) -> dict:
    from src.analysis.extras import text_stats
    return text_stats(scoped(n, strategy)["text"].dropna().astype(str).tolist())


@st.cache_data
def tfidf_terms(n: int, strategy: str) -> dict:
    from src.analysis.extras import distinctive_terms
    return distinctive_terms(scoped(n, strategy))


@st.cache_data
def gliner_entities(n: int, strategy: str, labels: tuple[str, ...]) -> pd.DataFrame:
    from src.analysis.gliner_ner import extract_domain_entities
    return extract_domain_entities(
        scoped(n, strategy)["text"].dropna().astype(str).tolist(), labels)


(tab_overview, tab_comments, tab_sent, tab_words, tab_ents, tab_chat, tab_rag,
 tab_faq) = st.tabs(
    ["Overview", "Comments", "Sentiment & Emotion", "Words & Phrases",
     "Entities & Graph", "Ask the Agent", "RAG Pipeline", "FAQ"]
)

# ----------------------------------------------------------------------------
with tab_overview:
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

# ----------------------------------------------------------------------------
with tab_comments:
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

# ----------------------------------------------------------------------------
with tab_sent:
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
                "(topic -1 = uncategorized noise, hidden here).")
        tc = df[df.topic != -1].topic.value_counts().head(10).reset_index()
        tc.columns = ["topic", "count"]
        st.plotly_chart(style(px.bar(tc, x="topic", y="count",
                                     color_discrete_sequence=["#fb7185"])),
                        width="stretch")

# ----------------------------------------------------------------------------
with tab_words:
    st.subheader("Word cloud & most frequent words")
    explain("The bigger the word, the more often people use it. Stop-words "
            "('the', 'and'…) are removed first so only meaningful words remain.")
    col5, col6 = st.columns([2, 1])
    with col5:
        st.image(wordcloud_png(n, strategy), width="stretch")
    with col6:
        st.dataframe(word_freq(n, strategy, 20), width="stretch", hide_index=True)

    st.subheader("Top emojis")
    explain("Emojis carry a lot of sentiment on YouTube and tell you as "
            "much as words do.")
    em = emojis(n, strategy)
    if em.empty:
        st.caption("No emojis found in this sample.")
    else:
        st.plotly_chart(style(px.bar(em, x="emoji", y="count",
                                     color_discrete_sequence=["#fbbf24"])),
                        width="stretch")

    st.subheader("Phrases people repeat")
    explain("Left/middle: word pairs and triples that appear together far more "
            "often than chance (PMI collocations) — these are set phrases like "
            "game titles. Right: simply the most repeated two-word sequences.")
    col7, col8, col9 = st.columns(3)
    with col7:
        st.caption("Strongest word pairs (PMI bigrams)")
        st.dataframe(collocations(n, strategy, 2, 15), width="stretch", hide_index=True)
    with col8:
        st.caption("Strongest word triples (PMI trigrams)")
        st.dataframe(collocations(n, strategy, 3, 15), width="stretch", hide_index=True)
    with col9:
        st.caption("Most frequent bigrams")
        st.dataframe(ngrams(n, strategy, 2, 15), width="stretch", hide_index=True)

    if "sentiment" in df:
        st.subheader("What fans vs critics say differently")
        explain("TF-IDF finds the words most characteristic of each sentiment "
                "group — the vocabulary that separates praise from complaints.")
        terms = tfidf_terms(n, strategy)
        cols = st.columns(max(len(terms), 1))
        for col, (lab, ws) in zip(cols, terms.items()):
            with col:
                st.caption(lab.capitalize())
                st.write("  \n".join(f"• {w}" for w in ws))

    st.subheader("Grammar profile (part-of-speech)")
    explain("Breaks all words into grammar categories. Many adjectives = a "
            "descriptive, opinionated crowd; many verbs = people describing "
            "actions and gameplay.")
    st.plotly_chart(style(px.bar(pos_distribution(n, strategy), x="POS", y="count",
                                 color_discrete_sequence=["#818cf8"])),
                    width="stretch")

# ----------------------------------------------------------------------------
with tab_ents:
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Names mentioned (NER)")
        explain("Named-entity recognition automatically spots names of people, "
                "companies, games and places inside the comments.")
        ents = list_column_counts(n, strategy, "entities", 20)
        if not ents.empty:
            st.plotly_chart(style(px.bar(ents, x="count", y="entities",
                                         orientation="h",
                                         color_discrete_sequence=["#a5b4fc"])),
                            width="stretch")
            takeaway(f"Most mentioned: **{ents.iloc[0].entities}**.")
    with col4:
        st.subheader("Keywords")
        explain("The most informative words/phrases per comment (YAKE), "
                "aggregated — a quick summary of what the comments are about.")
        kws = list_column_counts(n, strategy, "keywords", 20)
        if not kws.empty:
            st.plotly_chart(style(px.bar(kws, x="count", y="keywords",
                                         orientation="h",
                                         color_discrete_sequence=["#fda4af"])),
                            width="stretch")

    st.subheader("Domain entities (GLiNER zero-shot NER)")
    explain("Classic NER above only knows fixed categories like PERSON or ORG. "
            "GLiNER matches any label you type at inference time, so it can "
            "find things spaCy has no tag for — game features, platforms, "
            "prices. Edit the labels and re-run to ask for anything else.")
    from src.analysis.gliner_ner import DEFAULT_LABELS
    labels_raw = st.text_input("Labels (comma-separated)",
                               ", ".join(DEFAULT_LABELS))
    st.caption("First run downloads the GLiNER model (~500 MB); inference on "
               "large samples takes a while — lower the sidebar sample size "
               "if it's slow.")
    if st.button("Extract domain entities"):
        with st.spinner("Running GLiNER ..."):
            try:
                labels = tuple(x.strip() for x in labels_raw.split(",") if x.strip())
                gl = gliner_entities(n, strategy, labels)
                if gl.empty:
                    st.caption("No entities found for these labels in this sample.")
                else:
                    st.plotly_chart(style(px.bar(
                        gl.head(25), x="count", y="entity", color="label",
                        orientation="h",
                        color_discrete_sequence=px.colors.qualitative.Bold)),
                        width="stretch")
                    top = gl.iloc[0]
                    takeaway(f"Most mentioned: **{top.entity}** "
                             f"({top.label}, {top['count']} mentions) across "
                             f"{gl.label.nunique()} label types.")
                    with st.expander("Full entity table"):
                        st.dataframe(gl, width="stretch", hide_index=True)
            except Exception as e:
                st.error(f"GLiNER error: {e}")

    st.subheader("Who does what — relations & knowledge graph")
    explain("Extracts simple 'subject → verb → object' statements from comments "
            "(e.g. 'Rockstar delayed game') and draws them as a graph, so you "
            "can see claims, not just words.")
    col10, col11 = st.columns([1, 2])
    with col10:
        rels = relations(n, strategy)
        st.caption(f"{len(rels)} unique relations (showing 25)")
        st.dataframe(pd.DataFrame(rels[:25], columns=["verb(subject, object)"]),
                     width="stretch", hide_index=True)
    with col11:
        st.caption("Knowledge graph (top relations)")
        st.graphviz_chart(kg_dot(n, strategy, 25))

# ----------------------------------------------------------------------------
with tab_chat:
    st.subheader("Ask about the comments")
    explain("Type any question — the system retrieves the most relevant comments "
            "(RAG) and a local LLM answers using only them. Needs Ollama running.")

    c1, c2 = st.columns(2)
    mode = c1.radio(
        "Agent",
        ["Tool-calling (single)", "Multi-agent (router)",
         "Supervisor (full report)", "Swarm (reflection)",
         "DSPy (Chain-of-Thought)", "HyDE retrieval"],
        help="Different agent architectures: a single tool-calling agent, a "
             "router that picks a specialist, a supervisor that writes a full "
             "report, a swarm that critiques its own answer, DSPy "
             "chain-of-thought, and HyDE (retrieves via an LLM-invented "
             "hypothetical comment).")
    retrieval = c2.radio("Retrieval", ["semantic", "mmr"], horizontal=True,
                         help="semantic = most similar comments; mmr = similar "
                              "but diverse, avoids near-duplicates.")
    rerank = st.checkbox("Cross-encoder rerank", value=False,
                         help="A second, slower model re-orders the retrieved "
                              "comments for better relevance.")

    q = st.text_input("Your question", "What do people think about the graphics?")
    if st.button("Ask"):
        with st.spinner("Thinking ..."):
            try:
                if mode.startswith("Multi"):
                    from src.agents.multi_agent import ask_multi
                    cat, resp = ask_multi(q)
                    st.info(f"Routed to: **{cat}**")
                    st.write(resp)
                elif mode.startswith("Supervisor"):
                    from src.agents.supervisor import generate_report
                    st.write(generate_report(q))
                elif mode.startswith("Swarm"):
                    from src.agents.swarm import ask_swarm
                    st.write(ask_swarm(q))
                elif mode.startswith("HyDE"):
                    from src.rag.hyde import hyde_answer, hypothetical_answers
                    hypos = hypothetical_answers(q)
                    st.write(hyde_answer(q, hypotheticals=hypos))
                    with st.expander("Hypothetical comment used for retrieval"):
                        st.write(hypos[0])
                elif mode.startswith("DSPy"):
                    from src.rag.dspy_qa import answer_question_dspy
                    r = answer_question_dspy(q)
                    st.write(r.answer)
                    st.caption(f"Self-rated confidence: {r.confidence}")
                    with st.expander("Reasoning (Chain-of-Thought)"):
                        st.write(getattr(r, "reasoning", ""))
                else:
                    from src.rag.generation import answer_question
                    from src.rag.retrieval import mmr_retriever, semantic_retriever

                    retr = mmr_retriever() if retrieval == "mmr" else semantic_retriever()
                    if rerank:
                        from src.rag.postretrieval import rerank as rr
                        docs = rr(q, retr.invoke(q), top_k=settings.top_k)
                        from src.rag.generation import QA_PROMPT, get_llm
                        from src.rag.retrieval import format_docs
                        from langchain_core.output_parsers import StrOutputParser

                        chain = QA_PROMPT | get_llm() | StrOutputParser()
                        st.write(chain.invoke({"question": q,
                                               "context": format_docs(docs)}))
                    else:
                        st.write(answer_question(q, retriever=retr))
            except Exception as e:
                st.error(f"Agent error: {e}")

# ----------------------------------------------------------------------------
with tab_rag:
    st.subheader("RAG components")
    explain("A peek under the hood of the 'Ask the Agent' tab: how documents are "
            "split into chunks, how different retrieval strategies compare, and "
            "how reranking improves results.")

    st.markdown("### 1 · Chunking strategies")
    join_n = st.slider("Comments to join into one document", 5, 100, 20, step=5)
    chunk_strategy = st.radio(
        "Strategy", ["recursive", "token", "semantic"], horizontal=True,
        help="recursive = split on paragraphs/sentences; token = fixed token "
             "windows; semantic = split where the meaning shifts.")
    if st.button("Chunk"):
        with st.spinner("Chunking ..."):
            try:
                from src.rag import chunking

                doc = "\n".join(texts[:join_n])
                fn = {"recursive": chunking.recursive_chunks,
                      "token": chunking.token_chunks,
                      "semantic": chunking.semantic_chunks}[chunk_strategy]
                chunks = fn(doc)
                st.success(f"{chunk_strategy} → {len(chunks)} chunks")
                for i, c in enumerate(chunks[:8]):
                    st.text(f"[chunk {i}] {c[:200]}")
            except Exception as e:
                st.error(f"Chunking error: {e}")

    st.divider()

    st.markdown("### 2 · Retrieval strategy comparison")
    rq = st.text_input("Query", "graphics look amazing", key="rag_query")
    if st.button("Compare retrieval"):
        with st.spinner("Retrieving ..."):
            try:
                from src.rag.retrieval import (hybrid_retriever, lexical_retriever,
                                               mmr_retriever, semantic_retriever)

                strategies = {"Semantic": semantic_retriever(k=4),
                              "Lexical (BM25)": lexical_retriever(df, k=4),
                              "Hybrid": hybrid_retriever(df, k=4),
                              "MMR": mmr_retriever(k=4)}
                cols = st.columns(len(strategies))
                for col, (name, retr) in zip(cols, strategies.items()):
                    with col:
                        st.caption(name)
                        for d in retr.invoke(rq):
                            st.text(f"• {d.page_content[:80]}")
            except Exception as e:
                st.error(f"Retrieval error: {e}")

    st.divider()

    st.markdown("### 3 · Cross-encoder reranking")
    rrq = st.text_input("Query", "is the game realistic", key="rerank_query")
    if st.button("Rerank"):
        with st.spinner("Reranking ..."):
            try:
                from src.rag.postretrieval import rerank
                from src.rag.retrieval import semantic_retriever

                docs = semantic_retriever(k=10).invoke(rrq)
                reranked = rerank(rrq, docs, top_k=5)
                c_before, c_after = st.columns(2)
                with c_before:
                    st.caption("Before (semantic top 5)")
                    for d in docs[:5]:
                        st.text(f"• {d.page_content[:80]}")
                with c_after:
                    st.caption("After (cross-encoder top 5)")
                    for d in reranked:
                        st.text(f"• {d.page_content[:80]}")
            except Exception as e:
                st.error(f"Rerank error: {e}")
    st.divider()

    st.markdown("### 4 · HyDE (hypothetical document embeddings)")
    explain("Short questions embed poorly. HyDE first asks the LLM to invent a "
            "plausible comment that WOULD answer the question, then searches "
            "with that richer fake comment instead — it usually lands closer "
            "to real matches. Needs Ollama running.")
    hq = st.text_input("Question", "Do people complain about the release date?",
                       key="hyde_query")
    n_hypo = st.slider("Hypothetical comments to generate", 1, 3, 1,
                       help="With more than one, the pooled results are "
                            "reranked against the original question with the "
                            "cross-encoder.")
    if st.button("Run HyDE"):
        with st.spinner("Generating hypothetical comment and retrieving ..."):
            try:
                from src.rag.hyde import hyde_retrieve, hypothetical_answers
                from src.rag.retrieval import semantic_retriever

                hypos = hypothetical_answers(hq, n_hypo)
                for h in hypos:
                    st.info(h)
                c_plain, c_hyde = st.columns(2)
                with c_plain:
                    st.caption("Plain semantic search (raw question)")
                    for d in semantic_retriever(k=5).invoke(hq):
                        st.text(f"• {d.page_content[:80]}")
                with c_hyde:
                    st.caption("HyDE (searched with the hypothetical comment)")
                    for d in hyde_retrieve(hq, k=5, hypotheticals=hypos):
                        st.text(f"• {d.page_content[:80]}")
            except Exception as e:
                st.error(f"HyDE error: {e}")

# ----------------------------------------------------------------------------
with tab_faq:
    st.subheader("What does everything in this app actually do?")
    explain("A plain-English glossary of every method in the dashboard — what "
            "it does, and which tab to see it in. No prior NLP knowledge "
            "needed.")

    st.markdown("#### Text analysis")
    with st.expander("Sentiment analysis (VADER)"):
        st.write("Labels each comment positive, neutral or negative using "
                 "VADER, a rule-based scorer tuned for social media — it "
                 "understands slang, capitalization, punctuation and emojis. "
                 "See it in Sentiment & Emotion and the Overview timeline.")
    with st.expander("Emotion detection (NRC lexicon)"):
        st.write("Goes beyond positive/negative by counting words associated "
                 "with eight basic emotions (joy, trust, anticipation, "
                 "surprise, sadness, fear, anger, disgust) in the NRC emotion "
                 "dictionary. See Sentiment & Emotion.")
    with st.expander("Subjectivity vs polarity (TextBlob)"):
        st.write("Scores each comment on two axes: polarity (negative to "
                 "positive) and subjectivity (factual statement vs pure "
                 "opinion). The scatter plot in Sentiment & Emotion shows "
                 "whether the crowd is stating facts or venting feelings.")
    with st.expander("Language detection (langdetect)"):
        st.write("Guesses each comment's language from character and word "
                 "patterns, showing how international the audience is. See "
                 "the pie chart in Sentiment & Emotion.")
    with st.expander("Emoji frequency"):
        st.write("Counts every emoji across the sample — on YouTube, emojis "
                 "carry as much sentiment as words. See Words & Phrases.")
    with st.expander("Readability & text stats (textstat)"):
        st.write("Average words per comment, a comment-length histogram, and "
                 "the Flesch reading-ease / grade-level scores of the corpus "
                 "as a whole. See the Overview tab.")
    with st.expander("Question share"):
        st.write("The percentage of comments containing a question mark — a "
                 "quick proxy for how much of the audience is asking vs "
                 "stating. Shown as a metric on the Overview tab.")
    with st.expander("TF-IDF distinctive terms"):
        st.write("TF-IDF scores words by how characteristic they are of one "
                 "group of documents versus the rest. Here it contrasts the "
                 "vocabulary of positive vs negative comments — what fans and "
                 "critics say differently. See Words & Phrases.")

    st.markdown("#### Word-level analysis")
    with st.expander("Word cloud & word frequency"):
        st.write("The most frequent meaningful words after removing "
                 "stop-words like 'the' and 'and'; in the cloud, bigger means "
                 "more frequent. See Words & Phrases.")
    with st.expander("PMI collocations"):
        st.write("Finds word pairs and triples that co-occur far more often "
                 "than chance (pointwise mutual information) — set phrases "
                 "like game titles or memes, not just frequent words. See "
                 "Words & Phrases.")
    with st.expander("N-grams"):
        st.write("Simply the most repeated two-word sequences by raw count, "
                 "as a contrast to the chance-corrected PMI ranking next to "
                 "it. See Words & Phrases.")
    with st.expander("Part-of-speech distribution (spaCy)"):
        st.write("Tags every word with its grammar role (noun, verb, "
                 "adjective...) and plots the totals. Lots of adjectives "
                 "means a descriptive, opinionated crowd. See Words & "
                 "Phrases.")

    st.markdown("#### Entities & relations")
    with st.expander("Named-entity recognition (spaCy)"):
        st.write("A trained model spots names of people, companies, products "
                 "and places, keeping only categories relevant to gaming "
                 "discussion. Precomputed during enrichment; see Entities & "
                 "Graph.")
    with st.expander("GLiNER zero-shot domain NER"):
        st.write("Unlike spaCy's fixed categories, GLiNER accepts any label "
                 "strings at inference time ('game feature', 'platform', "
                 "'price') and finds matching spans with no training. Edit "
                 "the labels yourself in Entities & Graph.")
    with st.expander("Keyword extraction (YAKE)"):
        st.write("An unsupervised statistical method that pulls the most "
                 "informative words and phrases from each comment, "
                 "aggregated into a top-keywords chart. See Entities & "
                 "Graph.")
    with st.expander("Noun-Verb-Noun relations"):
        st.write("Scans part-of-speech tags for noun-verb-noun patterns and "
                 "formats them as verb(subject, object) — simple claims like "
                 "'Rockstar delayed game' extracted from raw text. See "
                 "Entities & Graph.")
    with st.expander("Knowledge graph"):
        st.write("Draws the extracted relations as a graph: nouns become "
                 "nodes, verbs become labeled arrows, so recurring claims "
                 "are visible at a glance. See Entities & Graph.")

    st.markdown("#### Topic modeling")
    with st.expander("BERTopic clusters"):
        st.write("Embeds every comment, clusters similar ones, and labels "
                 "each cluster with its characteristic words — grouping "
                 "comments by what they discuss without predefined "
                 "categories. Topic -1 is uncategorized noise and is hidden. "
                 "See Sentiment & Emotion.")

    st.markdown("#### RAG (retrieval-augmented generation)")
    with st.expander("Chunking strategies"):
        st.write("Before indexing, long text is split into chunks: recursive "
                 "(sliding windows on natural breaks), token (fixed token "
                 "windows), or semantic (split where the meaning shifts). "
                 "Compare them in the RAG Pipeline tab, section 1.")
    with st.expander("Retrieval strategies"):
        st.write("Semantic finds meaning-similar comments via embeddings; "
                 "lexical (BM25) matches exact keywords; hybrid fuses both; "
                 "MMR balances relevance with diversity to avoid "
                 "near-duplicate results. Compare them side by side in the "
                 "RAG Pipeline tab, section 2.")
    with st.expander("Cross-encoder reranking"):
        st.write("A second, more accurate model rescores each retrieved "
                 "comment against the question and reorders them, since "
                 "LLM answers are sensitive to context order. See the "
                 "before/after in the RAG Pipeline tab, section 3.")
    with st.expander("HyDE (hypothetical document embeddings)"):
        st.write("The LLM first invents a plausible comment that would "
                 "answer the question, and that richer fake comment is used "
                 "as the search query instead of the short question — often "
                 "retrieving better matches. See the RAG Pipeline tab, "
                 "section 4, or the HyDE mode in Ask the Agent.")

    st.markdown("#### Agents")
    with st.expander("Tool-calling agent (single)"):
        st.write("One LLM bound to a toolbox (search comments, compute "
                 "stats); it decides which tool to call, reads the result, "
                 "and answers — the ReAct loop. Default mode in Ask the "
                 "Agent.")
    with st.expander("Multi-agent router"):
        st.write("A classifier first labels your question by type, then a "
                 "router sends it to a specialist agent for that category — "
                 "deterministic routing that is more reliable on small local "
                 "models than tool-calling. See Ask the Agent.")
    with st.expander("Supervisor (full report)"):
        st.write("A supervisor sequences specialist agents — sentiment, "
                 "topics, entities, summary — each reporting back until a "
                 "complete comment-intelligence report is assembled. See Ask "
                 "the Agent.")
    with st.expander("Swarm (reflection)"):
        st.write("Two peer agents hand control back and forth: a researcher "
                 "drafts a grounded answer and a reviewer critiques it, "
                 "triggering a revision if needed before finalizing. See Ask "
                 "the Agent.")
    with st.expander("DSPy Chain-of-Thought QA"):
        st.write("Instead of a hand-written prompt, DSPy declares a typed "
                 "signature (context + question in, answer + confidence out) "
                 "and ChainOfThought adds an explicit reasoning step you can "
                 "inspect. See Ask the Agent.")

    st.markdown("#### Credits")
    with st.expander("Asma"):
        st.write("Best professor.")
