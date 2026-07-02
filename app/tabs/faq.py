"""Plain-English glossary of every method in the app."""
from __future__ import annotations

import streamlit as st

from app.data import Scope
from app.theme import explain


def render(scope: Scope) -> None:
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

    with st.expander("Topic fallback"):
        st.write("On some machines, BERTopic can fail because it depends on "
                 "native libraries such as UMAP, HDBSCAN and numba. The "
                 "pipeline first tries normal topic modeling. If it fails, "
                 "it continues with topic = -1, meaning no topic was assigned. "
                 "This keeps cleaning, sentiment, NER, keywords, indexing, "
                 "RAG and the dashboard working instead of crashing.")

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

    st.markdown("#### Setup & reliability")
    with st.expander("Why does the dashboard use port 5002 on macOS?"):
        st.write("macOS can use port 5000 for system services such as AirPlay "
                 "Receiver, which prevents Streamlit from starting there. To "
                 "avoid this conflict, the dashboard is configured to run on "
                 "port 5002. Open http://localhost:5002 after running the app.")

    with st.expander("Why do we check Git artifacts?"):
        st.write("Processed datasets, Chroma vector stores, MLflow runs and "
                 "virtual environments are generated locally and can become "
                 "large or machine-specific. The Git artifact check helps "
                 "make sure folders such as data/processed/, chroma_db/ and "
                 "mlruns/ are not accidentally committed.")

    with st.expander("Why should API keys stay in .env?"):
        st.write("API keys are private credentials. Keeping them in .env "
                 "allows the project to read them locally without hardcoding "
                 "or committing them to GitHub. This keeps the repository safe "
                 "to share with teammates and instructors.")

    st.markdown("#### Credits")
    with st.expander("Asma"):
        st.write("Best professor.")