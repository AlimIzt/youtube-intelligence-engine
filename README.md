# YouTube Intelligence Engine

End-to-end NLP pipeline that turns noisy YouTube comments into structured
insights and answers questions over them with **RAG + LLM agents**, monitored
with **MLflow**. Built with love for professors Farhad and especially Asma, Ollama, LangGraph, and MLflow.

```
YouTube API → clean → enrich (NER, keywords, sentiment, topics) → index → RAG/agent → dashboard
```

---

## Quick setup

**1. Prerequisites**
- **Required:** Python 3.10/3.11
- **Optional** — only for the *Ask the Agent* / RAG tabs (LLM features).
  The analytics tabs work without it. [Ollama](https://ollama.com) running,
  with the models pulled:
  ```bash
  ollama pull llama3.2:3b
  ollama pull nomic-embed-text
  ```
- Or just run `start.bat` on Windows.

**2. Install** *(required)*

Windows PowerShell:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**3. Configure** — **OPTIONAL, skip this.**
> The project runs on the bundled `data/raw/comments.csv` with sensible
> defaults, so no `.env` is needed. Only do this if you want to scrape a
> *different* video or use non-default Ollama models:
> ```bash
> copy .env.example .env            # Windows
> cp .env.example .env              # macOS/Linux
> ```
> Then add your `YOUTUBE_API_KEY` to `.env`.

---

## Run

### Recommended one-command launcher

```bash
python run.py
```

This prepares the data and vector index if needed, then starts the dashboard:

```
http://localhost:5002
```

On macOS, the dashboard uses port `5002` because port `5000` is often reserved
by AirPlay Receiver.

To also start MLflow:

```bash
python run.py --mlflow
```

MLflow will run on:

```
http://localhost:5001
```

### Manual pipeline run

```bash
# pipeline (run once, in order) — REQUIRED before first launch
python scripts/02_preprocess.py     # clean comments
python scripts/03_enrich.py         # NER, keywords, sentiment, topics
python scripts/04_build_index.py    # build Chroma vector store (needs Ollama)

# dashboard  → http://localhost:5002   (main entry point)
streamlit run app/dashboard.py --server.port 5002
```

**Optional extras:**
```bash
python scripts/05_run_agent.py      # optional: chat with the agent from the CLI
python scripts/01_scrape.py         # optional: scrape fresh comments (needs API key)
```

> Note: `04_build_index.py` and the *Ask the Agent* tab need Ollama running.
> The Overview / Insights / Words / Entities / Comments tabs do not.

### macOS notes

If `python` is not found on macOS, use:

```bash
python3
```

For example:

```bash
python3 scripts/02_preprocess.py
python3 scripts/03_enrich.py
python3 scripts/04_build_index.py
```

If Ollama is installed but not running, start it with:

```bash
ollama serve
```

If the dashboard does not open on `5002`, check whether another process is using
the port or change the Streamlit port in:

```
.streamlit/config.toml
```

### BERTopic note on macOS

BERTopic can fail on some macOS machines during the topic modeling stage,
especially around the UMAP / HDBSCAN dimensionality-reduction and clustering
step. This is usually not caused by the project code itself, but by native
Python dependencies that need local binary support, such as `hdbscan`, `numba`,
OpenMP, or compiler tools.

If topic modeling crashes with a segmentation fault or native dependency error,
try the following fixes:

```bash
xcode-select --install
python -m pip install --upgrade pip setuptools wheel
pip install --force-reinstall hdbscan umap-learn numba
```

If `python` is not available on macOS, use `python3` instead:

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --force-reinstall hdbscan umap-learn numba
```

The pipeline also includes a safe fallback in `src/pipeline.py`: it first tries
to run normal BERTopic topic modeling through `add_topics(df)`. If that fails,
it continues with `topic = -1`, meaning no topic was assigned. This fallback
keeps cleaning, NER, keywords, sentiment, vector indexing, RAG, and the dashboard
working instead of crashing.

---

### NLP methods *(optional demo)*

All NLP techniques are implemented in `src/` and demonstrated by one script:

```bash
python -m src.utils.nltk_setup        # one-time: download NLTK corpora
python scripts/06_nlp_methods_demo.py # runs every NLP method on the comments
```

| Method | Module |
|--------|--------|
| NLTK clean (tokenize, stopwords, lemmatize) | `preprocessing/nltk_clean.py` |
| FreqDist, n-grams, PMI collocations | `analysis/collocations.py` |
| TextBlob sentiment | `analysis/sentiment.py` |
| POS taggers: Unigram/Bigram/Trigram/Backoff/HMM/spaCy | `analysis/pos.py` |
| Chunking + Noun-Verb-Noun relations | `analysis/relations.py` |
| VADER / transformer sentiment | `analysis/sentiment.py` |
| TF-IDF + LogisticRegression classifier | `analysis/text_classifier.py` |
| pyABSA aspect-based sentiment (separate env) | `analysis/absa.py` |
| Chunking: recursive / token / semantic | `rag/chunking.py` |
| MMR retrieval (diversity) | `rag/retrieval.py` |
| Cross-encoder reranking | `rag/postretrieval.py` |
| DSPy Chain-of-Thought QA (signature + module) | `rag/dspy_qa.py` |
| Tool-calling agent | `agents/orchestrator.py` |
| Router: classifier → specialist | `agents/multi_agent.py` |
| Supervisor: sequenced specialists → report | `agents/supervisor.py` |
| Swarm: researcher ↔ reviewer handoff | `agents/swarm.py` |

> pyABSA pins `transformers==4.29`, so run it in its own venv (see the header of
> `src/analysis/absa.py`), using a dedicated `pyabsa_env`.

The **Insights** dashboard tab visualises most of these (sentiment, topics, NER,
keywords, word cloud, collocations, n-grams, POS distribution, and a
Noun-Verb-Noun knowledge graph). The **Ask the Agent** tab lets you switch
between the four agent orchestrators (tool-calling, router, supervisor, swarm)
plus the DSPy Chain-of-Thought module, pick semantic vs MMR retrieval, and
toggle cross-encoder reranking.

### Monitoring (MLflow) *(optional)*

Only needed if you want evaluation metrics and tracing; not required to run the app.

```bash
python -m src.evaluation.evaluate   # logs metrics
mlflow ui --port 5001               # → http://localhost:5001
```

> The dashboard uses port `5002` because macOS can reserve port `5000` for
> AirPlay Receiver by default, which would otherwise conflict.

---

## Layout

```
config.py            central config (paths, models, .env)
data/raw/            scraped comments.csv
data/processed/      cleaned + enriched data
src/scraping/        YouTube collection
src/preprocessing/   cleaning + spell correction
src/analysis/        NER, GLiNER, keywords, sentiment, emotions, topics
src/rag/             representation, vectorstore, retrieval, HyDE, generation
src/agents/          tools + LangGraph orchestrator/router/supervisor/swarm
src/evaluation/      MLflow evaluation
app/dashboard.py     Streamlit entry point (port 5002)
app/theme.py         global CSS, animated hero, explain/takeaway/style helpers
app/data.py          data loading, sidebar sample scope, cached analytics
app/tabs/            one module per dashboard tab (render(scope) each)
scripts/             numbered pipeline entrypoints
```

Adding a dashboard tab: create `app/tabs/<name>.py` with a
`render(scope: Scope)` function and append it to `TABS` in
`app/tabs/__init__.py`. Everything computed over comments should go through a
cached `(n, strategy)` function in `app/data.py` so it respects the sidebar sample scope.

---

We really love Professor Farhad and Professor Asma.