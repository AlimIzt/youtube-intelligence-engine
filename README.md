# YouTube Intelligence Engine

End-to-end NLP pipeline that turns noisy YouTube comments into structured
insights and answers questions over them with **RAG + LLM agents**, monitored
with **MLflow**. Built on the CSCI370 labs (Ollama + LangGraph + MLflow).

```
YouTube API → clean → enrich (NER, keywords, sentiment, topics) → index → RAG/agent → dashboard
```

---

## Quick setup

**1. Prerequisites**
- Python 3.10+
- [Ollama](https://ollama.com) running, with the models pulled:
  ```bash
  ollama pull llama3.2:3b
  ollama pull nomic-embed-text
  ```

**2. Install**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # macOS/Linux
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**3. Configure**
```bash
copy .env.example .env              # Windows  (cp on macOS/Linux)
```
Add your `YOUTUBE_API_KEY` to `.env` (only needed to scrape new data — sample
comments are already in `data/raw/comments.csv`).

---

## Run

```bash
# pipeline (run once, in order)
python scripts/02_preprocess.py     # clean comments
python scripts/03_enrich.py         # NER, keywords, sentiment, topics
python scripts/04_build_index.py    # build Chroma vector store

# dashboard  → http://localhost:5000
streamlit run app/dashboard.py

# agent (CLI, optional)
python scripts/05_run_agent.py
```

> To scrape fresh comments instead of using the sample: `python scripts/01_scrape.py`.

### NLP methods from the labs

All techniques from Labs 1, 2 and 5 are ported into `src/` and demonstrated by
one script:

```bash
python -m src.utils.nltk_setup        # one-time: download NLTK corpora
python scripts/06_nlp_methods_demo.py # runs every lab method on the comments
```

| Lab | Method | Module |
|-----|--------|--------|
| 1 | NLTK clean (tokenize, stopwords, lemmatize) | `preprocessing/nltk_clean.py` |
| 1 | FreqDist, n-grams, PMI collocations | `analysis/collocations.py` |
| 1/5 | TextBlob sentiment | `analysis/sentiment.py` |
| 2 | POS taggers: Unigram/Bigram/Trigram/Backoff/HMM/spaCy | `analysis/pos.py` |
| 2 | Chunking + Noun-Verb-Noun relations | `analysis/relations.py` |
| 5.1 | VADER / transformer sentiment | `analysis/sentiment.py` |
| 5.1 | TF-IDF + LogisticRegression classifier | `analysis/text_classifier.py` |
| 5.2 | pyABSA aspect-based sentiment (separate env) | `analysis/absa.py` |
| W8L7 | Chunking: recursive / token / semantic | `rag/chunking.py` |
| W8L7 | MMR retrieval (diversity) | `rag/retrieval.py` |
| W8L7 | Cross-encoder reranking | `rag/postretrieval.py` |
| 8 | Multi-agent classifier → router → specialist | `agents/multi_agent.py` |

> pyABSA pins `transformers==4.29`, so run it in its own venv (see the header of
> `src/analysis/absa.py`) — exactly as the lab used `pyabsa_env`.

The **Insights** dashboard tab visualises most of these (sentiment, topics, NER,
keywords, word cloud, collocations, n-grams, POS distribution, and a
Noun-Verb-Noun knowledge graph). The **Ask the Agent** tab lets you switch
between the tool-calling and multi-agent orchestrators, pick semantic vs MMR
retrieval, and toggle cross-encoder reranking.

### Monitoring (MLflow)
```bash
python -m src.evaluation.evaluate   # logs metrics
mlflow ui --port 5001               # → http://localhost:5001
```
> MLflow uses `--port 5001` because the dashboard already owns port 5000.

---

## Layout

```
config.py            central config (paths, models, .env)
data/raw/            scraped comments.csv
data/processed/      cleaned + enriched data
src/scraping/        YouTube collection
src/preprocessing/   cleaning + spell correction
src/analysis/        NER, keywords, sentiment, topics
src/rag/             representation, vectorstore, retrieval, generation
src/agents/          tools + LangGraph orchestrator
src/evaluation/      MLflow evaluation
app/dashboard.py     Streamlit UI (port 5000)
scripts/             numbered pipeline entrypoints
```
