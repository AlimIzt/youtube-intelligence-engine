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
