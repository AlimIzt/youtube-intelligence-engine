# Project Summary — YouTube Intelligence Engine

A short report of everything in the project.

## What it does
Scrapes YouTube comments (GTA VI trailer, ~19k comments), cleans and enriches
them with NLP, indexes them for Retrieval-Augmented Generation, and answers
questions through an LLM agent — all viewable in a dashboard with MLflow
monitoring. Runs fully locally on Ollama (`llama3.2:3b` + `nomic-embed-text`).

## How to run
- One click (Windows): double-click **`start.bat`** (sets up venv on first run).
- Or: `python run.py` → prepares data + index, then opens the dashboard on
  **http://localhost:5000**. Add `--mlflow` for the MLflow UI on `:5001`.

## Pipeline (scripts/)
| Step | Script | Output |
|------|--------|--------|
| Scrape | `01_scrape.py` | `data/raw/comments.csv` |
| Clean | `02_preprocess.py` | `comments_clean.csv` (19k from 20k raw) |
| Enrich | `03_enrich.py` | `comments_enriched.parquet` |
| Index | `04_build_index.py` | Chroma vector store (18,962 vectors) |
| Agent | `05_run_agent.py` | interactive CLI |
| Lab demo | `06_nlp_methods_demo.py` | runs every lab NLP method |

## NLP methods (ported from the labs)
- **Lab 1** — NLTK cleaning, FreqDist, n-grams, PMI collocations, TextBlob
  (`preprocessing/nltk_clean.py`, `analysis/collocations.py`)
- **Lab 2** — POS tagger comparison (Unigram/Bigram/Trigram/Backoff/HMM/spaCy),
  chunking, Noun-Verb-Noun relations + knowledge graph
  (`analysis/pos.py`, `analysis/relations.py`)
- **Lab 5.1** — VADER / TextBlob / transformer sentiment, TF-IDF + Logistic
  Regression classifier (`analysis/sentiment.py`, `analysis/text_classifier.py`)
- **Lab 5.2** — pyABSA aspect-based sentiment + aspect word clouds, runs in a
  separate env (`analysis/absa.py`)
- Also in the pipeline: spaCy **NER**, **KeyBERT/YAKE** keywords, **BERTopic**
  topic modeling (`analysis/ner.py`, `analysis/keywords.py`, `analysis/topics.py`)

## RAG & agents
- **Representation/index** — Ollama embeddings + Chroma (`rag/representation.py`,
  `rag/vectorstore.py`)
- **Retrieval** — semantic, lexical (BM25), hybrid (ensemble), metadata-filtered,
  and **MMR** for diversity (`rag/retrieval.py`) — W8L7
- **Chunking** — recursive / token / semantic (`rag/chunking.py`) — W8L7
- **Post-retrieval** — cross-encoder reranking (`rag/postretrieval.py`) — W8L7
- **Generation** — grounded QA + summarization (`rag/generation.py`); plus a
  **DSPy** Chain-of-Thought module with self-rated confidence (`rag/dspy_qa.py`) — Lab 6
- **Agents (all four Lab 8 patterns)** — tool-calling (`agents/orchestrator.py`),
  router (`agents/multi_agent.py`), supervisor that sequences specialists into a
  full report (`agents/supervisor.py`), and swarm researcher↔reviewer handoff
  (`agents/swarm.py`)

## Dashboard (app/dashboard.py, port 5000)
- **Overview** — counts, % positive, sample table
- **Insights** — sentiment, topics, NER, keywords, word cloud + frequency,
  PMI collocations, n-grams, POS distribution, Noun-Verb-Noun knowledge graph
- **Ask the Agent** — RAG Q&A with selectable agent (tool-calling / multi-agent),
  retrieval (semantic / MMR), and optional cross-encoder reranking

## Evaluation & monitoring
- `src/evaluation/evaluate.py` — retrieval hit-rate/MRR, generation latency and
  groundedness, all logged to **MLflow** (`mlflow ui --port 5001`).

## Notes
- API key is read from `.env` (not hardcoded).
- pyABSA needs its own venv (it pins `transformers==4.29`).
