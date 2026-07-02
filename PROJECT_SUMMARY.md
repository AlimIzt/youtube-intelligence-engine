# Project Summary — YouTube Intelligence Engine

A short report of everything in the project.

## What it does
Scrapes YouTube comments (GTA VI trailer, ~19k comments), cleans and enriches
them with NLP, indexes them for Retrieval-Augmented Generation, and answers
questions through an LLM agent — all viewable in a Streamlit dashboard with
MLflow monitoring. Runs fully locally on Ollama (`llama3.2:3b` +
`nomic-embed-text`).

## How to run
- One click (Windows): double-click **`start.bat`** (sets up venv on first run).
- One command: `python run.py` → prepares data + index, then opens the dashboard on
  **http://localhost:5002**. Add `--mlflow` for the MLflow UI on `:5001`.
- On macOS, use `python3` if `python` is not available:
  `python3 run.py`.

The dashboard uses port **5002** because macOS can reserve port 5000 for
AirPlay Receiver, which may conflict with Streamlit.

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
- Also in the pipeline: spaCy **NER**, **GLiNER** domain NER, **KeyBERT/YAKE**
  keywords, and **BERTopic** topic modeling
  (`analysis/ner.py`, `analysis/gliner_ner.py`, `analysis/keywords.py`,
  `analysis/topics.py`)

## RAG & agents
- **Representation/index** — Ollama embeddings + Chroma (`rag/representation.py`,
  `rag/vectorstore.py`)
- **Retrieval** — semantic, lexical (BM25), hybrid (ensemble), metadata-filtered,
  and **MMR** for diversity (`rag/retrieval.py`) — W8L7
- **Chunking** — recursive / token / semantic (`rag/chunking.py`) — W8L7
- **HyDE** — hypothetical document embeddings for richer retrieval queries
  (`rag/hyde.py`) — W8L7
- **Post-retrieval** — cross-encoder reranking (`rag/postretrieval.py`) — W8L7
- **Generation** — grounded QA + summarization (`rag/generation.py`); plus a
  **DSPy** Chain-of-Thought module with self-rated confidence (`rag/dspy_qa.py`) — Lab 6
- **Agents (all four Lab 8 patterns)** — tool-calling (`agents/orchestrator.py`),
  router (`agents/multi_agent.py`), supervisor that sequences specialists into a
  full report (`agents/supervisor.py`), and swarm researcher↔reviewer handoff
  (`agents/swarm.py`)

## Dashboard (app/dashboard.py, port 5002)
- **Overview** — counts, % positive, sample table, sentiment over time, comment length
- **Comments** — searchable/sample comment table
- **Sentiment & Emotion** — sentiment, emotion, subjectivity, language detection,
  topic clusters, and **topics by sentiment**
- **Words & Phrases** — word cloud, frequency, PMI collocations, n-grams, POS distribution
- **Entities & Graph** — spaCy NER, GLiNER custom entities, keywords, and
  Noun-Verb-Noun knowledge graph
- **Ask the Agent** — RAG Q&A with selectable agent patterns, retrieval options,
  HyDE mode, MMR retrieval, and optional cross-encoder reranking
- **RAG Pipeline** — explains chunking, retrieval, reranking, HyDE, and grounded generation
- **FAQ** — plain-English glossary and setup/reliability notes

## Reliability fixes added
- **Dashboard port fix:** Streamlit runs on port `5002` to avoid macOS port `5000`
  conflicts.
- **BERTopic fallback:** if BERTopic fails on macOS because of native dependency
  issues around UMAP/HDBSCAN/numba/OpenMP, the pipeline continues with
  `topic = -1` instead of crashing.
- **macOS documentation:** README includes setup steps, Ollama notes, port notes,
  and BERTopic troubleshooting.
- **Git artifact safety check:** `scripts/check_git_artifacts.py` checks that
  generated/local artifacts such as `data/processed/`, `chroma_db/`, `mlruns/`,
  and virtual environments are not accidentally committed.
- **Unit tests:** `tests/test_relations.py` tests Lab 2 Noun-Verb-Noun relation
  extraction, relation triples, deduplication, noun chunks, and knowledge graph DOT output.
- **Docstrings:** added clearer documentation to HyDE, GLiNER, supervisor, and swarm modules.

## Evaluation & monitoring
- `src/evaluation/evaluate.py` — retrieval hit-rate/MRR, generation latency and
  groundedness, all logged to **MLflow** (`mlflow ui --port 5001`).
- `tests/test_relations.py` — pytest unit tests for relation extraction.

## Notes
- API key is read from `.env` and is not hardcoded.
- `.env`, virtual environments, processed data, Chroma DB, MLflow runs, and cache
  files are ignored by Git.
- pyABSA needs its own venv because it pins `transformers==4.29`.
- On macOS, use `python3` instead of `python` if the command is not available.
- We really love Professor Farhad and Professor Asma.