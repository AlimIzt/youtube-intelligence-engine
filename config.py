"""Central configuration. Import `from config import settings`."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CHROMA_DIR = ROOT / "chroma_db"

for _d in (RAW, PROCESSED):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    # --- scraping ---
    youtube_api_key: str = os.getenv("YOUTUBE_API_KEY", "")
    video_ids: tuple[str, ...] = ("Dw_oH5oiUSE",)  # GTA VI trailer (default)
    scrape_target: int = 20000

    # --- files ---
    raw_csv: Path = RAW / "comments.csv"
    clean_csv: Path = PROCESSED / "comments_clean.csv"
    enriched_parquet: Path = PROCESSED / "comments_enriched.parquet"

    # --- models (Ollama, local) ---
    llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b")
    embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    spacy_model: str = "en_core_web_sm"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    # --- rag ---
    chroma_dir: Path = CHROMA_DIR
    collection_name: str = "youtube_comments"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 6

    # --- monitoring ---
    mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow_experiment: str = "youtube-intelligence-engine"


settings = Settings()