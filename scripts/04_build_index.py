"""Step 4: build the Chroma vector store from enriched comments."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import settings
from src.rag.representation import df_to_documents
from src.rag.vectorstore import build_vectorstore

if __name__ == "__main__":
    df = pd.read_parquet(settings.enriched_parquet)
    build_vectorstore(df_to_documents(df), reset=True)
