"""Step 3: NER + keywords + sentiment + topics -> data/processed/comments_enriched.parquet"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run

if __name__ == "__main__":
    run()
