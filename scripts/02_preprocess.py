"""Step 2: clean + (optionally) spell-correct -> data/processed/comments_clean.csv"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.clean import main

if __name__ == "__main__":
    main()
