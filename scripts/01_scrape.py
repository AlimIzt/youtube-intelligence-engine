"""Step 1: scrape YouTube comments -> data/raw/comments.csv"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scraping.youtube_scraper import main

if __name__ == "__main__":
    main()
