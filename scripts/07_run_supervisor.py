"""Step 7 (optional): Lab 8 supervisor agent — builds a full intelligence report."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.supervisor import main

if __name__ == "__main__":
    main()
