"""Step 9 (optional): Lab 6 DSPy Chain-of-Thought QA over the comments."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.dspy_qa import main

if __name__ == "__main__":
    main()
