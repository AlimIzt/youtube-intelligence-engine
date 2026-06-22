"""Step 8 (optional): Lab 8 swarm agent — researcher/reviewer handoff loop."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.swarm import main

if __name__ == "__main__":
    main()
