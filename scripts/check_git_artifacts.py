"""Check that generated local artifacts are not tracked by Git.

This script prevents accidentally committing generated files such as processed
datasets, Chroma vector stores, MLflow runs, local environments, and cache files.
Small placeholder files like .gitkeep are allowed.
"""

from __future__ import annotations

import subprocess


BLOCKED_PATHS = [
    "data/processed/",
    "chroma_db/",
    "mlruns/",
    "venv/",
    ".venv/",
    ".env",
    "__pycache__/",
    ".ipynb_checkpoints/",
    ".DS_Store",
]

ALLOWED_TRACKED_FILES = {
    "data/processed/.gitkeep",
}


def get_tracked_files() -> list[str]:
    """Return all files currently tracked by Git."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def is_blocked(path: str) -> bool:
    """Return True if the path is a generated/local artifact."""
    normalized = path.replace("\\", "/")

    if normalized in ALLOWED_TRACKED_FILES:
        return False

    for blocked in BLOCKED_PATHS:
        if blocked.endswith("/"):
            if normalized.startswith(blocked):
                return True
        elif normalized == blocked:
            return True

    return False


def main() -> int:
    """Run the Git artifact check."""
    tracked_files = get_tracked_files()
    blocked_files = [path for path in tracked_files if is_blocked(path)]

    if blocked_files:
        print("Blocked generated/local files are tracked by Git:")
        for path in blocked_files:
            print(f" - {path}")

        print("\nRemove them from tracking, for example:")
        print("git rm -r --cached data/processed chroma_db mlruns")
        return 1

    print("OK: no generated/local artifacts are tracked by Git.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())