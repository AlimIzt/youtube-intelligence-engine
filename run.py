"""One-command launcher for the YouTube Intelligence Engine.

Prepares everything the app needs, then starts the dashboard:
  1. checks Ollama is running and pulls the required models if missing
  2. downloads NLTK corpora
  3. runs the pipeline (preprocess -> enrich -> build index) if data is missing
  4. (optional) starts the MLflow UI on http://localhost:5001
  5. starts the Streamlit dashboard on http://localhost:5000

Usage:
  python run.py                 # start the app (build data only if missing)
  python run.py --rebuild       # force re-run the whole pipeline
  python run.py --mlflow        # also launch the MLflow UI
  python run.py --no-dashboard  # just prepare data, don't start the UI
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request

# make prints safe on non-UTF-8 consoles (e.g. Windows cp1251)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import settings

REQUIRED_MODELS = (settings.llm_model, settings.embed_model)


def step(msg: str) -> None:
    print(f"\n\033[1m==> {msg}\033[0m", flush=True)


# ---------------------------------------------------------------- Ollama
def check_ollama() -> None:
    step("Checking Ollama")
    import time

    installed = None
    for attempt in range(15):  # wait up to ~30s for a just-started server
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
                installed = {m["name"] for m in __import__("json").load(r).get("models", [])}
            break
        except Exception:
            if attempt == 0:
                print("  Waiting for Ollama to start ...")
            time.sleep(2)
    if installed is None:
        print("  ! Ollama is not reachable on :11434. Start it with `ollama serve`.")
        return
    print(f"  Ollama is up ({len(installed)} models installed).")
    for model in REQUIRED_MODELS:
        if model not in installed and f"{model}:latest" not in installed:
            print(f"  Pulling missing model: {model}")
            subprocess.run(["ollama", "pull", model], check=False)


# ---------------------------------------------------------------- data
def ensure_nltk() -> None:
    step("Ensuring NLTK corpora")
    from src.utils.nltk_setup import ensure_nltk as _e

    _e()
    print("  NLTK ready.")


def ensure_pipeline(rebuild: bool) -> None:
    if rebuild or not settings.clean_csv.exists():
        step("Preprocessing comments")
        from src.preprocessing.clean import main as preprocess

        preprocess()
    if rebuild or not settings.enriched_parquet.exists():
        step("Enriching (NER, keywords, sentiment, topics)")
        from src.pipeline import run as enrich

        enrich()
    if rebuild or not settings.chroma_dir.exists():
        step("Building vector index")
        import pandas as pd

        from src.rag.representation import df_to_documents
        from src.rag.vectorstore import build_vectorstore

        build_vectorstore(df_to_documents(pd.read_parquet(settings.enriched_parquet)))
    print("\n  Data + index are ready.")


# ---------------------------------------------------------------- services
def start_mlflow():
    step("Starting MLflow UI -> http://localhost:5001")
    return subprocess.Popen(
        [sys.executable, "-m", "mlflow", "ui",
         "--backend-store-uri", settings.mlflow_uri, "--port", "5001"]
    )


def start_dashboard() -> None:
    step("Starting dashboard -> http://localhost:5000  (Ctrl+C to stop)")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/dashboard.py"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch the YouTube Intelligence Engine.")
    ap.add_argument("--rebuild", action="store_true", help="force re-run the pipeline")
    ap.add_argument("--mlflow", action="store_true", help="also start the MLflow UI")
    ap.add_argument("--no-dashboard", action="store_true", help="prepare data only")
    args = ap.parse_args()

    check_ollama()
    ensure_nltk()
    ensure_pipeline(args.rebuild)

    mlflow_proc = start_mlflow() if args.mlflow else None
    try:
        if not args.no_dashboard:
            start_dashboard()
    finally:
        if mlflow_proc:
            mlflow_proc.terminate()


if __name__ == "__main__":
    main()
