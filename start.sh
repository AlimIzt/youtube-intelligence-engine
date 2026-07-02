#!/usr/bin/env bash
# One-command launcher for macOS / Linux (mirror of start.bat).
# Sets up the venv on first run, starts the Ollama server and the MLflow UI,
# then runs the app (dashboard on http://localhost:5000, MLflow on :5001).
#
# First time only, make it executable:  chmod +x start.sh
# Then run:  ./start.sh
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating virtual environment and installing dependencies (first run)..."
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  python -m spacy download en_core_web_sm
else
  source .venv/bin/activate
fi

export PYTHONUTF8=1

# Start the Ollama server in the background if it isn't already running.
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama server..."
  ollama serve >/dev/null 2>&1 &
fi

# Launch the app; --mlflow also starts the MLflow UI on port 5001.
# run.py waits for Ollama to be ready before pulling models / building the index.
python run.py --mlflow "$@"
