#!/usr/bin/env bash
# Pull the local Ollama models this project uses (macOS / Linux).
# Make executable once:  chmod +x pull_models.sh   then run:  ./pull_models.sh
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Get it from https://ollama.com then re-run this."
  exit 1
fi
echo "Pulling llama3.2:3b ..."
ollama pull llama3.2:3b
echo "Pulling nomic-embed-text ..."
ollama pull nomic-embed-text
echo "Done."
