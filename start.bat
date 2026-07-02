@echo off
REM One-click launcher for Windows. Sets up the venv on first run, starts the
REM Ollama server and the MLflow UI, then runs the app (data pipeline if needed
REM + dashboard on http://localhost:5000, MLflow on http://localhost:5001).
cd /d %~dp0

if not exist .venv (
    echo Creating virtual environment and installing dependencies ^(first run^)...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m spacy download en_core_web_sm
) else (
    call .venv\Scripts\activate.bat
)

set PYTHONUTF8=1

REM Start the Ollama server in the background (harmless error if already running).
echo Starting Ollama server...
start "Ollama" /min ollama serve

REM Launch the app; --mlflow also starts the MLflow UI on port 5001.
REM run.py waits for Ollama to be ready before pulling models / building the index.
python run.py --mlflow %*
pause
