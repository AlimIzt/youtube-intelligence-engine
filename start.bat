@echo off
REM One-click launcher for Windows. Sets up the venv on first run, then starts
REM the app (data pipeline if needed + dashboard on http://localhost:5000).
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
python run.py %*
pause
