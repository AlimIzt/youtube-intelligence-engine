@echo off
REM Pull the local Ollama models this project uses (Windows).
where ollama >nul 2>nul
if errorlevel 1 (
    echo Ollama is not installed. Get it from https://ollama.com then re-run this.
    pause
    exit /b 1
)
echo Pulling llama3.2:3b ...
ollama pull llama3.2:3b
echo Pulling nomic-embed-text ...
ollama pull nomic-embed-text
echo Done.
pause
