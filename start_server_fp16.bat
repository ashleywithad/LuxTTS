@echo off
REM LuxTTS API Server Startup Script (Float16 Mode)
REM Float16 is ~2x faster but uses slightly less precision

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting LuxTTS API Server in FLOAT16 mode...
echo.
echo   Web UI:        http://localhost:8000
echo   API Docs:      http://localhost:8000/docs
echo   Health Check:  http://localhost:8000/health
echo.
if not exist voice_samples\default.wav echo WARNING: No default voice sample found. Please place a voice sample at voice_samples\default.wav
echo.

set DTYPE=float16
python api_server.py

pause
