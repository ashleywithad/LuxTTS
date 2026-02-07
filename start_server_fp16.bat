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
dir /b voice_samples\*.wav voice_samples\*.mp3 >nul 2>&1
if errorlevel 1 echo WARNING: No voice samples found in voice_samples\ Please add .wav or .mp3 files.
echo.

set DTYPE=float16
set PYTHONWARNINGS=ignore::UserWarning
set PYTHONIOENCODING=utf-8
python api_server.py

pause
