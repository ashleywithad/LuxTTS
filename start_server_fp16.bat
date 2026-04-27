@echo off
REM LuxTTS API Server Startup Script (Float16 Mode)
REM Float16 halves VRAM usage but may be SLOWER on consumer GPUs (RTX 3050/3060)
REM Only use if VRAM is constrained (4GB cards). Float32 is faster otherwise.

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting LuxTTS API Server in FLOAT16 mode...
echo NOTE: Float16 reduces VRAM but may be slower on consumer GPUs.
echo       Use this only if you are running out of VRAM.
echo.
echo   Web UI:        http://localhost:8000
echo   API Docs:      http://localhost:8000/docs
echo   Health Check:  http://localhost:8000/health
echo.
dir /b voice_samples\*.wav voice_samples\*.mp3 >nul 2>&1
if errorlevel 1 echo WARNING: No voice samples found in voice_samples\ Please add .wav or .mp3 files.
echo.

set ENABLE_FP16=true
set ENABLE_TF32=true
set PYTHONWARNINGS=ignore::UserWarning
set PYTHONIOENCODING=utf-8
python api_server.py

pause