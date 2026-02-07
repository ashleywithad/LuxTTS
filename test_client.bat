@echo off
REM LuxTTS API Test Client
REM This script tests the API server

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Testing LuxTTS API...
echo Make sure the server is running (run start_server.bat first)
echo.

REM Run the test client
python client_example.py

pause
