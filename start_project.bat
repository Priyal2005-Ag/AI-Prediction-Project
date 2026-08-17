@echo off
title AI Disease Prediction Project

echo ==========================================
echo Checking Ollama...
echo ==========================================

curl.exe -s http://127.0.0.1:11434/api/tags > nul 2>&1

if %errorlevel% neq 0 (
    echo Ollama is not running. Starting Ollama...
    start "Ollama" cmd /k "ollama serve"
    timeout /t 8 /nobreak > nul
) else (
    echo Ollama is already running.
)

echo ==========================================
echo Starting RAG API on port 5000...
echo ==========================================

start "RAG API" cmd /k "cd /d C:\Users\Lenovo\OneDrive\AI Prediction project\RAG && python app_api.py"

timeout /t 8 /nobreak > nul

echo ==========================================
echo Starting Mango Random Forest on port 5001...
echo ==========================================

start "Mango RF" cmd /k "cd /d C:\Users\Lenovo\OneDrive\AI Prediction project\fruit disease classification && python app.py"

timeout /t 5 /nobreak > nul

echo ==========================================
echo Starting Web Server on port 5500...
echo ==========================================

start "Web Server" cmd /k "cd /d C:\Users\Lenovo\OneDrive\AI Prediction project && python -m http.server 5500"

timeout /t 3 /nobreak > nul

echo ==========================================
echo Opening Interface...
echo ==========================================

start "" "http://127.0.0.1:5500/interface.html"

echo.
echo ==========================================
echo PROJECT STARTED SUCCESSFULLY
echo ==========================================
echo.
echo Ollama    : 127.0.0.1:11434
echo RAG API   : 127.0.0.1:5000
echo Mango RF  : 127.0.0.1:5001
echo Web       : 127.0.0.1:5500
echo Interface : interface.html
echo.
pause