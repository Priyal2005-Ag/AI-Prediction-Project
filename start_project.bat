@echo off
title AI Disease Prediction System
color 0A

echo.
echo ================================================
echo       AI DISEASE PREDICTION SYSTEM
echo ================================================
echo.

REM =================================================
REM 1. START OLLAMA
REM =================================================

echo [1/4] Checking Ollama...

curl.exe -s http://127.0.0.1:11434/api/tags >nul 2>&1

if %errorlevel%==0 (
    echo Ollama is already running.
) else (
    echo Ollama is not running.
    echo Starting Ollama...

    REM Try to start Ollama Windows application
    if exist "%LOCALAPPDATA%\Programs\Ollama\Ollama.exe" (
        start "" "%LOCALAPPDATA%\Programs\Ollama\Ollama.exe"
    ) else (
        if exist "C:\Program Files\Ollama\Ollama.exe" (
            start "" "C:\Program Files\Ollama\Ollama.exe"
        ) else (
            echo Ollama.exe not found.
            echo Trying ollama serve...
            start "Ollama Server" cmd /k "ollama serve"
        )
    )

    echo Waiting for Ollama...
    
    set OLLAMA_READY=0

    for /L %%i in (1,1,20) do (
        curl.exe -s http://127.0.0.1:11434/api/tags >nul 2>&1

        if not errorlevel 1 (
            set OLLAMA_READY=1
            goto OLLAMA_STARTED
        )

        timeout /t 1 /nobreak >nul
    )

    :OLLAMA_STARTED

    if "%OLLAMA_READY%"=="0" (
        echo.
        echo ERROR: Ollama could not be started.
        echo Please make sure Ollama is installed correctly.
        pause
        exit /b 1
    )

    echo Ollama started successfully.
)

echo.

REM =================================================
REM 2. START RAG API
REM =================================================

echo [2/4] Starting RAG API...

start "RAG API" cmd /k "cd /d C:\Users\Lenovo\OneDrive\AI Prediction project\RAG && python app_api.py"

echo Waiting for RAG API...
timeout /t 10 /nobreak >nul

echo RAG API started.
echo.

REM =================================================
REM 3. START MANGO RANDOM FOREST API
REM =================================================

echo [3/4] Starting Mango Random Forest API...

start "Mango RF" cmd /k "cd /d C:\Users\Lenovo\OneDrive\AI Prediction project\fruit disease classification && python app.py"

echo Waiting for Mango RF API...
timeout /t 5 /nobreak >nul

echo Mango RF API started.
echo.

REM =================================================
REM 4. START WEB SERVER
REM =================================================

echo [4/4] Starting Web Server...

start "Web Server" cmd /k "cd /d C:\Users\Lenovo\OneDrive\AI Prediction project && python -m http.server 5500"

echo Waiting for Web Server...
timeout /t 3 /nobreak >nul

echo Web Server started.
echo.

REM =================================================
REM OPEN HTML INTERFACE
REM =================================================

echo ================================================
echo       OPENING AI INTERFACE
echo ================================================
echo.

start "" "http://127.0.0.1:5500/interface.html"

echo.
echo ================================================
echo       PROJECT STARTED SUCCESSFULLY
echo ================================================
echo.
echo Ollama     : http://127.0.0.1:11434
echo RAG API    : http://127.0.0.1:5000
echo Mango API  : http://127.0.0.1:5001
echo Web Server : http://127.0.0.1:5500
echo Interface  : http://127.0.0.1:5500/interface.html
echo.
echo ================================================
echo.
echo You can now use the AI Disease Prediction System.
echo.
pause