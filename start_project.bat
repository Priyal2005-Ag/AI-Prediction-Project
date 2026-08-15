@echo off
title AI Disease Prediction Project

echo ==========================================
echo Starting Ollama...
echo ==========================================
start "Ollama" cmd /k "ollama serve"

timeout /t 5 /nobreak > nul

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
echo CNN        : TensorFlow.js
echo KMeans     : student_kmeans.json
echo Mango RF   : port 5001
echo RAG        : port 5000
echo Web Server : port 5500
echo Interface  : interface.html
echo.
pause