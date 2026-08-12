@echo off
setlocal
title ArmInferX Launcher
cd /d "%~dp0"

echo ============================================================
echo   ArmInferX - Local Demo Launcher (no Docker needed)
echo ============================================================
echo.
REM --- Preflight: backend venv ---
if not exist "backend\.venv\Scripts\python.exe" (
    echo [ERROR] Backend venv missing: backend\.venv\Scripts\python.exe
    echo         Create it first:
    echo           cd backend
    echo           python -m venv .venv
    echo           .venv\Scripts\pip install -r requirements.txt
    goto :done
)

REM --- Preflight: frontend deps ---
if not exist "frontend\node_modules" (
    echo [ERROR] frontend\node_modules missing.
    echo         Install first:
    echo           cd frontend
    echo           npm install
    goto :done
)

echo [1/3] Starting backend  http://localhost:8000  (FastAPI + llama.cpp) ...
start "ArmInferX Backend" cmd /k "cd /d %~dp0backend && .venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000"

echo       Waiting for /health (model is NOT loaded at startup - lazy load)...
set /a tries=0
:wait_health
set /a tries+=1
REM Health gate. TimeoutSec must be generous: each probe spawns a fresh
REM powershell.exe (cold start ~2-4s on this machine), and Invoke-WebRequest
REM with -TimeoutSec 2 consistently times out -> the gate would never pass
REM and the frontend would only start via the 60-try WARN fallback (~2 min).
REM With -TimeoutSec 8 the probe returns within ~1s once uvicorn is up.
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:8000/health' -UseBasicParsing -TimeoutSec 8; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %errorlevel%==0 goto backend_up
if %tries% geq 60 (
    echo       [WARN] Backend not healthy after the retry limit - check the "ArmInferX Backend" window.
    goto frontend
)
REM Sleep ~1s. ping -n 2 127.0.0.1 works even when stdin is redirected
REM (timeout.exe refuses to run with redirected input).
ping -n 2 127.0.0.1 >nul
goto wait_health

:backend_up
echo       Backend is UP.
echo.

:frontend
echo [2/3] Starting frontend  http://localhost:3000  (Vite dev server) ...
start "ArmInferX Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"
REM Sleep ~5s for the dev server to bind before opening the browser.
ping -n 6 127.0.0.1 >nul

echo [3/3] Opening your browser...
start "" http://localhost:3000

echo.
echo ============================================================
echo   Demo is live. Keep the two new windows open; close them to
echo   stop the servers. Demo script: docs\final-demo-checklist.md
echo ============================================================

:done
if /i not "%1"=="--no-pause" pause
endlocal
