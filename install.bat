@echo off
:: Synesthesia — Windows launcher
:: Double-click this file to start the installation.
cd /d "%~dp0"

echo Starting Synesthesia...

:: check Docker is running
docker info > nul 2>&1
if errorlevel 1 (
    echo.
    echo Docker is not running.
    echo Please open Docker Desktop and wait for the whale icon in the taskbar, then double-click this file again.
    pause
    exit /b 1
)

:: load image if not already loaded
docker image inspect synesthesia:latest > nul 2>&1
if errorlevel 1 (
    echo Loading Synesthesia image ^(first run, takes a minute^)...
    docker load < synesthesia.tar.gz
)

:: stop any previous instance
docker rm -f synesthesia > nul 2>&1

:: check .env exists
if not exist ".env" (
    echo.
    echo Missing .env file with GROQ_API_KEY.
    echo Create a .env file next to this script containing:
    echo   GROQ_API_KEY=your_key_here
    pause
    exit /b 1
)

:: run
echo Launching...
docker run -d --name synesthesia -p 5001:5001 --env-file .env synesthesia:latest

:: wait for ready
echo Waiting for server...
:wait_loop
curl -sf http://localhost:5001/health > nul 2>&1
if errorlevel 1 (
    timeout /t 1 /nobreak > nul
    goto wait_loop
)

echo Opening browser...
start http://localhost:5001

echo.
echo Synesthesia is running at http://localhost:5001
echo Close this window to stop.
echo.
pause
docker rm -f synesthesia > nul 2>&1
