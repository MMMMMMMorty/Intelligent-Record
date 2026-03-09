@echo off
chcp 65001 >nul
title Intelligent-Record - Stop
cls

echo ========================================
echo   Intelligent-Record - Stop
echo ========================================
echo.

cd /d "%~dp0"

:: Check Docker
docker version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Docker is not running
    echo [OK] Services are already stopped
    pause
    exit /b 0
)

echo [Step 1/4] Stopping ASR engine...
docker ps -q -f name=intelligent-record-asr -f status=running >nul 2>&1
if not errorlevel 1 (
    docker stop intelligent-record-asr >nul 2>&1
    echo [OK] ASR stopped
) else (
    echo [OK] ASR not running
)

echo [Step 2/4] Stopping LLM engine...
docker ps -q -f name=intelligent-record-llm -f status=running >nul 2>&1
if not errorlevel 1 (
    docker stop intelligent-record-llm >nul 2>&1
    echo [OK] LLM stopped
) else (
    echo [OK] LLM not running
)

echo [Step 3/4] Stopping Backend...
docker ps -q -f name=intelligent-record-backend -f status=running >nul 2>&1
if not errorlevel 1 (
    docker stop intelligent-record-backend >nul 2>&1
    echo [OK] Backend stopped
) else (
    echo [OK] Backend not running
)

echo [Step 4/4] Checking network...
docker network ls | findstr "intelligent-record-network" >nul 2>&1
if not errorlevel 1 (
    echo [OK] Network preserved for next start
)

echo.
echo ========================================
echo   All services stopped successfully!
echo ========================================
echo.
echo Note: Containers are preserved for faster restart.
echo.
pause
