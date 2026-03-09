@echo off
chcp 65001 >nul
title Intelligent-Record
cls

echo ========================================
echo   Intelligent-Record
echo   Qwen3-ASR-1.7B + Qwen3-1.7B
echo ========================================
echo.

cd /d "%~dp0"

:: Check Docker
docker version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running!
    echo.
    echo How to fix:
    echo   1. Open Docker Desktop from Start Menu
    echo   2. Wait for the green icon to appear
    echo   3. Run this script again
    echo.
    pause
    exit /b 1
)

:: Check Models
if not exist "models\Qwen3-ASR-1.7B\config.json" (
    echo [ERROR] ASR model not found!
    echo   Expected: models\Qwen3-ASR-1.7B\config.json
    echo   Download: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
    pause
    exit /b 1
)
if not exist "models\Qwen3-1.7B\config.json" (
    echo [ERROR] LLM model not found!
    echo   Expected: models\Qwen3-1.7B\config.json
    echo   Download: https://huggingface.co/Qwen/Qwen3-1.7B
    pause
    exit /b 1
)

:: Check Port
netstat -ano | findstr ":8080" | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo [ERROR] Port 8080 is already in use!
    pause
    exit /b 1
)

echo [INFO] Starting backend...

:: Remove old backend container to ensure fresh start
echo [INFO] Removing old backend container...
docker rm -f intelligent-record-backend >nul 2>&1

:: Get current directory for volume mounting
set "CURRENT_DIR=%CD%"
set "CURRENT_DIR=%CURRENT_DIR:\=/%"

:: Create data directory if not exists
if not exist "backend\data" mkdir "backend\data"

:: Start backend with code mounted for hot-reload during testing
echo [INFO] Creating new container with code mounted...
docker run -d ^
    --name intelligent-record-backend ^
    --network intelligent-record-network ^
    -p 8080:8080 ^
    -v "//var/run/docker.sock:/var/run/docker.sock" ^
    -v "%CURRENT_DIR%/models:/models:ro" ^
    -v "%CURRENT_DIR%/backend/engine_manager.py:/app/engine_manager.py:ro" ^
    -v "%CURRENT_DIR%/backend/main.py:/app/main.py:ro" ^
    -v "%CURRENT_DIR%/backend/services:/app/services:ro" ^
    -v "%CURRENT_DIR%/backend/static:/app/static:ro" ^
    -v "%CURRENT_DIR%/backend/data:/app/data" ^
    -e ASR_API_URL=http://host.docker.internal:8001 ^
    -e ASR_MODEL_NAME=qwen3-asr ^
    -e LLM_API_URL=http://host.docker.internal:8002 ^
    -e LLM_MODEL_NAME=qwen3-1.7b ^
    -e DATABASE_PATH=/app/data/intelligent_record.db ^
    -e NVIDIA_VISIBLE_DEVICES=all ^
    -e MSYS_NO_PATHCONV=1 ^
    --gpus all ^
    intelligent-record-backend:latest

if errorlevel 1 (
    echo [ERROR] Failed to create container
    pause
    exit /b 1
)

:wait_service
:: Wait for service (max 60 seconds)
echo [INFO] Waiting for service to be ready...
set retry=0
:loop
    curl -s http://localhost:8080/api/health >nul 2>&1
    if not errorlevel 1 goto success
    timeout /t 2 /nobreak >nul
    set /a retry+=1
    if %retry% gtr 30 (
        echo [ERROR] Service failed to start
        docker logs intelligent-record-backend --tail 20
        pause
        exit /b 1
    )
    echo     Waiting... (^%retry^%/30)
goto loop

:success
echo [OK] Service is running!
echo.
echo ========================================
echo   Access: http://localhost:8080
echo ========================================
docker ps -f name=intelligent-record-backend --format "table {{.Names}}\t{{.Status}}"
pause
