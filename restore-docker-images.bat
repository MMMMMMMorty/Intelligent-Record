@echo off
chcp 65001 >nul
echo ==========================================
echo   Intelligent Record - Docker 镜像恢复脚本
echo ==========================================
echo.

REM 检查 Docker 是否运行
echo [1/5] 检查 Docker 状态...
docker info >nul 2>&1
if errorlevel 1 (
    echo [错误] Docker 未运行，请先启动 Docker Desktop
    pause
    exit /b 1
)
echo [✓] Docker 运行正常
echo.

REM 显示当前镜像
echo [2/5] 当前 Docker 镜像列表：
docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"
echo.

REM 步骤 1: 构建 qwen-asr-vllm 基础镜像
echo [3/5] 构建 qwen-asr-vllm 基础镜像...
docker build -f Dockerfile -t qwen-asr-vllm:latest .
if errorlevel 1 (
    echo [错误] qwen-asr-vllm 构建失败
    pause
    exit /b 1
)
echo [✓] qwen-asr-vllm 构建完成
echo.

REM 步骤 2: 构建 intelligent-record-asr 镜像
echo [4/5] 构建 intelligent-record-asr 镜像...
docker build -f Dockerfile.asr -t intelligent-record-asr:latest .
if errorlevel 1 (
    echo [错误] intelligent-record-asr 构建失败
    pause
    exit /b 1
)
echo [✓] intelligent-record-asr 构建完成
echo.

REM 步骤 3: 构建 intelligent-record-backend 镜像
echo [5/5] 构建 intelligent-record-backend 镜像...
docker build -f backend/Dockerfile -t intelligent-record-backend:latest backend/
if errorlevel 1 (
    echo [错误] intelligent-record-backend 构建失败
    pause
    exit /b 1
)
echo [✓] intelligent-record-backend 构建完成
echo.

REM 步骤 4: 拉取 vLLM 官方镜像
echo [*] 拉取 vllm/vllm-openai:nightly 镜像...
docker pull vllm/vllm-openai:nightly
if errorlevel 1 (
    echo [警告] vllm/vllm-openai:nightly 拉取失败，可能需要科学上网
    echo       您可以稍后再试：docker pull vllm/vllm-openai:nightly
) else (
    echo [✓] vllm/vllm-openai:nightly 拉取完成
)
echo.

REM 显示最终镜像列表
echo ==========================================
echo   镜像恢复完成！当前镜像列表：
echo ==========================================
docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"
echo.

echo [提示] 现在可以使用以下命令启动服务：
echo        docker-compose up -d
echo.
pause
