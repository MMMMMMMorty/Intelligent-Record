# Qwen3-ASR with vLLM Backend
# Based on vllm/vllm-openai:v0.14.0 with qwen-asr[vllm]

FROM vllm/vllm-openai:v0.14.0

# Use Tsinghua PyPI mirror for faster download in China
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install qwen-asr with vLLM support
RUN pip install --no-cache-dir "qwen-asr[vllm]==0.0.6"

# Set working directory
WORKDIR /app

# Expose port
EXPOSE 8000

# Default command to serve Qwen3-ASR model
# Note: Model path will be mounted at runtime
CMD ["python", "-m", "qwen_asr.cli.serve", "/models/Qwen3-ASR-0.6B", "--host", "0.0.0.0", "--port", "8000", "--dtype", "bfloat16", "--max-model-len", "512", "--gpu-memory-utilization", "0.20"]
