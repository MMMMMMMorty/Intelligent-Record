"""
Intelligent Document System - Main Application
FastAPI backend with WebSocket streaming support
"""

"""
Intelligent Document System - Main Application
FastAPI backend with WebSocket streaming support
"""

import os
import io
import json
import base64
import asyncio
import subprocess
import tempfile
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from database import db
from engine_manager import engine_manager
from services.document_processor import document_processor, ExtractedInfo

# ASR Session storage for vLLM streaming mode
asr_sessions: Dict[str, Dict] = {}


def clean_asr_text(text: str) -> str:
    """
    Clean ASR output - remove language tags and formatting
    """
    if not text:
        return ""
    import re
    # Remove language tags like "language Chinese", "language None"
    text = re.sub(r'\s*language\s+\w+\s*', ' ', text, flags=re.IGNORECASE)
    # Remove <asr_text> tags
    text = re.sub(r'<asr_text>', ' ', text, flags=re.IGNORECASE)
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def convert_audio_to_wav(audio_data: bytes, input_format: str = "webm") -> bytes:
    """
    Convert audio data (webm/opus) to WAV format using ffmpeg
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{input_format}", delete=False) as input_file:
            input_file.write(audio_data)
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
            output_path = output_file.name

        # Use ffmpeg to convert to wav (16kHz, mono, 16-bit)
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                output_path
            ],
            check=True,
            capture_output=True,
            timeout=30
        )

        with open(output_path, "rb") as f:
            wav_data = f.read()

        # Cleanup temp files
        os.unlink(input_path)
        os.unlink(output_path)

        return wav_data
    except Exception as e:
        print(f"Audio conversion error: {e}")
        # Return original data if conversion fails
        return audio_data

# Configuration
ASR_API_URL = os.getenv("ASR_API_URL", "http://localhost:8000")
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "qwen3-asr")
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8002")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3-1.7b")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8080").split(",")

# Default system prompt for chat (empty = no system prompt, pure conversational mode)
DEFAULT_SYSTEM_PROMPT = ""


def clean_think_content(text: str) -> str:
    """
    清理 LLM 输出中的 <think>...</think> 思维链内容
    Qwen3 等模型会输出思考过程，需要过滤掉
    """
    import re
    if not text:
        return text
    # 移除 <think>...</think> 标签及其内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 清理多余的空白
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# Pydantic models
class TranscriptionResponse(BaseModel):
    id: int
    title: Optional[str]
    text: str
    language: Optional[str]
    duration_seconds: Optional[float]
    created_at: str


class TranscriptionCreate(BaseModel):
    title: Optional[str] = None
    text: str
    language: Optional[str] = None
    duration_seconds: Optional[float] = None


class ChatRequest(BaseModel):
    audio_base64: Optional[str] = None
    question: Optional[str] = None
    context: Optional[str] = None
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.7
    max_tokens: int = 1536  # 模型总上下文2048，留出约500 tokens给输入


class ChatResponse(BaseModel):
    question: str
    answer: str
    transcription_id: Optional[int] = None
    success: bool
    error: Optional[str] = None


class AnalysisRequest(BaseModel):
    text: str
    prompt: Optional[str] = "请对以下内容进行提炼和分析，以问答格式呈现："
    system_prompt: Optional[str] = None  # None = no system prompt, pure conversational mode
    temperature: float = 0.7
    max_tokens: int = 1536  # 模型总上下文2048，留出约500 tokens给输入


class AnalysisResponse(BaseModel):
    result: str
    success: bool
    error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await db.init()
    print("Database initialized")
    yield
    # Shutdown
    print("Application shutting down")


app = FastAPI(
    title="Intelligengt Record",
    description="Real-time speech recognition with Qwen3-ASR",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main page"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    # Check ASR engine (streaming mode - check if root endpoint responds)
    try:
        async with httpx.AsyncClient() as client:
            # ASR streaming service doesn't have /v1/models, check root endpoint
            response = await client.get(f"{ASR_API_URL}/", timeout=5.0)
            asr_ready = response.status_code in [200, 404]  # 404 is ok if service is running
    except:
        asr_ready = False

    # Check LLM engine
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{LLM_API_URL}/v1/models", timeout=5.0)
            llm_ready = response.status_code == 200
    except:
        llm_ready = False

    return {
        "status": "healthy",
        "asr_engine": "ready" if asr_ready else "not_ready",
        "llm_engine": "ready" if llm_ready else "not_ready",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/transcriptions", response_model=List[TranscriptionResponse])
async def list_transcriptions(limit: int = 50):
    """List all transcription records"""
    records = await db.get_transcriptions(limit=limit)
    return [
        TranscriptionResponse(
            id=r["id"],
            title=r["title"],
            text=r["text"],
            language=r["language"],
            duration_seconds=r["duration_seconds"],
            created_at=r["created_at"]
        ) for r in records
    ]


@app.get("/api/transcriptions/{transcription_id}", response_model=TranscriptionResponse)
async def get_transcription(transcription_id: int):
    """Get a single transcription"""
    record = await db.get_transcription(transcription_id)
    if not record:
        raise HTTPException(status_code=404, detail="Transcription not found")

    return TranscriptionResponse(
        id=record["id"],
        title=record["title"],
        text=record["text"],
        language=record["language"],
        duration_seconds=record["duration_seconds"],
        created_at=record["created_at"]
    )


@app.post("/api/transcriptions", response_model=dict)
async def create_transcription(data: TranscriptionCreate):
    """Create a new transcription record"""
    transcription_id = await db.create_transcription(
        user_id=None,  # Anonymous for now
        title=data.title or f"Recording {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        audio_path=None,
        text=data.text,
        language=data.language,
        duration_seconds=data.duration_seconds
    )
    return {"id": transcription_id, "status": "created"}


@app.delete("/api/transcriptions/{transcription_id}")
async def delete_transcription(transcription_id: int):
    """Delete a transcription"""
    success = await db.delete_transcription(transcription_id)
    if not success:
        raise HTTPException(status_code=404, detail="Transcription not found")
    return {"status": "deleted"}


async def call_llm(
    messages: list,
    model: str = LLM_MODEL_NAME,
    temperature: float = 0.7,
    max_tokens: int = 1536,  # 模型总上下文2048，留出约500 tokens给输入
    stream: bool = False
) -> dict:
    """
    Call LLM API
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LLM_API_URL}/v1/chat/completions",
                json=payload,
                timeout=60.0
            )
            
            # 检查响应状态
            if response.status_code != 200:
                error_text = response.text
                print(f"LLM API error: status={response.status_code}, body={error_text}")
                return {"success": False, "error": f"LLM API error {response.status_code}: {error_text}"}
            
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                # 过滤 <think> 思维链内容
                content = clean_think_content(content)
                return {"success": True, "content": content}
            else:
                print(f"LLM API invalid response: {result}")
                return {"success": False, "error": "Invalid response format"}

    except Exception as e:
        print(f"LLM call exception: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/chat/text", response_model=ChatResponse)
async def chat_text(request: ChatRequest):
    """
    Text-based chat with LLM
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")

    # Build messages (no system prompt if empty - pure conversational mode)
    messages = []
    if request.system_prompt and request.system_prompt.strip():
        messages.append({"role": "system", "content": request.system_prompt})

    if request.context:
        messages.append({"role": "user", "content": f"背景信息：{request.context}"})
        messages.append({"role": "assistant", "content": "我已了解背景信息，请提问。"})

    messages.append({"role": "user", "content": request.question})

    # Call LLM
    result = await call_llm(
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "LLM call failed"))

    return ChatResponse(
        question=request.question,
        answer=result["content"],
        success=True
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_voice(request: ChatRequest):
    """
    Voice-based chat: ASR -> LLM
    """
    if not request.audio_base64:
        raise HTTPException(status_code=400, detail="Audio data is required")

    try:
        # Step 1: ASR - Convert audio to text
        audio_data = base64.b64decode(request.audio_base64)

        # Convert webm/opus to wav format for ASR
        wav_data = convert_audio_to_wav(audio_data, input_format="webm")
        audio_b64_for_api = base64.b64encode(wav_data).decode('utf-8')

        async with httpx.AsyncClient() as client:
            asr_response = await client.post(
                f"{ASR_API_URL}/v1/chat/completions",
                json={
                    "model": ASR_MODEL_NAME,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio_url",
                                    "audio_url": {
                                        "url": f"data:audio/wav;base64,{audio_b64_for_api}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": "将这段语音转录为文字"
                                }
                            ]
                        }
                    ]
                },
                timeout=60.0
            )

            if asr_response.status_code != 200:
                raise HTTPException(
                    status_code=asr_response.status_code,
                    detail=f"ASR error: {asr_response.text}"
                )

            asr_result = asr_response.json()
            question = asr_result.get("choices", [{}])[0].get("message", {}).get("content", "")

        if not question:
            raise HTTPException(status_code=400, detail="Could not recognize speech")

        # Step 2: Save transcription
        transcription_id = await db.create_transcription(
            user_id=None,
            title=f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            audio_path=None,
            text=question,
            language="zh"
        )

        # Step 3: LLM - Generate answer (no system prompt if empty)
        messages = []
        if request.system_prompt and request.system_prompt.strip():
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": question})

        llm_result = await call_llm(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        if not llm_result["success"]:
            raise HTTPException(status_code=500, detail=llm_result.get("error", "LLM call failed"))

        return ChatResponse(
            question=question,
            answer=llm_result["content"],
            transcription_id=transcription_id,
            success=True
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/api/analysis", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    """
    Chat with LLM - pure conversational mode (no preset prompts)
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        # Pure conversational mode: send user text directly without any preset prompt
        user_content = request.text

        # Build messages (no system prompt if not provided - pure conversational mode)
        messages = []
        if request.system_prompt and request.system_prompt.strip():
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": user_content})

        result = await call_llm(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "LLM call failed"))

        return AnalysisResponse(
            result=result["content"],
            success=True
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/transcribe")
async def transcribe_audio(audio_base64: str, language: Optional[str] = None):
    """
    Non-streaming transcription endpoint
    Accepts base64 encoded audio
    """
    try:
        # Decode base64 audio
        audio_data = base64.b64decode(audio_base64)

        # Convert webm/opus to wav format for ASR
        wav_data = convert_audio_to_wav(audio_data, input_format="webm")
        audio_b64_for_api = base64.b64encode(wav_data).decode('utf-8')

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ASR_API_URL}/v1/chat/completions",
                json={
                    "model": ASR_MODEL_NAME,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio_url",
                                    "audio_url": {
                                        "url": f"data:audio/wav;base64,{audio_b64_for_api}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": "将这段语音转录为文字"
                                }
                            ]
                        }
                    ]
                },
                timeout=60.0
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"ASR engine error: {response.text}"
                )

            result = response.json()
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            return {
                "success": True,
                "text": text,
                "language": language or "auto"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for TRUE streaming transcription
    Frontend sends audio chunks, backend streams ASR results token by token
    """
    await websocket.accept()
    print(f"WebSocket connection established: {websocket.client}")

    # Audio buffer for accumulating chunks
    audio_buffer = bytearray()
    is_processing = False

    try:
        while True:
            # Receive message from frontend
            message = await websocket.receive_json()
            msg_type = message.get("type")

            if msg_type == "audio":
                # Receive audio chunk
                audio_base64 = message.get("data", "")
                is_final = message.get("is_final", False)

                try:
                    # Decode and add to buffer
                    audio_data = base64.b64decode(audio_base64)
                    audio_buffer.extend(audio_data)

                    # Only process when is_final=True (recording stopped)
                    # This avoids duplicate recognition during streaming
                    if not is_final:
                        continue

                    # Process only when we have data and not already processing
                    if len(audio_buffer) > 0 and not is_processing:
                        is_processing = True

                        # Convert webm/opus to wav format for ASR
                        wav_data = convert_audio_to_wav(bytes(audio_buffer), input_format="webm")
                        combined_base64 = base64.b64encode(wav_data).decode('utf-8')

                        async with httpx.AsyncClient() as client:
                            # TRUE STREAMING: Use stream=true for SSE
                            async with client.stream(
                                "POST",
                                f"{ASR_API_URL}/v1/chat/completions",
                                json={
                                    "model": ASR_MODEL_NAME,
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "audio_url",
                                                    "audio_url": {
                                                        "url": f"data:audio/wav;base64,{combined_base64}"
                                                    }
                                                },
                                                {
                                                    "type": "text",
                                                    "text": "转录这段语音"
                                                }
                                            ]
                                        }
                                    ],
                                    "stream": True  # Enable streaming
                                },
                                timeout=60.0
                            ) as response:
                                if response.status_code == 200:
                                    # Track already sent content to avoid duplication
                                    last_sent_text = ""
                                    async for line in response.aiter_lines():
                                        # Parse SSE format: data: {...}
                                        if line.startswith("data: "):
                                            data_str = line[6:]  # Remove "data: " prefix

                                            # Check for stream end
                                            if data_str == "[DONE]":
                                                break

                                            try:
                                                data = json.loads(data_str)
                                                # Extract delta content
                                                delta = data.get("choices", [{}])[0].get("delta", {})
                                                content = delta.get("content", "")

                                                if content:
                                                    # Clean the new content
                                                    cleaned_content = clean_asr_text(content)
                                                    # Send only the NEW delta, not accumulated text
                                                    await websocket.send_json({
                                                        "type": "token",
                                                        "token": cleaned_content,
                                                        "text": cleaned_content,  # Send only delta, not full text
                                                        "is_final": False
                                                    })

                                            except json.JSONDecodeError:
                                                continue

                                    # Send final result - don't accumulate, just signal completion
                                    await websocket.send_json({
                                        "type": "transcription",
                                        "text": "",  # Frontend already has all tokens
                                        "is_final": True,
                                        "success": True
                                    })

                                else:
                                    error_text = await response.aread()
                                    print(f"ASR error: {error_text}")
                                    await websocket.send_json({
                                        "type": "error",
                                        "error": f"ASR error: {response.status_code}"
                                    })

                        is_processing = False

                        # Clear buffer on final chunk
                        if is_final:
                            audio_buffer = bytearray()

                except Exception as e:
                    print(f"Error processing audio: {e}")
                    is_processing = False
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e)
                    })

            elif msg_type == "reset":
                # Reset state
                audio_buffer = bytearray()
                is_processing = False
                await websocket.send_json({"type": "reset_ok"})

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {websocket.client}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass


# ============================================================================
# Engine Management APIs - GPU-aware container control
# ============================================================================

@app.get("/api/engine/status")
async def get_engine_status():
    """Get status of all engines and GPU info"""
    # Use async version that refreshes engine status before returning
    return await engine_manager.get_status_async()


@app.post("/api/engine/{engine_name}/start")
async def start_engine(engine_name: str):
    """Start an engine container"""
    if engine_name not in ["asr", "llm"]:
        raise HTTPException(status_code=400, detail="Invalid engine name. Use 'asr' or 'llm'")

    success, message = await engine_manager.start_engine(engine_name)

    if success:
        return {"success": True, "message": message}
    else:
        raise HTTPException(status_code=500, detail=message)


@app.post("/api/engine/{engine_name}/stop")
async def stop_engine(engine_name: str):
    """Stop an engine container"""
    if engine_name not in ["asr", "llm"]:
        raise HTTPException(status_code=400, detail="Invalid engine name. Use 'asr' or 'llm'")

    success, message = await engine_manager.stop_engine(engine_name)

    if success:
        return {"success": True, "message": message}
    else:
        raise HTTPException(status_code=500, detail=message)


@app.get("/api/engine/gpu")
async def get_gpu_info():
    """Get GPU information"""
    gpu = engine_manager.get_gpu_info()
    allocation = engine_manager.calculate_memory_allocation(gpu.total_gb) if gpu.available else {}
    return {
        "gpu": {
            "available": gpu.available,
            "name": gpu.name,
            "total_gb": gpu.total_gb,
            "used_gb": gpu.used_gb,
            "free_gb": gpu.free_gb,
            "utilization_percent": gpu.utilization_percent
        },
        "allocation": allocation
    }


# ============================================================================
# ASR Streaming with vLLM backend
# Accumulates audio chunks and calls vLLM on finish
# ============================================================================

def float32_to_int16_pcm(float32_array: np.ndarray) -> bytes:
    """Convert float32 array to int16 PCM bytes"""
    int16_array = np.clip(float32_array * 32767, -32768, 32767).astype(np.int16)
    return int16_array.tobytes()

def create_wav_header(sample_rate: int, num_channels: int, bits_per_sample: int, data_size: int) -> bytes:
    """Create WAV file header"""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    header = b'RIFF'
    header += (data_size + 36).to_bytes(4, 'little')  # Chunk size
    header += b'WAVE'
    header += b'fmt '
    header += (16).to_bytes(4, 'little')  # Subchunk1 size
    header += (1).to_bytes(2, 'little')   # Audio format (PCM)
    header += (num_channels).to_bytes(2, 'little')
    header += (sample_rate).to_bytes(4, 'little')
    header += (byte_rate).to_bytes(4, 'little')
    header += (block_align).to_bytes(2, 'little')
    header += (bits_per_sample).to_bytes(2, 'little')
    header += b'data'
    header += (data_size).to_bytes(4, 'little')
    return header

# ASR session mapping: frontend_session_id -> asr_service_session_id
asr_session_mapping: Dict[str, str] = {}

# ASR text buffer for smart segmentation: frontend_session_id -> accumulated_text
# This enables long speech recognition by auto-resetting ASR state while preserving context
asr_text_buffer: Dict[str, str] = {}

# Smart segmentation configuration
ASR_SEGMENT_THRESHOLD = 40  # Reset ASR session when text exceeds this length (chars)
ASR_CONTEXT_KEEP_CHARS = 0  # No context kept - avoid repetition issues


@app.post("/api/asr/start")
async def asr_streaming_start():
    """Start ASR streaming session - forwards to ASR service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ASR_API_URL}/api/start",
                timeout=5.0
            )
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="ASR service failed to start session")

            data = response.json()
            asr_session_id = data.get("session_id")

            # Create frontend session mapping
            frontend_session_id = str(uuid.uuid4())
            asr_session_mapping[frontend_session_id] = asr_session_id

            return {"session_id": frontend_session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR start failed: {str(e)}")


@app.post("/api/asr/chunk")
async def asr_streaming_chunk(request: Request, session_id: str = Query(...)):
    """Send audio chunk to ASR streaming with smart segmentation for long speech.

    Automatically resets ASR session when text exceeds threshold while preserving
    recent context. This prevents state accumulation crashes while maintaining
    recognition accuracy. Completely transparent to frontend users.
    """
    try:
        if session_id not in asr_session_mapping:
            raise HTTPException(status_code=400, detail="Invalid session_id")

        asr_session_id = asr_session_mapping[session_id]

        # Read PCM float32 data from frontend
        body = await request.body()

        # Forward to ASR service
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ASR_API_URL}/api/chunk?session_id={asr_session_id}",
                content=body,
                headers={"Content-Type": "application/octet-stream"},
                timeout=10.0
            )
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="ASR service chunk processing failed")

            result = response.json()
            current_text = result.get("text", "") or ""

            # Smart segmentation: Check if we need to reset ASR session
            if len(current_text) >= ASR_SEGMENT_THRESHOLD:
                # Accumulate current text to buffer (save all but context keep chars)
                if ASR_CONTEXT_KEEP_CHARS > 0 and len(current_text) > ASR_CONTEXT_KEEP_CHARS:
                    saved_text = current_text[:-ASR_CONTEXT_KEEP_CHARS]
                else:
                    saved_text = current_text
                asr_text_buffer[session_id] = asr_text_buffer.get(session_id, "") + saved_text

                # Finish current ASR session
                try:
                    await client.post(
                        f"{ASR_API_URL}/api/finish?session_id={asr_session_id}",
                        timeout=5.0
                    )
                except Exception:
                    pass  # Ignore errors during cleanup

                # Start a new ASR session
                start_resp = await client.post(
                    f"{ASR_API_URL}/api/start",
                    timeout=5.0
                )
                if start_resp.status_code == 200:
                    new_asr_session_id = start_resp.json().get("session_id")
                    asr_session_mapping[session_id] = new_asr_session_id

                # Reset current_text to empty for new session
                current_text = ""

            # Always prepend buffer to current text for consistent display
            buffer_text = asr_text_buffer.get(session_id, "")
            if buffer_text:
                result["text"] = buffer_text + current_text

            return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR chunk failed: {str(e)}")


@app.post("/api/asr/finish")
async def asr_streaming_finish(session_id: str = Query(...)):
    """Finish ASR streaming - returns final result including accumulated buffer text"""
    import sys
    print(f"[DEBUG] Finish called with session_id={session_id}", flush=True)
    print(f"[DEBUG] Current mappings: {asr_session_mapping}", flush=True)
    sys.stdout.flush()

    try:
        if session_id not in asr_session_mapping:
            print(f"[DEBUG] Session not found: {session_id}", flush=True)
            raise HTTPException(status_code=400, detail="Invalid session_id")

        asr_session_id = asr_session_mapping[session_id]
        print(f"[DEBUG] Mapped to ASR session: {asr_session_id}", flush=True)

        # Forward to ASR service
        print(f"[DEBUG] Sending request to {ASR_API_URL}/api/finish?session_id={asr_session_id}", flush=True)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ASR_API_URL}/api/finish?session_id={asr_session_id}",
                timeout=10.0
            )
            print(f"[DEBUG] ASR finish response: status={response.status_code}, text={response.text}", flush=True)

            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"ASR service finish failed: {response.text}")

            # Get final result and merge with buffer
            try:
                result = response.json()
            except Exception as json_err:
                print(f"[DEBUG] JSON parse error: {json_err}, response text: {response.text}", flush=True)
                raise HTTPException(status_code=500, detail=f"Invalid JSON from ASR: {response.text[:100]}")

            # Merge buffer with final text
            final_text = result.get("text", "") or ""
            buffered_text = asr_text_buffer.get(session_id, "")
            if buffered_text:
                result["text"] = buffered_text + final_text

            # Clean up mappings and buffer
            del asr_session_mapping[session_id]
            asr_text_buffer.pop(session_id, None)

            return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"[DEBUG] Exception: {e}", flush=True)
        # Clean up mappings and buffer on error
        if session_id in asr_session_mapping:
            del asr_session_mapping[session_id]
        asr_text_buffer.pop(session_id, None)
        raise HTTPException(status_code=500, detail=f"ASR finish failed: {str(e)}")


# ============================================================================
# Document Record APIs - 智能文档生成相关接口
# ============================================================================

class DocumentRecordCreate(BaseModel):
    transcription_id: int
    title: str


class DocumentRecordResponse(BaseModel):
    id: int
    transcription_id: int
    title: str
    status: str
    created_at: str
    updated_at: str


class ExtractInfoResponse(BaseModel):
    success: bool
    extracted_info: Optional[Dict]
    error: Optional[str]


class GenerateRecordResponse(BaseModel):
    success: bool
    record_content: Optional[str]
    error: Optional[str]


@app.post("/api/records", response_model=DocumentRecordResponse)
async def create_document_record(data: DocumentRecordCreate):
    """
    创建文档记录
    """
    # 检查转写记录是否存在
    transcription = await db.get_transcription(data.transcription_id)
    if not transcription:
        raise HTTPException(status_code=404, detail="Transcription not found")
    
    record_id = await db.create_document_record(
        transcription_id=data.transcription_id,
        title=data.title,
        status="draft"
    )
    
    return DocumentRecordResponse(
        id=record_id,
        transcription_id=data.transcription_id,
        title=data.title,
        status="draft",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )


@app.get("/api/records", response_model=List[DocumentRecordResponse])
async def list_document_records(status: Optional[str] = None, limit: int = 50):
    """
    获取文档列表
    """
    records = await db.list_document_records(status=status, limit=limit)
    return [
        DocumentRecordResponse(
            id=r["id"],
            transcription_id=r["transcription_id"],
            title=r["title"],
            status=r["status"],
            created_at=r["created_at"],
            updated_at=r["updated_at"]
        ) for r in records
    ]


@app.get("/api/records/{record_id}")
async def get_document_record(record_id: int):
    """
    获取文档详情
    """
    record = await db.get_document_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    return record


@app.post("/api/records/{record_id}/extract", response_model=ExtractInfoResponse)
async def extract_record_info(record_id: int):
    """
    从转写文本中提取信息
    这是文档生成的第一步
    """
    record = await db.get_document_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    # 获取转写文本
    transcription = await db.get_transcription(record["transcription_id"])
    if not transcription:
        raise HTTPException(status_code=404, detail="Transcription not found")
    
    # 确保 LLM 引擎已启动
    await engine_manager.update_engine_status("llm")
    if engine_manager.engines["llm"].status != "ready":
        success, message = await engine_manager.start_engine("llm")
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to start LLM engine: {message}")
    
    try:
        # 更新状态为处理中
        await db.update_document_record(record_id, status="extracting")
        
        # 调用分段处理提取信息
        extracted_info = await document_processor.extract_information(transcription["text"])
        
        # 保存提取的信息
        await db.update_document_record(
            record_id,
            extracted_info=json.dumps(extracted_info.__dict__, ensure_ascii=False, default=list),
            status="extracted"
        )
        
        return ExtractInfoResponse(
            success=True,
            extracted_info=extracted_info.__dict__,
            error=None
        )
        
    except Exception as e:
        await db.update_document_record(record_id, status="error")
        return ExtractInfoResponse(
            success=False,
            extracted_info=None,
            error=str(e)
        )


@app.post("/api/records/{record_id}/generate", response_model=GenerateRecordResponse)
async def generate_document_record(record_id: int):
    """
    生成文档内容
    这是文档生成的第二步
    """
    record = await db.get_document_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    # 确保 LLM 引擎已启动
    await engine_manager.update_engine_status("llm")
    if engine_manager.engines["llm"].status != "ready":
        success, message = await engine_manager.start_engine("llm")
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to start LLM engine: {message}")
    
    try:
        # 更新状态为生成中
        await db.update_document_record(record_id, status="generating")
        
        # 解析已提取的信息
        extracted_info = None
        if record.get("extracted_info"):
            info_dict = json.loads(record["extracted_info"])
            extracted_info = ExtractedInfo(**info_dict)
        
        # 获取转写文本
        transcription = await db.get_transcription(record["transcription_id"])
        
        # 生成文档
        if extracted_info:
            record_content = await document_processor.generate_record(
                extracted_info,
                transcription["text"]
            )
        else:
            # 如果没有提取信息，直接生成
            record_content = "（请先执行信息提取步骤）"
        
        # 保存文档内容
        await db.update_document_record(
            record_id,
            record_content=record_content,
            status="generated"
        )
        
        return GenerateRecordResponse(
            success=True,
            record_content=record_content,
            error=None
        )
        
    except Exception as e:
        await db.update_document_record(record_id, status="error")
        return GenerateRecordResponse(
            success=False,
            record_content=None,
            error=str(e)
        )


@app.post("/api/records/{record_id}/update")
async def update_document_record_content(record_id: int, content: str = Query(...)):
    """
    更新文档内容（编辑后保存）
    """
    record = await db.get_document_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    success = await db.update_document_record(
        record_id,
        record_content=content,
        status="completed"
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update record")
    
    return {"success": True, "message": "Record updated"}


@app.delete("/api/records/{record_id}")
async def delete_document_record(record_id: int):
    """
    删除文档
    """
    success = await db.delete_document_record(record_id)
    if not success:
        raise HTTPException(status_code=404, detail="Record not found")
    
    return {"success": True, "message": "Record deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
