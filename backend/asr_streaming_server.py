#!/usr/bin/env python3
"""
ASR Streaming Server for Qwen3-ASR
Custom implementation with strict VRAM control (6GB target)
Optimized for 8GB GPUs (RTX 5060 Ti / 4080S testing)
"""
import argparse
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from flask import Flask, Response, jsonify, request
from qwen_asr import Qwen3ASRModel


@dataclass
class Session:
    state: object
    created_at: float
    last_seen: float
    accumulated_text: str = ""      # Cross-segment accumulated text
    baseline_text: str = ""         # Baseline text from overlap (for dedup)
    segment_count: int = 1          # Segment counter


app = Flask(__name__)

# Global ASR model instance
global asr
asr = None

# Session management
SESSIONS: Dict[str, Session] = {}
SESSION_TTL_SEC = 10 * 60  # 10 minutes timeout

# Configurable streaming parameters
UNFIXED_CHUNK_NUM = 4
UNFIXED_TOKEN_NUM = 5
CHUNK_SIZE_SEC = 1.0

# Smart segmentation configuration
SAFE_TOKEN_RATIO = 0.70  # Trigger at 70% of max_model_len
MAX_MODEL_LEN = 256      # Will be overridden by args
OVERLAP_SECONDS = 1.0    # 1s overlap: ~4 chars (balance accuracy and duplication)
OVERLAP_SAMPLES = int(OVERLAP_SECONDS * 16000)  # 16000 samples @ 16kHz
# Chinese speech: ~4 chars/second, so 1s = ~4 chars overlap
OVERLAP_CHARS_ESTIMATE = int(OVERLAP_SECONDS * 4)  # ~4 chars


def _estimate_tokens(state) -> int:
    """Estimate token count for current state (audio + text)

    Qwen3-ASR token structure:
    - Encoder (audio): ~50 tokens per second of audio
    - Decoder (text): ~1.5 tokens per Chinese character
    """
    if not state:
        return 0

    # Audio tokens from accumulated audio
    audio_accum = getattr(state, 'audio_accum', None)
    if audio_accum is not None and len(audio_accum) > 0:
        audio_tokens = int(len(audio_accum) / 16000 * 50)
    else:
        audio_tokens = 0

    # Text tokens
    text = getattr(state, 'text', '')
    text_tokens = int(len(text) * 1.5)

    return audio_tokens + text_tokens


def _longest_common_suffix_prefix(s1: str, s2: str, max_len: int = 20) -> int:
    """Find longest overlap where s1 ends with something s2 starts with"""
    if not s1 or not s2:
        return 0

    max_overlap = min(len(s1), len(s2), max_len)
    for i in range(max_overlap, 0, -1):
        if s1[-i:] == s2[:i]:
            return i
    return 0


def _gc_sessions():
    """Garbage collect expired sessions"""
    now = time.time()
    dead = [sid for sid, s in SESSIONS.items() if now - s.last_seen > SESSION_TTL_SEC]
    for sid in dead:
        try:
            asr.finish_streaming_transcribe(SESSIONS[sid].state)
        except Exception:
            pass
        SESSIONS.pop(sid, None)


def _get_session(session_id: str) -> Optional[Session]:
    """Get session by ID, updating last_seen timestamp"""
    _gc_sessions()
    s = SESSIONS.get(session_id)
    if s:
        s.last_seen = time.time()
    return s


def _perform_segmentation(s: Session):
    """Perform smart segmentation with audio overlap for seamless continuity

    Strategy:
    1. Save current segment text to accumulated buffer
    2. Record baseline text (last N chars) for deduplication
    3. Extract overlap audio (last 1.5 seconds) for continuity
    4. Finish current ASR state
    5. Initialize new ASR state and feed overlap audio
    6. Record new baseline text from overlap recognition

    This ensures no audio gaps while avoiding text duplication.
    """
    global asr

    # 1. Save current segment text
    current_text = getattr(s.state, 'text', '')
    s.accumulated_text += current_text

    # 2. Record baseline text for deduplication (last ~6 chars = ~1.5 seconds)
    # This is the text that will be re-recognized in the overlap audio
    if len(current_text) >= OVERLAP_CHARS_ESTIMATE:
        s.baseline_text = current_text[-OVERLAP_CHARS_ESTIMATE:]
    else:
        s.baseline_text = current_text

    print(f"[Segmentation] Segment {s.segment_count} completed. "
          f"Accumulated: {len(s.accumulated_text)} chars, "
          f"Baseline: '{s.baseline_text}'", flush=True)

    # 3. Extract overlap audio (last 1.5 seconds)
    audio_accum = getattr(s.state, 'audio_accum', None)
    if audio_accum is not None and len(audio_accum) >= OVERLAP_SAMPLES:
        overlap_audio = audio_accum[-OVERLAP_SAMPLES:].copy()
    else:
        overlap_audio = np.array([], dtype=np.float32)

    # 4. Finish current ASR state
    try:
        asr.finish_streaming_transcribe(s.state)
    except Exception as e:
        print(f"[Segmentation] Warning: finish_streaming_transcribe failed: {e}", flush=True)

    # 5. Initialize new ASR state
    s.state = asr.init_streaming_state(
        unfixed_chunk_num=UNFIXED_CHUNK_NUM,
        unfixed_token_num=UNFIXED_TOKEN_NUM,
        chunk_size_sec=CHUNK_SIZE_SEC,
    )

    # 6. Feed overlap audio to new state for continuity
    if len(overlap_audio) > 0:
        try:
            asr.streaming_transcribe(overlap_audio, s.state)
            print(f"[Segmentation] Fed {len(overlap_audio)} samples overlap, "
                  f"new state text: '{getattr(s.state, 'text', '')}'", flush=True)
        except Exception as e:
            print(f"[Segmentation] Warning: feeding overlap failed: {e}", flush=True)

    # 7. Increment segment counter
    s.segment_count += 1


def _get_full_text(s: Session) -> str:
    """Get full text combining accumulated and current segment

    Handles deduplication of overlap region using baseline_text.
    Uses fuzzy matching to handle ASR variations in overlap region.
    """
    current_text = getattr(s.state, 'text', '')

    # If we have baseline text (from previous segment's overlap), remove it
    if s.baseline_text and current_text:
        # Try exact match first
        if current_text.startswith(s.baseline_text):
            current_text = current_text[len(s.baseline_text):]
        else:
            # Fuzzy match: find longest common suffix/prefix with relaxed constraints
            # ASR may have slight variations (punctuation differences, etc.)
            baseline_clean = _clean_text_for_comparison(s.baseline_text)
            current_clean = _clean_text_for_comparison(current_text)

            # Try matching on cleaned text
            if current_clean.startswith(baseline_clean):
                # Find corresponding position in original text
                # Approximate: assume similar length ratio
                ratio = len(s.baseline_text) / len(baseline_clean) if baseline_clean else 1
                estimated_len = int(len(baseline_clean) * ratio)
                # Search around estimated position for best match
                for offset in range(-3, 4):
                    pos = estimated_len + offset
                    if 0 <= pos <= len(current_text):
                        current_text = current_text[pos:]
                        break
            else:
                # Partial fuzzy match
                overlap_len = _longest_common_suffix_prefix(baseline_clean, current_clean, max_len=len(baseline_clean))
                if overlap_len > 5:  # Only remove if significant overlap found
                    ratio = len(s.baseline_text) / len(baseline_clean) if baseline_clean else 1
                    estimated_len = int(overlap_len * ratio)
                    if estimated_len <= len(current_text):
                        current_text = current_text[estimated_len:]

    return s.accumulated_text + current_text


def _clean_text_for_comparison(text: str) -> str:
    """Clean text for fuzzy comparison (remove punctuation, normalize)"""
    import re
    # Remove common punctuation and whitespace
    text = re.sub(r'[，。、！？.,!?\s]', '', text)
    return text.lower().strip()


# HTML interface (same as official demo)
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Qwen3-ASR Streaming</title>
  <style>
    :root{
      --bg:#ffffff;
      --card:#ffffff;
      --muted:#5b6472;
      --text:#0f172a;
      --border:#e5e7eb;
      --ok:#059669;
      --warn:#d97706;
      --danger:#e11d48;
    }

    html, body { height: 100%; }

    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans";
      background: var(--bg);
      color:var(--text);
    }

    .wrap{
      height: 100vh;
      max-width: none;
      margin: 0;
      padding: 16px;
      box-sizing: border-box;
      display: flex;
    }

    .card{
      width: 100%;
      height: 100%;
      background: var(--card);
      border:1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      box-sizing: border-box;
      box-shadow: 0 10px 30px rgba(0,0,0,.06);

      display: flex;
      flex-direction: column;
      gap: 12px;
      min-height: 0;
    }

    h1{ font-size: 16px; margin: 0; letter-spacing:.2px;}

    .row{ display:flex; gap:12px; align-items:center; flex-wrap: wrap; }

    button{
      border:1px solid var(--border); border-radius: 12px;
      padding: 10px 14px; cursor:pointer; color:var(--text);
      background: #f8fafc;
      transition: transform .05s ease, background .15s ease, border-color .15s ease;
      font-weight: 700;
    }
    button:hover{ background: #f1f5f9; border-color:#cbd5e1; }
    button:active{ transform: translateY(1px); }
    button.primary{ border-color: rgba(5,150,105,.35); background: rgba(5,150,105,.10); }
    button.danger{ border-color: rgba(225,29,72,.35); background: rgba(225,29,72,.10); }
    button:disabled{ opacity:.5; cursor:not-allowed; }

    .pill{
      font-size: 12px; padding: 6px 10px; border-radius: 999px;
      border:1px solid var(--border); color: var(--muted);
      background: #f8fafc;
      user-select:none;
    }
    .pill.ok{ color: #065f46; border-color: rgba(5,150,105,.35); background: rgba(5,150,105,.10); }
    .pill.warn{ color: #92400e; border-color: rgba(217,119,6,.35); background: rgba(217,119,6,.10); }
    .pill.err{ color: #9f1239; border-color: rgba(225,29,72,.35); background: rgba(225,29,72,.10); }

    .panel{
      border:1px solid var(--border);
      border-radius: 12px;
      background: #ffffff;
      padding: 12px;
    }

    .panel.textpanel{
      flex: 1;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }

    .label{ color:var(--muted); font-size: 12px; margin-bottom: 6px; }
    .mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New"; }

    #text{
      flex: 1;
      min-height: 0;
      white-space: pre-wrap;
      line-height: 1.6;
      font-size: 15px;
      padding: 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #f8fafc;
      overflow: auto;
    }

    a{ color: #2563eb; text-decoration:none; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Qwen3-ASR Streaming</h1>

      <div class="row">
        <button id="btnStart" class="primary">Start / 开始</button>
        <button id="btnStop" class="danger" disabled>Stop / 停止</button>
        <span id="status" class="pill warn">Idle / 未开始</span>
        <a href="javascript:void(0)" id="btnClear" class="mono" style="margin-left:auto;">Clear / 清空</a>
      </div>

      <div class="panel">
        <div class="label">Language / 语言</div>
        <div id="lang" class="mono">—</div>
      </div>

      <div class="panel textpanel">
        <div class="label">Text / 文本</div>
        <div id="text"></div>
      </div>
    </div>
  </div>

<script>
(() => {
  const $ = (id) => document.getElementById(id);

  const btnStart = $("btnStart");
  const btnStop  = $("btnStop");
  const btnClear = $("btnClear");
  const statusEl = $("status");
  const langEl   = $("lang");
  const textEl   = $("text");

  const CHUNK_MS = 500;
  const TARGET_SR = 16000;

  let audioCtx = null;
  let processor = null;
  let source = null;
  let mediaStream = null;

  let sessionId = null;
  let running = false;

  let buf = new Float32Array(0);
  let pushing = false;

  function setStatus(text, cls){
    statusEl.textContent = text;
    statusEl.className = "pill " + (cls || "");
  }

  function lockUI(on){
    btnStart.disabled = on;
    btnStop.disabled = !on;
  }

  function concatFloat32(a, b){
    const out = new Float32Array(a.length + b.length);
    out.set(a, 0);
    out.set(b, a.length);
    return out;
  }

  function resampleLinear(input, srcSr, dstSr){
    if (srcSr === dstSr) return input;
    const ratio = dstSr / srcSr;
    const outLen = Math.max(0, Math.round(input.length * ratio));
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++){
      const x = i / ratio;
      const x0 = Math.floor(x);
      const x1 = Math.min(x0 + 1, input.length - 1);
      const t = x - x0;
      out[i] = input[x0] * (1 - t) + input[x1] * t;
    }
    return out;
  }

  async function apiStart(){
    const r = await fetch("/api/start", {method:"POST"});
    if(!r.ok) throw new Error(await r.text());
    const j = await r.json();
    sessionId = j.session_id;
  }

  async function apiPushChunk(float32_16k){
    const r = await fetch("/api/chunk?session_id=" + encodeURIComponent(sessionId), {
      method: "POST",
      headers: {"Content-Type":"application/octet-stream"},
      body: float32_16k.buffer
    });
    if(!r.ok) throw new Error(await r.text());
    return await r.json();
  }

  async function apiFinish(){
    const r = await fetch("/api/finish?session_id=" + encodeURIComponent(sessionId), {method:"POST"});
    if(!r.ok) throw new Error(await r.text());
    return await r.json();
  }

  btnClear.onclick = () => { textEl.textContent = ""; };

  async function stopAudioPipeline(){
    try{
      if (processor){ processor.disconnect(); processor.onaudioprocess = null; }
      if (source) source.disconnect();
      if (audioCtx) await audioCtx.close();
      if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    }catch(e){}
    processor = null; source = null; audioCtx = null; mediaStream = null;
  }

  btnStart.onclick = async () => {
    if (running) return;

    textEl.textContent = "";
    langEl.textContent = "—";
    buf = new Float32Array(0);

    try{
      setStatus("Starting… / 启动中…", "warn");
      lockUI(true);

      await apiStart();

      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        },
        video: false
      });

      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      source = audioCtx.createMediaStreamSource(mediaStream);

      processor = audioCtx.createScriptProcessor(4096, 1, 1);
      const chunkSamples = Math.round(TARGET_SR * (CHUNK_MS / 1000));

      processor.onaudioprocess = (e) => {
        if (!running) return;
        const input = e.inputBuffer.getChannelData(0);
        const resampled = resampleLinear(input, audioCtx.sampleRate, TARGET_SR);
        buf = concatFloat32(buf, resampled);
        if (!pushing) pump();
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      running = true;
      setStatus("Listening… / 识别中…", "ok");

    }catch(err){
      console.error(err);
      setStatus("Start failed / 启动失败: " + err.message, "err");
      lockUI(false);
      running = false;
      sessionId = null;
      await stopAudioPipeline();
    }
  };

  async function pump(){
    if (pushing) return;
    pushing = true;

    const chunkSamples = Math.round(TARGET_SR * (CHUNK_MS / 1000));

    try{
      while (running && buf.length >= chunkSamples){
        const chunk = buf.slice(0, chunkSamples);
        buf = buf.slice(chunkSamples);

        const j = await apiPushChunk(chunk);
        langEl.textContent = j.language || "—";
        textEl.textContent = j.text || "";
        if (running) setStatus("Listening… / 识别中…", "ok");
      }
    }catch(err){
      console.error(err);
      if (running) setStatus("Backend error / 后端错误: " + err.message, "err");
    }finally{
      pushing = false;
    }
  }

  btnStop.onclick = async () => {
    if (!running) return;

    running = false;
    setStatus("Finishing… / 收尾中…", "warn");
    lockUI(false);

    await stopAudioPipeline();

    try{
      if (sessionId){
        const j = await apiFinish();
        langEl.textContent = j.language || "—";
        textEl.textContent = j.text || "";
      }
      setStatus("Stopped / 已停止", "");
    }catch(err){
      console.error(err);
      setStatus("Finish failed / 收尾失败: " + err.message, "err");
    }finally{
      sessionId = null;
      buf = new Float32Array(0);
      pushing = false;
    }
  };
})();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    """Serve the web interface"""
    return Response(INDEX_HTML, mimetype="text/html; charset=utf-8")


@app.post("/api/start")
def api_start():
    """Initialize a new streaming session"""
    session_id = uuid.uuid4().hex
    state = asr.init_streaming_state(
        unfixed_chunk_num=UNFIXED_CHUNK_NUM,
        unfixed_token_num=UNFIXED_TOKEN_NUM,
        chunk_size_sec=CHUNK_SIZE_SEC,
    )
    now = time.time()
    SESSIONS[session_id] = Session(state=state, created_at=now, last_seen=now)
    return jsonify({"session_id": session_id})


@app.post("/api/chunk")
def api_chunk():
    """Process an audio chunk with smart segmentation for unlimited length recognition"""
    global MAX_MODEL_LEN, SAFE_TOKEN_RATIO

    session_id = request.args.get("session_id", "")
    s = _get_session(session_id)
    if not s:
        return jsonify({"error": "invalid session_id"}), 400

    if request.mimetype != "application/octet-stream":
        return jsonify({"error": "expect application/octet-stream"}), 400

    raw = request.get_data(cache=False)
    if len(raw) % 4 != 0:
        return jsonify({"error": "float32 bytes length not multiple of 4"}), 400

    wav = np.frombuffer(raw, dtype=np.float32).reshape(-1)

    # Check if segmentation is needed before processing
    safe_threshold = int(MAX_MODEL_LEN * SAFE_TOKEN_RATIO)
    current_tokens = _estimate_tokens(s.state)
    incoming_audio_tokens = int(len(wav) / 16000 * 50)

    if current_tokens + incoming_audio_tokens >= safe_threshold:
        print(f"[Segmentation] Triggered: current={current_tokens}, incoming={incoming_audio_tokens}, "
              f"threshold={safe_threshold}", flush=True)
        _perform_segmentation(s)

    # Process audio
    asr.streaming_transcribe(wav, s.state)

    # Return full text (accumulated + current segment with dedup)
    full_text = _get_full_text(s)

    return jsonify(
        {
            "language": getattr(s.state, "language", "") or "",
            "text": full_text,
            "segment": s.segment_count,
        }
    )


@app.post("/api/finish")
def api_finish():
    """Finalize streaming and return complete transcription across all segments"""
    session_id = request.args.get("session_id", "")
    s = _get_session(session_id)
    if not s:
        return jsonify({"error": "invalid session_id"}), 400

    # Finish current segment
    asr.finish_streaming_transcribe(s.state)

    # Get final full text
    final_text = _get_full_text(s)

    out = {
        "language": getattr(s.state, "language", "") or "",
        "text": final_text,
        "segments": s.segment_count,
    }
    SESSIONS.pop(session_id, None)

    print(f"[Finish] Total segments: {s.segment_count}, Final text length: {len(final_text)} chars", flush=True)
    return jsonify(out)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ASR Streaming Server with strict VRAM control (6GB target)"
    )
    parser.add_argument(
        "--asr-model-path",
        default="/models/Qwen3-ASR-1.7B",
        help="Model name or local path"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.75,
        help="vLLM GPU memory utilization (0.75 for 6GB on 8GB card)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=256,
        help="Maximum model context length (reduce for lower VRAM)"
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="Maximum number of concurrent sequences (1 for streaming)"
    )
    parser.add_argument(
        "--unfixed-chunk-num",
        type=int,
        default=4,
        help="Number of initial chunks without prefix prompt"
    )
    parser.add_argument(
        "--unfixed-token-num",
        type=int,
        default=5,
        help="Number of tokens to rollback for prefix"
    )
    parser.add_argument(
        "--chunk-size-sec",
        type=float,
        default=1.0,
        help="Audio chunk size in seconds"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    global asr
    global UNFIXED_CHUNK_NUM
    global UNFIXED_TOKEN_NUM
    global CHUNK_SIZE_SEC
    global MAX_MODEL_LEN
    global SAFE_TOKEN_RATIO

    UNFIXED_CHUNK_NUM = args.unfixed_chunk_num
    UNFIXED_TOKEN_NUM = args.unfixed_token_num
    CHUNK_SIZE_SEC = args.chunk_size_sec
    MAX_MODEL_LEN = args.max_model_len

    safe_threshold = int(MAX_MODEL_LEN * SAFE_TOKEN_RATIO)
    print(f"[Config] Smart segmentation enabled: threshold={safe_threshold} tokens (max={MAX_MODEL_LEN})", flush=True)

    print("=" * 60)
    print("ASR Streaming Server - 6GB VRAM Optimized")
    print("=" * 60)
    print(f"Model: {args.asr_model_path}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"Max model length: {args.max_model_len}")
    print(f"Max num sequences: {args.max_num_seqs}")
    print(f"Host: {args.host}:{args.port}")
    print("=" * 60)
    print("Loading model...")

    # Initialize ASR model with strict VRAM limits
    # These kwargs are passed directly to vLLM
    asr = Qwen3ASRModel.LLM(
        model=args.asr_model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_new_tokens=32,  # Streaming ASR only needs short outputs
    )

    print("Model loaded successfully!")
    print(f"Starting server on http://{args.host}:{args.port}")
    print("=" * 60)

    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        use_reloader=False,
        threaded=True
    )


if __name__ == "__main__":
    main()
