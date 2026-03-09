/**
 * Intelligent Document - Frontend Application
 * Real-time speech recognition with WebSocket streaming
 */

class IntelligentDocument {
    constructor() {
        this.ws = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.isConnected = false;
        this.finalText = '';
        this.currentText = '';
        this.stream = null;

        // Mode: 'transcribe' or 'chat'
        this.mode = 'transcribe';
        this.isProcessingChat = false;

        // Engine control state
        this.engineProgress = {};
        this.engineOperationInProgress = false; // Global flag to prevent concurrent operations

        // Streaming ASR state
        this.audioCtx = null;
        this.source = null;
        this.processor = null;
        this.audioBuffer = null;
        this.pushingChunk = false;
        this.asrSessionId = null;
        this.asrPort = 8001;

        // Document records state
        this.currentRecordId = null;
        this.records = [];

        // UI Elements
        this.elements = {
            btnStart: document.getElementById('btn-start'),
            btnStop: document.getElementById('btn-stop'),
            btnClear: document.getElementById('btn-clear'),
            btnTestHealth: document.getElementById('btn-test-health'),
            btnLoadHistory: document.getElementById('btn-load-history'),
            btnModeTranscribe: document.getElementById('btn-mode-transcribe'),
            btnModeAnalysis: document.getElementById('btn-mode-analysis'),
            asrIndicator: document.getElementById('asr-indicator'),
            asrStatus: document.getElementById('asr-status'),
            llmIndicator: document.getElementById('llm-indicator'),
            llmStatus: document.getElementById('llm-status'),
            wsIndicator: document.getElementById('ws-indicator'),
            wsStatus: document.getElementById('ws-status'),
            recordingIndicator: document.getElementById('recording-indicator'),
            recordStatusText: document.getElementById('record-status-text'),
            realtimeResult: document.getElementById('realtime-result'),
            sectionTranscribe: document.getElementById('section-transcribe'),
            sectionAnalysis: document.getElementById('section-analysis'),
            recordingSection: document.getElementById('recording-section'),
            analysisSource: document.getElementById('analysis-source'),
            analysisOutput: document.getElementById('analysis-output'),
            btnAnalyze: document.getElementById('btn-analyze'),
            modeDesc: document.getElementById('mode-desc'),
            historyList: document.getElementById('history-list'),
            apiResult: document.getElementById('api-result'),
            // Engine control elements
            btnStartAsr: document.getElementById('btn-start-asr'),
            btnStopAsr: document.getElementById('btn-stop-asr'),
            btnStartLlm: document.getElementById('btn-start-llm'),
            btnStopLlm: document.getElementById('btn-stop-llm'),
            asrEngineIndicator: document.getElementById('engine-asr-indicator'),
            asrEngineBadge: document.getElementById('engine-asr-badge'),
            llmEngineIndicator: document.getElementById('engine-llm-indicator'),
            llmEngineBadge: document.getElementById('engine-llm-badge'),
            asrEngineMemory: document.getElementById('engine-asr-memory'),
            llmEngineMemory: document.getElementById('engine-llm-memory'),
            asrEngineProgress: document.getElementById('engine-asr-progress'),
            asrEngineProgressFill: document.getElementById('engine-asr-progress-fill'),
            asrEngineProgressText: document.getElementById('engine-asr-progress-text'),
            llmEngineProgress: document.getElementById('engine-llm-progress'),
            llmEngineProgressFill: document.getElementById('engine-llm-progress-fill'),
            llmEngineProgressText: document.getElementById('engine-llm-progress-text'),
            gpuName: document.getElementById('gpu-name'),
            gpuMemory: document.getElementById('gpu-memory'),
            gpuBarFill: document.getElementById('gpu-bar-fill'),
            gpuPercent: document.getElementById('gpu-percent'),
            // Engine model names
            asrEngineModel: document.getElementById('engine-asr-model'),
            llmEngineModel: document.getElementById('engine-llm-model'),
            // Document records elements
            btnRefreshRecords: document.getElementById('btn-refresh-records'),
            documentRecordsList: document.getElementById('document-records-list'),
            recordModal: document.getElementById('record-modal'),
            createRecordModal: document.getElementById('create-record-modal'),
            btnCloseModal: document.getElementById('btn-close-modal'),
            btnCloseCreateModal: document.getElementById('btn-close-create-modal'),
            btnExtractInfo: document.getElementById('btn-extract-info'),
            btnGenerateRecord: document.getElementById('btn-generate-record'),
            btnSaveRecord: document.getElementById('btn-save-record'),
            btnCreateRecordConfirm: document.getElementById('btn-create-record-confirm'),
            btnCreateRecordCancel: document.getElementById('btn-create-record-cancel'),
            modalRecordTitle: document.getElementById('modal-record-title'),
            modalRecordStatus: document.getElementById('modal-record-status'),
            modalRecordDate: document.getElementById('modal-record-date'),
            modalSourceText: document.getElementById('modal-source-text'),
            modalRecordContent: document.getElementById('modal-record-content'),
            createRecordTranscription: document.getElementById('create-record-transcription'),
            createRecordTitle: document.getElementById('create-record-title')
        };

        this.init();
    }

    init() {
        // Bind events
        this.elements.btnStart.addEventListener('click', () => this.startRecording());
        this.elements.btnStop.addEventListener('click', () => this.stopRecording());
        this.elements.btnClear.addEventListener('click', () => this.clearResult());
        this.elements.btnTestHealth.addEventListener('click', () => this.testHealth());
        this.elements.btnLoadHistory.addEventListener('click', () => this.loadHistory());

        // Mode switching
        this.elements.btnModeTranscribe.addEventListener('click', () => this.setMode('transcribe'));
        this.elements.btnModeAnalysis.addEventListener('click', () => this.setMode('analysis'));

        // Analysis button
        if (this.elements.btnAnalyze) {
            this.elements.btnAnalyze.addEventListener('click', () => this.analyzeContent());
        }

        // Engine control events
        this.elements.btnStartAsr.addEventListener('click', () => this.startEngine('asr'));
        this.elements.btnStopAsr.addEventListener('click', () => this.stopEngine('asr'));
        this.elements.btnStartLlm.addEventListener('click', () => this.startEngine('llm'));
        this.elements.btnStopLlm.addEventListener('click', () => this.stopEngine('llm'));

        // Document records events
        if (this.elements.btnRefreshRecords) {
            this.elements.btnRefreshRecords.addEventListener('click', () => this.loadDocumentRecords());
        }
        if (this.elements.btnCloseModal) {
            this.elements.btnCloseModal.addEventListener('click', () => this.closeRecordModal());
        }
        if (this.elements.btnCloseCreateModal) {
            this.elements.btnCloseCreateModal.addEventListener('click', () => this.closeCreateRecordModal());
        }
        if (this.elements.btnExtractInfo) {
            this.elements.btnExtractInfo.addEventListener('click', () => this.extractRecordInfo());
        }
        if (this.elements.btnGenerateRecord) {
            this.elements.btnGenerateRecord.addEventListener('click', () => this.generateRecord());
        }
        if (this.elements.btnSaveRecord) {
            this.elements.btnSaveRecord.addEventListener('click', () => this.saveRecordContent());
        }
        if (this.elements.btnCreateRecordConfirm) {
            this.elements.btnCreateRecordConfirm.addEventListener('click', () => this.createDocumentRecord());
        }
        if (this.elements.btnCreateRecordCancel) {
            this.elements.btnCreateRecordCancel.addEventListener('click', () => this.closeCreateRecordModal());
        }

        // Initial checks
        this.checkHealth();
        this.updateEngineStatus();
        this.loadDocumentRecords();
        setInterval(() => this.checkHealth(), 10000); // Check every 10s
        setInterval(() => this.updateEngineStatus(), 5000); // Update engine status every 5s
    }

    /**
     * Set working mode
     */
    setMode(mode) {
        this.mode = mode;

        // Update button styles
        this.elements.btnModeTranscribe.classList.toggle('active', mode === 'transcribe');
        this.elements.btnModeAnalysis.classList.toggle('active', mode === 'analysis');

        // Update description
        this.elements.modeDesc.textContent = mode === 'transcribe'
            ? '将语音转换为文字'
            : '对文字内容进行提炼分析';

        // Show/hide sections
        this.elements.sectionTranscribe.classList.toggle('hidden', mode !== 'transcribe');
        this.elements.recordingSection.classList.toggle('hidden', mode !== 'transcribe');
        this.elements.sectionAnalysis.classList.toggle('hidden', mode !== 'analysis');

        // Clear results
        this.clearResult();

        console.log('Mode switched to:', mode);
    }

    /**
     * Check health status (ASR and LLM)
     */
    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();

            // Update ASR status
            if (data.asr_engine === 'ready') {
                this.elements.asrIndicator.className = 'indicator online';
                this.elements.asrStatus.textContent = '服务正常';
            } else {
                this.elements.asrIndicator.className = 'indicator offline';
                this.elements.asrStatus.textContent = '服务未就绪';
            }

            // Update LLM status
            if (data.llm_engine === 'ready') {
                this.elements.llmIndicator.className = 'indicator online';
                this.elements.llmStatus.textContent = '服务正常';
            } else {
                this.elements.llmIndicator.className = 'indicator offline';
                this.elements.llmStatus.textContent = '服务未就绪';
            }
        } catch (error) {
            // Safely update indicators with null checks
            if (this.elements.asrIndicator) {
                this.elements.asrIndicator.className = 'indicator offline';
            }
            if (this.elements.asrStatus) {
                this.elements.asrStatus.textContent = '连接失败';
            }
            if (this.elements.llmIndicator) {
                this.elements.llmIndicator.className = 'indicator offline';
            }
            if (this.elements.llmStatus) {
                this.elements.llmStatus.textContent = '连接失败';
            }
        }
    }

    /**
     * Connect to WebSocket
     */
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            try {
                const wsUrl = `ws://${window.location.host}/ws/stream`;
                console.log('Connecting to WebSocket:', wsUrl);

                this.ws = new WebSocket(wsUrl);

                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    this.isConnected = true;
                    this.elements.wsIndicator.className = 'indicator online';
                    this.elements.wsStatus.textContent = '已连接';
                    resolve();
                };

                this.ws.onmessage = (event) => {
                    this.handleWebSocketMessage(event.data);
                };

                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.elements.wsIndicator.className = 'indicator offline';
                    this.elements.wsStatus.textContent = '连接错误';
                    reject(error);
                };

                this.ws.onclose = () => {
                    console.log('WebSocket closed');
                    this.isConnected = false;
                    this.elements.wsIndicator.className = 'indicator offline';
                    this.elements.wsStatus.textContent = '未连接';
                };

            } catch (error) {
                console.error('Failed to connect:', error);
                reject(error);
            }
        });
    }

    /**
     * Clean ASR text - remove language tags and formatting for streaming ASR
     */
    cleanAsrText(text) {
        if (!text) return '';

        // Convert to string
        text = String(text);

        // Skip standalone language marker tokens (sent separately from backend)
        const languageMarkers = ['chinese', 'english', 'none', 'zh', 'en', 'ja', 'ko', 'fr', 'de', 'es'];
        const lowerTrimmed = text.toLowerCase().trim();
        if (languageMarkers.includes(lowerTrimmed)) {
            return '';
        }

        // Remove language tags in various formats
        // Matches: "language Chinese", "language None", "languageNone", "language=zh", etc.
        text = text.replace(/language\s*[a-z]*[=\s]*[a-z]*/gi, '');
        text = text.replace(/language[=:]?\s*[a-z]*/gi, '');

        // Remove <asr_text> tags if present
        text = text.replace(/<asr_text>/gi, '');
        text = text.replace(/<\/asr_text>/gi, '');

        // Clean up any extra spaces and punctuation artifacts
        text = text.replace(/\s+/g, ' ').trim();

        // Remove leading/trailing punctuation artifacts from partial recognition
        text = text.replace(/^[\s,\.]+|[\s,\.]+$/g, '');

        return text;
    }

    /**
     * Handle WebSocket messages - TRUE STREAMING
     */
    handleWebSocketMessage(data) {
        try {
            const message = JSON.parse(data);
            console.log('Received:', message);

            if (message.type === 'token') {
                // Cumulative: append cleaned token
                const rawToken = message.token || message.text;
                const cleanedToken = this.cleanAsrText(rawToken);

                if (cleanedToken) {
                    this.currentText += cleanedToken;
                    this.updateResultDisplay();
                }

            } else if (message.type === 'transcription') {
                // Final result - move current to final
                if (message.is_final && message.text) {
                    const cleanedText = this.cleanAsrText(message.text);
                    if (!this.finalText.includes(cleanedText)) {
                        this.finalText += (this.finalText ? ' ' : '') + cleanedText;
                    }
                    this.currentText = '';
                    this.saveTranscription(cleanedText);
                    this.updateResultDisplay();
                }

            } else if (message.type === 'error') {
                this.showError(message.error);
            } else if (message.type === 'pong') {
                // Heartbeat response
            }
        } catch (error) {
            console.error('Failed to parse message:', error);
        }
    }

    /**
     * Start recording - TRUE STREAMING using qwen-asr-demo-streaming API
     */
    async startRecording() {
        if (this.isRecording) {
            console.warn('Already recording');
            return;
        }

        try {
            // Get microphone permission
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,  // Request native 16kHz to avoid resampling
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            // Start streaming ASR session
            await this.apiStart();

            // Create AudioContext for real-time audio processing
            this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000  // Force 16kHz
            });
            this.source = this.audioCtx.createMediaStreamSource(this.stream);

            // ScriptProcessor for raw PCM access
            this.processor = this.audioCtx.createScriptProcessor(4096, 1, 1);

            const CHUNK_MS = 200;  // 200ms chunks for lower latency (was 500ms)
            const TARGET_SR = 16000;
            const chunkSamples = Math.round(TARGET_SR * (CHUNK_MS / 1000));

            this.audioBuffer = new Float32Array(0);
            this.pushingChunk = false;

            this.processor.onaudioprocess = (e) => {
                if (!this.isRecording) return;
                const input = e.inputBuffer.getChannelData(0);
                const resampled = this.resampleLinear(input, this.audioCtx.sampleRate, TARGET_SR);
                this.audioBuffer = this.concatFloat32(this.audioBuffer, resampled);
                if (!this.pushingChunk) {
                    this.pumpAudioChunks(chunkSamples);
                }
            };

            this.source.connect(this.processor);
            this.processor.connect(this.audioCtx.destination);

            this.isRecording = true;
            this.finalText = '';
            this.currentText = '';

            // Update UI
            this.elements.btnStart.disabled = true;
            this.elements.btnStop.disabled = false;
            this.elements.recordingIndicator.classList.add('active');
            this.elements.recordStatusText.textContent = '正在识别...';
            this.updateResultDisplay();

            console.log('Streaming recording started');

        } catch (error) {
            console.error('Failed to start recording:', error);
            this.showError('无法访问麦克风: ' + error.message);
        }
    }

    /**
     * Linear resampling for audio
     * CRITICAL: Always copy data, never return input directly
     * because AudioBuffer data is reused by browser after callback
     */
    resampleLinear(input, srcSr, dstSr) {
        // Always copy to avoid AudioBuffer reuse issues
        if (srcSr === dstSr) {
            return new Float32Array(input);
        }
        const ratio = dstSr / srcSr;
        const outLen = Math.max(0, Math.round(input.length * ratio));
        const out = new Float32Array(outLen);
        for (let i = 0; i < outLen; i++) {
            const x = i / ratio;
            const x0 = Math.floor(x);
            const x1 = Math.min(x0 + 1, input.length - 1);
            const t = x - x0;
            out[i] = input[x0] * (1 - t) + input[x1] * t;
        }
        return out;
    }

    /**
     * Concatenate Float32Arrays
     */
    concatFloat32(a, b) {
        const out = new Float32Array(a.length + b.length);
        out.set(a, 0);
        out.set(b, a.length);
        return out;
    }

    /**
     * Pump audio chunks to ASR server with streaming results
     */
    async pumpAudioChunks(chunkSamples) {
        if (this.pushingChunk) return;
        this.pushingChunk = true;

        try {
            while (this.isRecording && this.audioBuffer.length >= chunkSamples) {
                const chunk = this.audioBuffer.slice(0, chunkSamples);
                this.audioBuffer = this.audioBuffer.slice(chunkSamples);

                const result = await this.apiPushChunk(chunk);

                // Handle streaming ASR results
                // result.text contains the current partial transcription
                if (result.text !== undefined) {
                    const cleanedText = this.cleanAsrText(result.text);

                    // Only update if we have meaningful text
                    if (cleanedText) {
                        this.currentText = cleanedText;
                        this.updateResultDisplay();

                        // Log for debugging
                        console.log('ASR partial result:', cleanedText);
                    }
                }

                // Handle completion status
                if (result.is_final || result.done) {
                    console.log('ASR chunk finalized');
                }
            }
        } catch (error) {
            console.error('Chunk push error:', error);
        } finally {
            this.pushingChunk = false;
        }
    }

    /**
     * ASR Streaming API: Start session
     */
    /**
     * ASR Streaming API: Start session (via backend proxy)
     */
    async apiStart() {
        const response = await fetch('/api/asr/start', { method: 'POST' });
        if (!response.ok) {
            throw new Error('Failed to start ASR session');
        }
        const data = await response.json();
        this.asrSessionId = data.session_id;
        console.log('ASR session started:', this.asrSessionId);
    }

    /**
     * Convert Float32Array to Int16Array (PCM 16-bit)
     */
    float32ToInt16(float32Array) {
        const int16Array = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            // Clamp to [-1, 1] and convert to int16
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return int16Array;
    }

    /**
     * ASR Streaming API: Push audio chunk (via backend proxy)
     * Sends PCM Float32, 16kHz, mono audio data (as expected by qwen-asr)
     */
    async apiPushChunk(float32Array) {
        // Send Float32 directly (qwen-asr expects float32, not int16)
        const response = await fetch(
            `/api/asr/chunk?session_id=${encodeURIComponent(this.asrSessionId)}`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/octet-stream' },
                body: float32Array.buffer
            }
        );
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Failed to push audio chunk: ${errorText}`);
        }
        return await response.json();
    }

    /**
     * ASR Streaming API: Finish session (via backend proxy)
     */
    async apiFinish() {
        const response = await fetch(
            `/api/asr/finish?session_id=${encodeURIComponent(this.asrSessionId)}`,
            { method: 'POST' }
        );
        if (!response.ok) {
            throw new Error('Failed to finish ASR session');
        }
        const result = await response.json();

        // Clean the final text
        if (result.text) {
            result.text = this.cleanAsrText(result.text);
        }

        return result;
    }

    /**
     * Stop recording - TRUE STREAMING
     */
    async stopRecording() {
        if (!this.isRecording) {
            return;
        }

        this.isRecording = false;

        // Update UI
        this.elements.btnStart.disabled = false;
        this.elements.btnStop.disabled = true;
        this.elements.recordStatusText.textContent = '正在完成识别...';

        try {
            // Push remaining audio buffer if any
            if (this.audioBuffer && this.audioBuffer.length > 0 && this.asrSessionId) {
                await this.apiPushChunk(this.audioBuffer);
                this.audioBuffer = new Float32Array(0);
            }

            // Get final result from ASR
            if (this.asrSessionId) {
                const result = await this.apiFinish();
                const finalText = result.text || '';

                // Merge current partial with final result
                if (finalText) {
                    this.currentText = finalText;
                    this.updateResultDisplay();

                    // Save to database
                    this.saveTranscription(finalText);
                    console.log('ASR final result:', finalText);
                }
            }
        } catch (error) {
            console.error('ASR finish error:', error);
        }

        // Cleanup audio pipeline
        try {
            if (this.processor) {
                this.processor.disconnect();
                this.processor.onaudioprocess = null;
            }
            if (this.source) this.source.disconnect();
            if (this.audioCtx) await this.audioCtx.close();
            if (this.stream) this.stream.getTracks().forEach(t => t.stop());
        } catch (e) {
            console.error('Audio cleanup error:', e);
        }

        this.processor = null;
        this.source = null;
        this.audioCtx = null;
        this.stream = null;
        this.asrSessionId = null;
        this.audioBuffer = null;

        this.elements.recordingIndicator.classList.remove('active');
        this.elements.recordStatusText.textContent = '录音已停止';

        console.log('Streaming recording stopped');
    }

    /**
     * Analyze content using LLM
     */
    async analyzeContent() {
        const sourceText = this.elements.analysisSource.value.trim();

        if (!sourceText) {
            this.showError('请输入需要分析的内容');
            return;
        }

        this.elements.btnAnalyze.disabled = true;
        this.elements.btnAnalyze.textContent = '分析中...';

        try {
            const response = await fetch('/api/analysis', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: sourceText,
                    prompt: ''  // No preset prompt - pure conversational mode
                })
            });

            const data = await response.json();

            if (data.success) {
                // 将换行符转换为 <br> 标签以便正确显示
                const formattedResult = this.escapeHtml(data.result).replace(/\n/g, '<br>');
                this.elements.analysisOutput.innerHTML = formattedResult;
            } else {
                this.showError(data.error || '分析失败');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError('分析失败: ' + error.message);
        } finally {
            this.elements.btnAnalyze.disabled = false;
            this.elements.btnAnalyze.textContent = '开始分析';
        }
    }

    /**
     * Send audio chunk to server
     */
    async sendAudioChunk(blob) {
        try {
            const base64 = await this.blobToBase64(blob);

            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    type: 'audio',
                    data: base64,
                    is_final: false
                }));
            }
        } catch (error) {
            console.error('Failed to send audio chunk:', error);
        }
    }

    /**
     * Send final audio chunk
     */
    async sendFinalChunk() {
        try {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    type: 'audio',
                    data: '',
                    is_final: true
                }));
            }
        } catch (error) {
            console.error('Failed to send final chunk:', error);
        }
    }

    /**
     * Convert Blob to Base64
     */
    blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                const base64 = reader.result.split(',')[1];
                resolve(base64);
            };
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    }

    /**
     * Update result display
     */
    updateResultDisplay() {
        if (!this.finalText && !this.currentText) {
            this.elements.realtimeResult.innerHTML = '<span class="placeholder">等待识别结果...</span>';
            return;
        }

        let html = '';
        if (this.finalText) {
            html += `<span class="final-text">${this.escapeHtml(this.finalText)}</span>`;
        }
        if (this.currentText) {
            if (this.finalText) html += ' ';
            html += `<span class="current-text">${this.escapeHtml(this.currentText)}<span class="cursor">|</span></span>`;
        }

        this.elements.realtimeResult.innerHTML = html;
        this.elements.realtimeResult.scrollTop = this.elements.realtimeResult.scrollHeight;
    }

    /**
     * Clear result
     */
    clearResult() {
        this.finalText = '';
        this.currentText = '';
        this.updateResultDisplay();

        // Clear analysis display
        if (this.elements.analysisOutput) {
            this.elements.analysisOutput.innerHTML = '<span class="placeholder">分析结果将显示在这里...</span>';
        }

        // Reset WebSocket
        if (this.ws) {
            this.ws.send(JSON.stringify({ type: 'reset' }));
        }

        // Clear audio blob
        this.audioBlob = null;
        this.audioChunks = [];
    }

    /**
     * Save transcription to server
     */
    async saveTranscription(text) {
        try {
            const response = await fetch('/api/transcriptions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    language: 'zh'
                })
            });

            if (response.ok) {
                console.log('Transcription saved');
                // Reload history
                this.loadHistory();
            }
        } catch (error) {
            console.error('Failed to save transcription:', error);
        }
    }

    /**
     * Load history
     */
    async loadHistory() {
        try {
            const response = await fetch('/api/transcriptions?limit=10');
            const data = await response.json();

            if (data.length === 0) {
                this.elements.historyList.innerHTML = '<div class="placeholder">暂无记录</div>';
                return;
            }

            this.elements.historyList.innerHTML = data.map(item => `
                <div class="history-item">
                    <h4>${this.escapeHtml(item.title || '未命名')}</h4>
                    <p>${this.escapeHtml(item.text.substring(0, 100))}${item.text.length > 100 ? '...' : ''}</p>
                    <div class="meta">
                        ${new Date(item.created_at).toLocaleString()}
                        ${item.language ? ` | 语言: ${item.language}` : ''}
                    </div>
                    <div class="actions">
                        <button class="btn btn-secondary btn-small" onclick="app.copyText('${this.escapeHtml(item.text)}')">复制</button>
                        <button class="btn btn-secondary btn-small" onclick="app.deleteTranscription(${item.id})">删除</button>
                        <button class="btn btn-secondary btn-small" onclick="app.createRecordFromTranscription(${item.id}, '${this.escapeHtml(item.title || '未命名')}')">生成文档</button>
                    </div>
                </div>
            `).join('');

        } catch (error) {
            console.error('Failed to load history:', error);
            this.elements.historyList.innerHTML = '<div class="placeholder">加载失败</div>';
        }
    }

    /**
     * Delete transcription
     */
    async deleteTranscription(id) {
        if (!confirm('确定要删除这条记录吗？')) {
            return;
        }

        try {
            const response = await fetch(`/api/transcriptions/${id}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                this.loadHistory();
            }
        } catch (error) {
            console.error('Failed to delete:', error);
        }
    }

    /**
     * Copy text to clipboard
     */
    copyText(text) {
        navigator.clipboard.writeText(text).then(() => {
            alert('已复制到剪贴板');
        });
    }

    /**
     * Test health check API
     */
    async testHealth() {
        try {
            this.elements.apiResult.textContent = 'Testing...';
            const response = await fetch('/api/health');
            const data = await response.json();
            this.elements.apiResult.textContent = JSON.stringify(data, null, 2);
        } catch (error) {
            this.elements.apiResult.textContent = `Error: ${error.message}`;
        }
    }

    // ============================================================================
    // Engine Control Methods
    // ============================================================================

    /**
     * Update engine status display
     */
    async updateEngineStatus() {
        try {
            const response = await fetch('/api/engine/status');
            const data = await response.json();

            // Update GPU info
            if (data.gpu && data.gpu.available) {
                this.elements.gpuName.textContent = data.gpu.name;
                this.elements.gpuMemory.textContent = `${data.gpu.used_gb} / ${data.gpu.total_gb} GB`;
                const percent = Math.round((data.gpu.used_gb / data.gpu.total_gb) * 100);
                this.elements.gpuBarFill.style.width = `${percent}%`;
                this.elements.gpuPercent.textContent = `${percent}%`;

                // Color based on usage
                if (percent > 80) {
                    this.elements.gpuBarFill.className = 'gpu-bar-fill high';
                } else if (percent > 50) {
                    this.elements.gpuBarFill.className = 'gpu-bar-fill medium';
                } else {
                    this.elements.gpuBarFill.className = 'gpu-bar-fill';
                }
            } else {
                this.elements.gpuName.textContent = 'GPU 不可用';
                this.elements.gpuMemory.textContent = '-- / -- GB';
                this.elements.gpuBarFill.style.width = '0%';
                this.elements.gpuPercent.textContent = 'N/A';
            }

            // Update ASR engine
            const asrStatus = data.engines.asr.status;
            const asrMemory = data.engines.asr.actual_memory_gb;
            const llmStatus = data.engines.llm.status;
            const llmMemory = data.engines.llm.actual_memory_gb;

            // Update model names dynamically from backend
            if (data.engines.asr.model_name && this.elements.asrEngineModel) {
                this.elements.asrEngineModel.textContent = data.engines.asr.model_name;
            }
            if (data.engines.llm.model_name && this.elements.llmEngineModel) {
                this.elements.llmEngineModel.textContent = data.engines.llm.model_name;
            }

            // Save ASR port for streaming API
            if (data.engines.asr.port) {
                this.asrPort = data.engines.asr.port;
            }

            // Pass both statuses for mutual exclusion check
            this.updateEngineCard('asr', asrStatus, asrMemory, llmStatus);
            this.updateEngineCard('llm', llmStatus, llmMemory, asrStatus);

        } catch (error) {
            console.error('Failed to update engine status:', error);
        }
    }

    /**
     * Update engine card UI
     */
    updateEngineCard(engine, status, memoryGB = 0, otherEngineStatus = null) {
        const indicator = this.elements[`${engine}EngineIndicator`];
        const badge = this.elements[`${engine}EngineBadge`];
        const btnStart = this.elements[`btnStart${engine.charAt(0).toUpperCase() + engine.slice(1)}`];
        const btnStop = this.elements[`btnStop${engine.charAt(0).toUpperCase() + engine.slice(1)}`];
        const progress = this.elements[`${engine}EngineProgress`];
        const progressFill = this.elements[`${engine}EngineProgressFill`];
        const progressText = this.elements[`${engine}EngineProgressText`];
        const memoryEl = this.elements[`${engine}EngineMemory`];

        // Mutual exclusion: disable start button when other engine is running
        const isOtherEngineBusy = otherEngineStatus === 'ready';

        // Update memory display
        if (memoryEl) {
            if (status === 'ready' && memoryGB > 0) {
                memoryEl.textContent = `显存占用: ${memoryGB.toFixed(1)} GB`;
            } else {
                memoryEl.textContent = '';
            }
        }

        // Update indicator and badge
        indicator.className = 'indicator ' + status;

        const statusText = {
            'offline': '离线',
            'starting': '启动中',
            'ready': '运行中',
            'stopping': '停止中',
            'error': '错误'
        };
        badge.textContent = statusText[status] || status;

        // Check global operation lock
        const isLocked = this.engineOperationInProgress;

        // Sync progress with status - clear progress when status is stable
        if (status === 'ready' || status === 'offline' || status === 'error') {
            this.engineProgress[engine] = 0;
        }

        // Update buttons visibility based on status
        if (status === 'offline' || status === 'error') {
            btnStart.classList.remove('hidden');
            btnStop.classList.add('hidden');
            progress.classList.add('hidden');
        } else if (status === 'ready') {
            btnStart.classList.add('hidden');
            btnStop.classList.remove('hidden');
            progress.classList.add('hidden');
        } else if (status === 'starting') {
            btnStart.classList.add('hidden');
            btnStop.classList.add('hidden');  // Hide stop button during startup
            progress.classList.remove('hidden');
            // Ensure minimum progress display when starting
            let progressPercent = Math.round(this.engineProgress[engine] || 0);
            if (progressPercent < 5) progressPercent = 5;
            progressFill.style.width = `${progressPercent}%`;
            progressText.textContent = `启动中... ${progressPercent}%`;
        } else if (status === 'stopping') {
            btnStart.classList.add('hidden');
            btnStop.classList.add('hidden');
            progress.classList.remove('hidden');
            progressFill.style.width = '50%';
            progressText.textContent = '停止中...';
        }

        // Apply global lock: disable all engine buttons if any operation in progress
        if (isLocked) {
            btnStart.disabled = true;
            btnStop.disabled = true;
            btnStart.title = '当前有引擎操作正在进行，请稍后再试';
        } else {
            // Re-enable buttons based on status when not locked
            if (status === 'offline' || status === 'error') {
                // Disable start button if other engine is running (mutual exclusion)
                btnStart.disabled = isOtherEngineBusy;
                btnStop.disabled = true;
                btnStart.title = isOtherEngineBusy ? '请先停止另一个引擎' : '';
                // Update button text for mutual exclusion
                const btnText = btnStart.querySelector('.btn-text');
                if (btnText) {
                    btnText.textContent = isOtherEngineBusy ? '请先停止另一个引擎' : '启动服务';
                }
            } else if (status === 'ready') {
                btnStart.disabled = true;
                btnStop.disabled = false;
                btnStart.title = '';
            } else if (status === 'starting' || status === 'stopping') {
                btnStart.disabled = true;
                btnStop.disabled = true;
                btnStart.title = '';
            }
        }
    }

    /**
     * Set global operation lock for engine controls
     */
    setEngineOperationLock(locked) {
        this.engineOperationInProgress = locked;
        // Force refresh of all engine cards to update button states
        this.updateEngineStatus();
    }

    /**
     * Start an engine
     */
    async startEngine(engine) {
        // Check if operation already in progress
        if (this.engineOperationInProgress) {
            this.showError('当前有引擎操作正在进行，请稍后再试');
            return;
        }

        // Set global lock to prevent concurrent operations
        this.setEngineOperationLock(true);

        // Initialize progress tracking
        this.engineProgress = this.engineProgress || {};
        this.engineProgress[engine] = 5; // Start from 5% to show activity

        // Start progress animation (simulated progress while waiting)
        const progressInterval = setInterval(() => {
            if (this.engineProgress[engine] < 90) {
                // Slow down as progress increases
                const increment = Math.max(0.3, (90 - this.engineProgress[engine]) / 15);
                this.engineProgress[engine] += Math.random() * increment;
                // Don't update UI here - let the polling handle it
            }
        }, 1000);

        try {
            const response = await fetch(`/api/engine/${engine}/start`, {
                method: 'POST'
            });

            clearInterval(progressInterval);

            if (response.ok) {
                // Hold at 95% - actual completion will be detected by polling
                this.engineProgress[engine] = 95;
            } else {
                const data = await response.json();
                this.engineProgress[engine] = 0;
                this.showError(`启动${engine.toUpperCase()}失败: ${data.detail || '未知错误'}`);
            }
        } catch (error) {
            clearInterval(progressInterval);
            this.engineProgress[engine] = 0;
            console.error(`Failed to start ${engine}:`, error);
            this.showError(`启动${engine.toUpperCase()}失败: ${error.message}`);
        } finally {
            // Release lock and update UI
            this.setEngineOperationLock(false);
            // Note: Progress is NOT cleared here - it will be cleared by updateEngineCard
            // when status changes to 'ready' or 'offline'
        }
    }

    /**
     * Stop an engine
     */
    async stopEngine(engine) {
        // Check if operation already in progress
        if (this.engineOperationInProgress) {
            this.showError('当前有引擎操作正在进行，请稍后再试');
            return;
        }

        // Set global lock
        this.setEngineOperationLock(true);

        // Set stopping state immediately for UI feedback
        this.engineProgress[engine] = 50;

        try {
            const response = await fetch(`/api/engine/${engine}/stop`, {
                method: 'POST'
            });

            if (!response.ok) {
                const data = await response.json();
                this.showError(`停止${engine.toUpperCase()}失败: ${data.detail || '未知错误'}`);
            }
            // Note: Progress will be cleared by updateEngineCard when status changes to 'offline'
        } catch (error) {
            console.error(`Failed to stop ${engine}:`, error);
            this.showError(`停止${engine.toUpperCase()}失败: ${error.message}`);
        } finally {
            // Release lock - updateEngineCard will handle UI based on actual status
            this.setEngineOperationLock(false);
        }
    }

    // ============================================================================
    // Document Records Methods
    // ============================================================================

    /**
     * Load document records list
     */
    async loadDocumentRecords() {
        try {
            const response = await fetch('/api/records?limit=50');
            const data = await response.json();
            this.records = data;

            if (data.length === 0) {
                this.elements.documentRecordsList.innerHTML = `
                    <div class="placeholder">
                        暂无文档记录
                        <br><br>
                        <button class="btn btn-record" onclick="app.openCreateRecordModal()">
                            <span class="btn-text">创建新文档</span>
                        </button>
                    </div>
                `;
                return;
            }

            const statusMap = {
                'draft': { text: '草稿', class: 'draft' },
                'extracting': { text: '提取中', class: 'extracted' },
                'extracted': { text: '已提取', class: 'extracted' },
                'generating': { text: '生成中', class: 'generated' },
                'generated': { text: '已生成', class: 'generated' },
                'completed': { text: '已完成', class: 'completed' },
                'error': { text: '错误', class: 'error' }
            };

            this.elements.documentRecordsList.innerHTML = data.map(record => {
                const status = statusMap[record.status] || { text: record.status, class: 'draft' };
                return `
                    <div class="document-record-item">
                        <h4>${this.escapeHtml(record.title)}</h4>
                        <div class="meta">
                            <span class="status ${status.class}">${status.text}</span>
                            <span>${new Date(record.created_at).toLocaleString()}</span>
                        </div>
                        <div class="actions">
                            <button class="btn btn-secondary btn-small" onclick="app.openRecordModal(${record.id})">查看/编辑</button>
                            <button class="btn btn-secondary btn-small" onclick="app.deleteDocumentRecord(${record.id})">删除</button>
                        </div>
                    </div>
                `;
            }).join('') + `
                <div style="text-align: center; margin-top: 15px;">
                    <button class="btn btn-record" onclick="app.openCreateRecordModal()">
                        <span class="btn-text">创建新文档</span>
                    </button>
                </div>
            `;
        } catch (error) {
            console.error('Failed to load document records:', error);
            this.elements.documentRecordsList.innerHTML = '<div class="placeholder">加载失败</div>';
        }
    }

    /**
     * Open create record modal
     */
    async openCreateRecordModal() {
        // Load transcriptions for dropdown
        try {
            const response = await fetch('/api/transcriptions?limit=50');
            const transcriptions = await response.json();

            const select = this.elements.createRecordTranscription;
            select.innerHTML = '<option value="">请选择转写记录...</option>' +
                transcriptions.map(t => `<option value="${t.id}">${this.escapeHtml(t.title)} - ${t.text.substring(0, 50)}...</option>`).join('');
        } catch (error) {
            console.error('Failed to load transcriptions:', error);
        }

        this.elements.createRecordModal.classList.remove('hidden');
    }

    /**
     * Close create record modal
     */
    closeCreateRecordModal() {
        this.elements.createRecordModal.classList.add('hidden');
        this.elements.createRecordTranscription.value = '';
        this.elements.createRecordTitle.value = '';
    }

    /**
     * Create new document record
     */
    async createDocumentRecord() {
        const transcriptionId = this.elements.createRecordTranscription.value;
        const title = this.elements.createRecordTitle.value.trim();

        if (!transcriptionId) {
            this.showError('请选择转写记录');
            return;
        }
        if (!title) {
            this.showError('请输入文档标题');
            return;
        }

        try {
            const response = await fetch('/api/records', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    transcription_id: parseInt(transcriptionId),
                    title: title
                })
            });

            if (response.ok) {
                this.closeCreateRecordModal();
                this.loadDocumentRecords();
            } else {
                const data = await response.json();
                this.showError(data.detail || '创建失败');
            }
        } catch (error) {
            console.error('Failed to create record:', error);
            this.showError('创建失败: ' + error.message);
        }
    }

    /**
     * Open record detail modal
     */
    async openRecordModal(recordId) {
        this.currentRecordId = recordId;
        const record = this.records.find(r => r.id === recordId);
        if (!record) return;

        // Get full record details
        try {
            const response = await fetch(`/api/records/${recordId}`);
            const fullRecord = await response.json();

            this.elements.modalRecordTitle.textContent = fullRecord.title;

            const statusMap = {
                'draft': '草稿',
                'extracting': '提取中',
                'extracted': '已提取',
                'generating': '生成中',
                'generated': '已生成',
                'completed': '已完成',
                'error': '错误'
            };

            this.elements.modalRecordStatus.textContent = statusMap[fullRecord.status] || fullRecord.status;
            this.elements.modalRecordStatus.className = 'record-status ' + fullRecord.status;
            this.elements.modalRecordDate.textContent = new Date(fullRecord.created_at).toLocaleString();
            this.elements.modalSourceText.textContent = fullRecord.transcription_text || '';
            this.elements.modalRecordContent.value = fullRecord.record_content || '';

            // Update button states based on status
            const canExtract = ['draft', 'error'].includes(fullRecord.status);
            const canGenerate = ['extracted', 'error'].includes(fullRecord.status);

            this.elements.btnExtractInfo.disabled = !canExtract;
            this.elements.btnGenerateRecord.disabled = !canGenerate;

            this.elements.recordModal.classList.remove('hidden');
        } catch (error) {
            console.error('Failed to load record details:', error);
            this.showError('加载文档详情失败');
        }
    }

    /**
     * Close record modal
     */
    closeRecordModal() {
        this.elements.recordModal.classList.add('hidden');
        this.currentRecordId = null;
    }

    /**
     * Extract information from record
     */
    async extractRecordInfo() {
        if (!this.currentRecordId) return;

        this.elements.btnExtractInfo.disabled = true;
        this.elements.btnExtractInfo.textContent = '提取中...';

        try {
            const response = await fetch(`/api/records/${this.currentRecordId}/extract`, {
                method: 'POST'
            });

            const data = await response.json();

            if (data.success) {
                this.showError('信息提取成功！');
                // Reload to update status
                await this.loadDocumentRecords();
                this.openRecordModal(this.currentRecordId);
            } else {
                this.showError(data.error || '提取失败');
            }
        } catch (error) {
            console.error('Failed to extract info:', error);
            this.showError('提取失败: ' + error.message);
        } finally {
            this.elements.btnExtractInfo.disabled = false;
            this.elements.btnExtractInfo.textContent = '提取信息';
        }
    }

    /**
     * Generate document record
     */
    async generateRecord() {
        if (!this.currentRecordId) return;

        this.elements.btnGenerateRecord.disabled = true;
        this.elements.btnGenerateRecord.textContent = '生成中...';

        try {
            const response = await fetch(`/api/records/${this.currentRecordId}/generate`, {
                method: 'POST'
            });

            const data = await response.json();

            if (data.success) {
                this.elements.modalRecordContent.value = data.record_content;
                this.showError('文档生成成功！');
                // Reload to update status
                await this.loadDocumentRecords();
                this.openRecordModal(this.currentRecordId);
            } else {
                this.showError(data.error || '生成失败');
            }
        } catch (error) {
            console.error('Failed to generate record:', error);
            this.showError('生成失败: ' + error.message);
        } finally {
            this.elements.btnGenerateRecord.disabled = false;
            this.elements.btnGenerateRecord.textContent = '生成文档';
        }
    }

    /**
     * Save record content
     */
    async saveRecordContent() {
        if (!this.currentRecordId) return;

        const content = this.elements.modalRecordContent.value;

        try {
            const response = await fetch(`/api/records/${this.currentRecordId}/update?content=${encodeURIComponent(content)}`, {
                method: 'POST'
            });

            const data = await response.json();

            if (data.success) {
                this.showError('保存成功！');
                await this.loadDocumentRecords();
            } else {
                this.showError(data.detail || '保存失败');
            }
        } catch (error) {
            console.error('Failed to save record:', error);
            this.showError('保存失败: ' + error.message);
        }
    }

    /**
     * Delete document record
     */
    async deleteDocumentRecord(recordId) {
        if (!confirm('确定要删除这条文档吗？')) {
            return;
        }

        try {
            const response = await fetch(`/api/records/${recordId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                this.loadDocumentRecords();
            } else {
                const data = await response.json();
                this.showError(data.detail || '删除失败');
            }
        } catch (error) {
            console.error('Failed to delete record:', error);
            this.showError('删除失败: ' + error.message);
        }
    }

    /**
     * Create record from transcription (shortcut from history)
     */
    async createRecordFromTranscription(transcriptionId, title) {
        const recordTitle = prompt('请输入文档标题:', title + ' 文档');
        if (!recordTitle) return;

        try {
            const response = await fetch('/api/records', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    transcription_id: transcriptionId,
                    title: recordTitle
                })
            });

            if (response.ok) {
                this.loadDocumentRecords();
                alert('文档创建成功！请在智能文档板块查看。');
            } else {
                const data = await response.json();
                this.showError(data.detail || '创建失败');
            }
        } catch (error) {
            console.error('Failed to create record:', error);
            this.showError('创建失败: ' + error.message);
        }
    }

    /**
     * Show error message
     */
    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        document.querySelector('.container').appendChild(errorDiv);
        setTimeout(() => errorDiv.remove(), 5000);
    }

    /**
     * Escape HTML
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize app
const app = new IntelligentDocument();
