# /workspace/new_new_main.py
# Final Speed Tuning: Aggressive VAD + In-Memory TTS

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch
import numpy as np
import asyncio
import logging
import sys
import os
import io
import re
import scipy.io.wavfile as wavfile # メモリ書き出し用

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- 必要なモジュールのインポート ---
try:
    from transcribe_func import GLOBAL_ASR_MODEL_INSTANCE
    from supporter_generator import generate_answer_stream
    # TTSのsynthesize_speechは使わず、モデルを直接叩くためインポート不要だが
    # グローバルモデルへのアクセスのために new_text_to_speech を参照
    import new_text_to_speech as tts_module
    from new_speaker_filter import SpeakerGuard
except ImportError as e:
    logger.error(f"[ERROR] 必要なモジュールが見つかりません: {e}")
    sys.exit(1)

# --- グローバル設定 ---
PROCESSING_DIR = "incoming_audio"
os.makedirs(PROCESSING_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using Device: {DEVICE}")

app = FastAPI()
app.mount(f"/download", StaticFiles(directory=PROCESSING_DIR), name="download")

# SpeakerGuard初期化
speaker_guard = SpeakerGuard()
NEXT_AUDIO_IS_REGISTRATION = False

# --- Silero VAD のロード ---
logger.info("⏳ Loading Silero VAD model...")
try:
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    vad_model.to(DEVICE)
    logger.info("✅ Silero VAD model loaded.")
except Exception as e:
    logger.critical(f"Silero VAD Load Failed: {e}")
    sys.exit(1)


# --- API: 登録モード切替 ---
@app.post("/enable-registration")
async def enable_registration():
    global NEXT_AUDIO_IS_REGISTRATION
    NEXT_AUDIO_IS_REGISTRATION = True
    logger.info("【モード切替】次の発話を新規話者として登録します")
    return {"message": "登録モード待機中"}


# --- ヘルパー: 音声処理パイプライン ---
async def process_voice_pipeline(audio_float32_np, websocket: WebSocket, chat_history: list):
    global NEXT_AUDIO_IS_REGISTRATION
    
    # SpeakerGuard用に Tensor化 (1, samples)
    voice_tensor = torch.from_numpy(audio_float32_np).float().unsqueeze(0)

    # ---------------------------
    # 0. 話者登録モード
    # ---------------------------
    if NEXT_AUDIO_IS_REGISTRATION:
        temp_reg_path = f"{PROCESSING_DIR}/reg_{id(audio_float32_np)}.wav"
        import soundfile as sf
        sf.write(temp_reg_path, audio_float32_np, 16000)
        
        success = await asyncio.to_thread(speaker_guard.register_new_speaker, temp_reg_path)
        NEXT_AUDIO_IS_REGISTRATION = False
        
        if success:
            await websocket.send_json({"status": "ignored", "message": "✅ 新しいメンバーを登録しました！"})
        else:
            await websocket.send_json({"status": "error", "message": "登録に失敗しました"})
        return

    # ---------------------------
    # 1. 話者認識 (SpeakerGuard)
    # ---------------------------
    is_allowed = await asyncio.to_thread(speaker_guard.verify_tensor, voice_tensor)

    if not is_allowed:
        logger.info("[Access Denied] 登録されていない話者です。")
        await websocket.send_json({"status": "ignored", "message": "🚫 未登録の声です (ブロック)"})
        return

    # ---------------------------
    # 2. Whisper 文字起こし
    # ---------------------------
    try:
        if GLOBAL_ASR_MODEL_INSTANCE is None:
            raise ValueError("Whisper Model not loaded")

        # faster-whisper は numpy array を直接受け取れる
        segments = await asyncio.to_thread(
            GLOBAL_ASR_MODEL_INSTANCE.transcribe, 
            audio_float32_np
        )
        
        text = "".join([s[2] for s in GLOBAL_ASR_MODEL_INSTANCE.ts_words(segments)])
        
        if not text.strip():
            # logger.info("[TASK] 空の認識結果")
            return

        logger.info(f"📝 認識: {text}")
        await websocket.send_json({
            "status": "transcribed",
            "question_text": text
        })

        # ---------------------------
        # 3. LLM & TTS ストリーミング
        # ---------------------------
        await handle_llm_tts(text, websocket, chat_history)

    except Exception as e:
        logger.error(f"Pipeline Error: {e}", exc_info=True)
        await websocket.send_json({"status": "error", "message": "処理エラー"})


# --- ヘルパー: 回答生成と音声合成 (In-Memory高速化版) ---
async def handle_llm_tts(text: str, websocket: WebSocket, chat_history: list):
    text_buffer = ""
    sentence_count = 0
    full_answer = ""
    # 句読点で細かく区切る
    split_pattern = r'(?<=[。！？\n、])'

    iterator = generate_answer_stream(text, history=chat_history)

    # ★高速化: ファイル保存をスキップしてメモリ上でWAV生成
    async def send_audio_chunk_memory(phrase):
        if not phrase: return
        try:
            # TTSモデルを直接呼び出し
            # new_text_to_speech.py のグローバルモデルを使用
            model = tts_module.GLOBAL_TTS_MODEL
            spk_id = tts_module.GLOBAL_SPEAKER_ID
            
            if model is None: return

            # 推論 (GPU)
            sr, audio_data = await asyncio.to_thread(
                model.infer,
                text=phrase,
                speaker_id=spk_id,
                style="Neutral",
                style_weight=0.5, # 少し弱めて速度優先
                sdp_ratio=0.2,
                noise=0.6,
                noise_w=0.8,
                length=1.0
            )
            
            # Int16変換
            if audio_data.dtype != np.int16:
                audio_norm = audio_data / np.abs(audio_data).max()
                audio_int16 = (audio_norm * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data

            # メモリ上でWAV化 (BytesIO)
            mem_file = io.BytesIO()
            wavfile.write(mem_file, sr, audio_int16)
            wav_bytes = mem_file.getvalue()
            
            # 送信
            await websocket.send_bytes(wav_bytes)

        except Exception as e:
            logger.error(f"TTS Gen Error: {e}")

    try:
        for chunk in iterator:
            text_buffer += chunk
            full_answer += chunk
            
            if full_answer.strip() == "[SILENCE]":
                await websocket.send_json({"status": "ignored", "message": "（応答なし）"})
                return

            sentences = re.split(split_pattern, text_buffer)
            if len(sentences) > 1:
                for sent in sentences[:-1]:
                    if sent.strip():
                        sentence_count += 1
                        await websocket.send_json({"status": "reply_chunk", "text_chunk": sent})
                        # メモリ経由で送信
                        await send_audio_chunk_memory(sent)
                text_buffer = sentences[-1]
        
        if text_buffer.strip():
            sentence_count += 1
            await websocket.send_json({"status": "reply_chunk", "text_chunk": text_buffer})
            await send_audio_chunk_memory(text_buffer)

        chat_history.append({"role": "user", "parts": [text]})
        chat_history.append({"role": "model", "parts": [full_answer]})
        
        await websocket.send_json({"status": "complete", "answer_text": full_answer})

    except Exception as e:
        logger.error(f"LLM/TTS Error: {e}")


# ---------------------------
# WebSocket エンドポイント (鬼VAD設定)
# ---------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WS] Client Connected.")
    
    # ★★★ 最重要チューニング箇所 ★★★
    # threshold: 0.8 (自信度80%未満はノイズとみなす。これで息遣いを除去)
    # min_silence_duration_ms: 200 (200ms無音なら即終了。デフォルト100だと早すぎる場合があるので微調整)
    # speech_pad_ms: 10 (余計な余韻をつけない)
    vad_iterator = VADIterator(
        vad_model, 
        threshold=0.8, 
        min_silence_duration_ms=200, 
        speech_pad_ms=10
    )
    
    audio_buffer = [] 
    is_speaking = False
    
    WINDOW_SIZE_SAMPLES = 512 
    SAMPLE_RATE = 16000
    chat_history = []

    try:
        while True:
            # 1. 受信
            data_bytes = await websocket.receive_bytes()
            audio_chunk_np = np.frombuffer(data_bytes, dtype=np.float32).copy()
            
            # 2. 512サンプル分割ループ
            offset = 0
            while offset + WINDOW_SIZE_SAMPLES <= len(audio_chunk_np):
                window_np = audio_chunk_np[offset : offset + WINDOW_SIZE_SAMPLES]
                offset += WINDOW_SIZE_SAMPLES
                
                # Tensor化
                window_tensor = torch.from_numpy(window_np).unsqueeze(0).to(DEVICE)

                # VAD判定
                speech_dict = await asyncio.to_thread(vad_iterator, window_tensor, return_seconds=True)
                
                if speech_dict:
                    if "start" in speech_dict:
                        logger.info("🗣️ Start")
                        is_speaking = True
                        await websocket.send_json({"status": "processing", "message": "👂 聞いています..."})
                        audio_buffer = [window_np] 
                    
                    elif "end" in speech_dict:
                        logger.info("🤫 End (Cut!)") # 即カットログ
                        if is_speaking:
                            is_speaking = False
                            audio_buffer.append(window_np)
                            
                            full_audio = np.concatenate(audio_buffer)
                            
                            # ノイズ判定 (0.2秒以下は無視)
                            if len(full_audio) / SAMPLE_RATE < 0.2:
                                await websocket.send_json({"status": "ignored", "message": "..."})
                            else:
                                await websocket.send_json({"status": "processing", "message": "🧠 AI思考中..."})
                                await process_voice_pipeline(full_audio, websocket, chat_history)
                            
                            audio_buffer = [] 
                
                else:
                    if is_speaking:
                        audio_buffer.append(window_np)

    except WebSocketDisconnect:
        logger.info("[WS] Disconnected")
    except Exception as e:
        logger.error(f"[WS ERROR] {e}", exc_info=True)
    finally:
        vad_iterator.reset_states()


# ---------------------------
# フロントエンド
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def get_root():
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device.width, initial-scale=1.0">
        <title>Ultra Fast AI Talk</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; display: grid; place-items: center; min-height: 90vh; background: #202c33; color: #e9edef; margin: 0; }
            #container { background: #111b21; padding: 0; border-radius: 0; text-align: center; width: 100%; max-width: 600px; height: 100vh; display: flex; flex-direction: column; }
            @media (min-width: 600px) { #container { height: 90vh; border-radius: 12px; } }
            
            header { background: #202c33; padding: 15px; border-bottom: 1px solid #374045; font-weight: bold; font-size: 1.1rem; display: flex; justify-content: space-between; align-items: center; }
            #chat-box { flex: 1; overflow-y: auto; padding: 20px; background-color: #0b141a; }
            .row { display: flex; width: 100%; margin-bottom: 8px; }
            .row.ai { justify-content: flex-start; }
            .row.user { justify-content: flex-end; }
            .bubble { padding: 8px 12px; border-radius: 8px; max-width: 75%; font-size: 0.95rem; line-height: 1.4; word-wrap: break-word; }
            .ai .bubble { background: #202c33; color: #e9edef; border-top-left-radius: 0; }
            .user .bubble { background: #005c4b; color: #e9edef; border-top-right-radius: 0; }
            #controls { background: #202c33; padding: 15px; border-top: 1px solid #374045; }
            button { padding: 10px 20px; border-radius: 24px; border: none; font-size: 1rem; cursor: pointer; margin: 0 5px; font-weight: bold; }
            #btn-start { background: #00a884; color: #fff; }
            #btn-stop { background: #ef5350; color: #fff; display: none; }
            #btn-register { background: #3b4a54; color: #fff; font-size: 0.8rem; padding: 8px 15px; }
            #status { margin-bottom: 10px; font-size: 0.9rem; color: #8696a0; min-height: 1.2em; }
        </style>
    </head>
    <body>
        <div id="container">
            <header>
                <span>AI Agent ⚡</span>
                <button id="btn-register">＋ メンバー追加</button>
            </header>
            <div id="chat-box"></div>
            <div id="controls">
                <div id="status">接続待機中...</div>
                <button id="btn-start">会話を始める</button>
                <button id="btn-stop">終了する</button>
            </div>
        </div>
        <script>
            let socket;
            let audioContext;
            let processor;
            let source;
            let isRecording = false;
            const btnStart = document.getElementById('btn-start');
            const btnStop = document.getElementById('btn-stop');
            const btnRegister = document.getElementById('btn-register');
            const statusDiv = document.getElementById('status');
            const chatBox = document.getElementById('chat-box');
            let audioQueue = [];
            let isPlaying = false;
            let currentAiBubble = null;

            function logChat(role, text) {
                const row = document.createElement('div');
                row.className = `row ${role}`;
                const bubble = document.createElement('div');
                bubble.className = 'bubble';
                bubble.textContent = text;
                row.appendChild(bubble);
                chatBox.appendChild(row);
                chatBox.scrollTop = chatBox.scrollHeight;
                return bubble;
            }

            btnRegister.onclick = async () => {
                try {
                    await fetch('/enable-registration', { method: 'POST' });
                    statusDiv.textContent = "🆕 新規メンバー登録モード";
                    statusDiv.style.color = "#00a884";
                } catch(e) {}
            };

            async function startRecording() {
                try {
                    statusDiv.textContent = "接続中...";
                    const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
                    socket = new WebSocket(wsProtocol + window.location.host + '/ws');
                    socket.binaryType = 'arraybuffer';
                    socket.onopen = async () => {
                        console.log("WS Connected");
                        statusDiv.textContent = "🎙️ 準備OK";
                        statusDiv.style.color = "#e9edef";
                        btnStart.style.display = 'none';
                        btnStop.style.display = 'inline-block';
                        await initAudioStream();
                    };
                    socket.onmessage = async (event) => {
                        if (event.data instanceof ArrayBuffer) {
                            audioQueue.push(event.data);
                            processAudioQueue();
                        } else {
                            const data = JSON.parse(event.data);
                            if (data.status === 'processing') {
                                statusDiv.textContent = data.message;
                                if (data.message.includes("聞いて")) statusDiv.style.color = "#ef5350"; 
                                else if (data.message.includes("思考中")) statusDiv.style.color = "#00a884";
                            }
                            if (data.status === 'transcribed') logChat('user', data.question_text);
                            if (data.status === 'reply_chunk') {
                                if (!currentAiBubble) currentAiBubble = logChat('ai', '');
                                currentAiBubble.textContent += data.text_chunk;
                                chatBox.scrollTop = chatBox.scrollHeight;
                            }
                            if (data.status === 'complete') {
                                if (!currentAiBubble && data.answer_text) logChat('ai', data.answer_text);
                                currentAiBubble = null;
                                statusDiv.textContent = "🎙️ 準備OK";
                                statusDiv.style.color = "#e9edef";
                            }
                            if (data.status === 'ignored') statusDiv.textContent = data.message;
                        }
                    };
                    socket.onclose = () => stopRecording();
                } catch (e) { statusDiv.textContent = "接続エラー"; }
            }

            async function initAudioStream() {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
                source = audioContext.createMediaStreamSource(stream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);
                processor.onaudioprocess = (e) => {
                    if (!socket || socket.readyState !== WebSocket.OPEN) return;
                    socket.send(e.inputBuffer.getChannelData(0).buffer);
                };
                source.connect(processor);
                processor.connect(audioContext.destination);
                isRecording = true;
            }

            function stopRecording() {
                isRecording = false;
                if (source) source.disconnect();
                if (processor) processor.disconnect();
                if (audioContext) audioContext.close();
                if (socket) socket.close();
                btnStart.style.display = 'inline-block';
                btnStop.style.display = 'none';
                statusDiv.textContent = "停止中";
            }

            async function processAudioQueue() {
                if (isPlaying || audioQueue.length === 0) return;
                isPlaying = true;
                const wavData = audioQueue.shift();
                try {
                    if (!audioContext || audioContext.state === 'closed') audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const audioBuffer = await audioContext.decodeAudioData(wavData);
                    const source = audioContext.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(audioContext.destination);
                    source.onended = () => { isPlaying = false; processAudioQueue(); };
                    source.start(0);
                } catch(e) { isPlaying = false; }
            }

            btnStart.onclick = startRecording;
            btnStop.onclick = stopRecording;
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)