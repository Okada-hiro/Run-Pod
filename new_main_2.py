#今はこれ! 12/12 15:18

# /workspace/new_main_1.py
# Complete Version: Fast Resampling + VAD + Speaker Register + Instant Voice Switch

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel # 追加
import torch
import numpy as np
import asyncio
import logging
import sys
import os
import io
import re

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
    from new_answer_generator import generate_answer_stream
    # ★ switch_model を追加インポート
    from new_text_to_speech import synthesize_speech, synthesize_speech_to_memory, switch_model
    from speaker_filter import SpeakerGuard
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

# --- ★API: モデル切り替え (追加) ---
class ModelChangeReq(BaseModel):
    model_key: str

@app.post("/change_model")
async def api_change_model(req: ModelChangeReq):
    logger.info(f"🔄 モデル切り替えリクエスト: {req.model_key}")
    # switch_model はメモリ上の変数を変えるだけなので一瞬です
    success = switch_model(req.model_key)
    
    if success:
        return {"status": "ok", "message": f"声を '{req.model_key}' に変更しました"}
    else:
        return {"status": "error", "message": "指定されたモデルが見つかりません"}


# --- ヘルパー: 音声処理パイプライン ---
async def process_voice_pipeline(audio_float32_np, websocket: WebSocket, chat_history: list):
    global NEXT_AUDIO_IS_REGISTRATION

    # SpeakerGuard用に Tensor化
    voice_tensor = torch.from_numpy(audio_float32_np).float().unsqueeze(0)
    
    speaker_id = "Unknown"
    is_allowed = False

    # 1. 話者判定 / 登録ロジック
    if NEXT_AUDIO_IS_REGISTRATION:
        temp_reg_path = f"{PROCESSING_DIR}/reg_{id(audio_float32_np)}.wav"
        import soundfile as sf
        sf.write(temp_reg_path, audio_float32_np, 16000)
        
        new_id = await asyncio.to_thread(speaker_guard.register_new_speaker, temp_reg_path)
        NEXT_AUDIO_IS_REGISTRATION = False 
        
        if new_id:
            speaker_id = new_id
            is_allowed = True
            await websocket.send_json({"status": "system_info", "message": f"✅ {new_id} を登録しました！会話を続けます。"})
        else:
            await websocket.send_json({"status": "error", "message": "登録に失敗しました"})
            return
            
    else:
        is_allowed, detected_id = await asyncio.to_thread(speaker_guard.identify_speaker, voice_tensor)
        speaker_id = detected_id

    # 2. アクセス制御
    if not is_allowed:
        duration_sec = len(audio_float32_np) / 16000
        if duration_sec < 2.5:
            logger.info(f"[Ignored] Short audio ({duration_sec:.2f}s) failed auth.")
            await websocket.send_json({"status": "ignored", "message": "..."})
            return

        logger.info("[Access Denied] 登録されていない話者です。")
        await websocket.send_json({
            "status": "system_alert", 
            "message": "⚠️ 外部の会話(未登録)を検知しました。",
            "alert_type": "unregistered" 
        })
        return

    # 3. Whisper 文字起こし
    try:
        if GLOBAL_ASR_MODEL_INSTANCE is None:
            raise ValueError("Whisper Model not loaded")

        segments = await asyncio.to_thread(
            GLOBAL_ASR_MODEL_INSTANCE.transcribe, 
            audio_float32_np
        )
        text = "".join([s[2] for s in GLOBAL_ASR_MODEL_INSTANCE.ts_words(segments)])
        
        if not text.strip():
            return

        text_with_context = f"【{speaker_id}】 {text}"
        logger.info(f"[TASK] {text_with_context}")

        await websocket.send_json({
            "status": "transcribed",
            "question_text": text,
            "speaker_id": speaker_id 
        })

        # 4. LLM & TTS
        await handle_llm_tts(text_with_context, websocket, chat_history)

    except Exception as e:
        logger.error(f"Pipeline Error: {e}", exc_info=True)
        await websocket.send_json({"status": "error", "message": "処理エラー"})


# --- ヘルパー: 回答生成と音声合成 ---
async def handle_llm_tts(text_for_llm: str, websocket: WebSocket, chat_history: list):
    text_buffer = ""
    sentence_count = 0
    full_answer = ""
    split_pattern = r'(?<=[。！？\n、])'

    iterator = generate_answer_stream(text_for_llm, history=chat_history)

    async def send_audio_chunk(phrase, idx):
        # 1. 音声をメモリ上で作成
        # ★重要: new_text_to_speech 側で既に 16kHz Raw PCM に変換されて返ってきます
        wav_bytes = await asyncio.to_thread(synthesize_speech_to_memory, phrase)
        
        if wav_bytes:
            try:
                # 0.1秒単位で送信 (ネットワーク負荷分散)
                CHUNK_SIZE = 3200 # 16kHz * 2byte * 0.1s
                total_len = len(wav_bytes)
                offset = 0
                
                while offset < total_len:
                    chunk = wav_bytes[offset : offset + CHUNK_SIZE]
                    await websocket.send_bytes(chunk)
                    offset += CHUNK_SIZE
                    await asyncio.sleep(0)
                    
                logger.info(f"🚀 Sent audio {idx} (Size: {len(wav_bytes)})")
            except RuntimeError:
                pass

    try:
        for chunk in iterator:
            text_buffer += chunk
            full_answer += chunk
            
            if full_answer.strip() == "[SILENCE]":
                await websocket.send_json({"status": "system_alert", "message": "⚠️ 会話外と判断しました", "alert_type": "irrelevant"})
                return

            sentences = re.split(split_pattern, text_buffer)
            if len(sentences) > 1:
                for sent in sentences[:-1]:
                    if sent.strip():
                        sentence_count += 1
                        await websocket.send_json({"status": "reply_chunk", "text_chunk": sent})
                        await send_audio_chunk(sent, sentence_count)
                text_buffer = sentences[-1]
        
        if text_buffer.strip():
            sentence_count += 1
            await websocket.send_json({"status": "reply_chunk", "text_chunk": text_buffer})
            await send_audio_chunk(text_buffer, sentence_count)

        chat_history.append({"role": "user", "parts": [text_for_llm]})
        chat_history.append({"role": "model", "parts": [full_answer]})
        
        await websocket.send_json({"status": "complete", "answer_text": full_answer})

    except Exception as e:
        logger.error(f"LLM/TTS Error: {e}")


# --- WebSocket エンドポイント ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WS] Client Connected.")
    
    vad_iterator = VADIterator(
        vad_model, 
        threshold=0.9, 
        sampling_rate=16000, 
        min_silence_duration_ms=200, 
        speech_pad_ms=50
    )

    audio_buffer = [] 
    is_speaking = False
    
    WINDOW_SIZE_SAMPLES = 512 
    SAMPLE_RATE = 16000
    
    chat_history = []

    try:
        while True:
            data_bytes = await websocket.receive_bytes()
            audio_chunk_np = np.frombuffer(data_bytes, dtype=np.float32).copy()
            
            offset = 0
            while offset + WINDOW_SIZE_SAMPLES <= len(audio_chunk_np):
                window_np = audio_chunk_np[offset : offset + WINDOW_SIZE_SAMPLES]
                offset += WINDOW_SIZE_SAMPLES
                window_tensor = torch.from_numpy(window_np).unsqueeze(0).to(DEVICE)

                speech_dict = await asyncio.to_thread(vad_iterator, window_tensor, return_seconds=True)
                
                if speech_dict:
                    if "start" in speech_dict:
                        is_speaking = True
                        audio_buffer = [window_np]
                        await websocket.send_json({"status": "processing", "message": "👂 聞いています..."})
                    
                    elif "end" in speech_dict:
                        if is_speaking:
                            is_speaking = False
                            audio_buffer.append(window_np)
                            full_audio = np.concatenate(audio_buffer)
                            
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


# --- フロントエンド (HTML/JS) ---
@app.get("/", response_class=HTMLResponse)
async def get_root():
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device.width, initial-scale=1.0">
        <title>Team Chat AI</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; display: grid; place-items: center; min-height: 90vh; background: #202c33; color: #e9edef; margin: 0; }
            #container { background: #111b21; padding: 0; border-radius: 0; text-align: center; width: 100%; max-width: 600px; height: 100vh; display: flex; flex-direction: column; box-shadow: 0 0 20px rgba(0,0,0,0.5); position: relative; overflow: hidden; }
            @media (min-width: 600px) { #container { height: 90vh; border-radius: 12px; } }
            
            header { background: #202c33; padding: 15px; border-bottom: 1px solid #374045; font-weight: bold; font-size: 1.1rem; display: flex; justify-content: space-between; align-items: center; z-index: 10; }
            
            /* ★追加: モデル選択プルダウン */
            #model-select {
                background: #2a3942; color: #fff; border: 1px solid #555;
                padding: 6px 10px; border-radius: 6px; font-size: 0.85rem;
                margin-right: 12px; cursor: pointer; outline: none;
            }
            #model-select:hover { background: #37424b; }

            #chat-box { 
                flex: 1; overflow-y: auto; padding: 20px; 
                background-image: url("https://user-images.githubusercontent.com/15075759/28719144-86dc0f70-73b1-11e7-911d-60d70fcded21.png");
                background-repeat: repeat; background-size: 400px; background-color: #0b141a;
            }

            .row { display: flex; width: 100%; margin-bottom: 8px; flex-direction: column; }
            .row.ai { align-items: flex-start; }
            .row.user { align-items: flex-end; }
            .row.system { align-items: center; margin-bottom: 12px; }
            .speaker-name { font-size: 0.75rem; color: #8696a0; margin-bottom: 2px; margin-left: 5px; margin-right: 5px;}
            .bubble { padding: 8px 12px; border-radius: 8px; max-width: 75%; font-size: 0.95rem; line-height: 1.4; word-wrap: break-word; box-shadow: 0 1px 0.5px rgba(0,0,0,0.13); }
            .ai .bubble { background: #202c33; color: #e9edef; border-top-left-radius: 0; }
            .user-type-0 .bubble { background: #005c4b; color: #e9edef; border-top-right-radius: 0; }
            .user-type-1 .bubble { background: #0078d4; color: #fff; border-top-right-radius: 0; }
            .user-type-unknown .bubble { background: #374045; color: #e9edef; border-top-right-radius: 0; }
            .system-bubble { background: #4a3b00; color: #ffecb3; font-size: 0.85rem; padding: 6px 16px; border-radius: 16px; border: 1px solid #ffb300; text-align: center; max-width: 90%; font-weight: 500; }

            #toast-container { position: absolute; top: 70px; left: 50%; transform: translateX(-50%); z-index: 100; width: 90%; max-width: 400px; pointer-events: none; }
            .toast { background: rgba(30, 30, 30, 0.95); color: #fff; padding: 12px 16px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); border-left: 4px solid #f44336; margin-bottom: 10px; font-size: 0.9rem; display: flex; flex-direction: column; gap: 8px; opacity: 0; animation: slideDown 0.3s forwards, fadeOut 0.5s forwards 2.5s; pointer-events: auto; }
            @keyframes slideDown { from { transform: translateY(-20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
            @keyframes fadeOut { from { opacity: 1; } to { opacity: 0; visibility: hidden; } }
            .toast-btn { align-self: flex-end; background: transparent; border: 1px solid #666; color: #ccc; font-size: 0.75rem; padding: 4px 8px; border-radius: 4px; cursor: pointer; }

            #controls { background: #202c33; padding: 15px; border-top: 1px solid #374045; }
            button { padding: 10px 20px; border-radius: 24px; border: none; font-size: 1rem; cursor: pointer; margin: 0 5px; font-weight: bold; transition: opacity 0.2s; }
            button:active { opacity: 0.7; }
            #btn-start { background: #00a884; color: #fff; }
            #btn-stop { background: #ef5350; color: #fff; display: none; }
            #btn-register { background: #3b4a54; color: #fff; font-size: 0.8rem; padding: 8px 15px; }
            #status { margin-bottom: 10px; font-size: 0.9rem; color: #8696a0; min-height: 1.2em; }
        </style>
    </head>
    <body>
        <div id="container">
            <header>
                <span>Team Chat AI</span>
                <div style="display:flex; align-items:center;">
                    <select id="model-select">
                        <option value="default">Default Voice</option>
                        <option value="second">Second Voice</option>
                    </select>
                    <button id="btn-register">＋ メンバー追加</button>
                </div>
            </header>
            
            <div id="toast-container"></div> <div id="chat-box"></div>
            
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
            let sourceInput;
            let isRecording = false;
            
            const btnStart = document.getElementById('btn-start');
            const btnStop = document.getElementById('btn-stop');
            const btnRegister = document.getElementById('btn-register');
            const modelSelect = document.getElementById('model-select');
            const statusDiv = document.getElementById('status');
            const chatBox = document.getElementById('chat-box');
            const toastContainer = document.getElementById('toast-container');

            let audioQueue = [];
            let isPlaying = false;
            let currentSourceNode = null;
            let currentAiBubble = null;
            let muteUnregisteredWarning = false;

            // --- ★モデル切り替え処理 ---
            modelSelect.onchange = async () => {
                const key = modelSelect.value;
                // statusDiv.textContent = "声色を変更中..."; 
                try {
                    const res = await fetch('/change_model', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ model_key: key })
                    });
                    const data = await res.json();
                    if(data.status === 'ok') {
                        showToast("✅ " + data.message);
                    } else {
                        showToast("❌ " + data.message);
                    }
                } catch(e) { console.error(e); }
            };

            // --- Toast通知 ---
            function showToast(message) {
                if (muteUnregisteredWarning) return;
                const toast = document.createElement('div');
                toast.className = 'toast';
                const msgText = document.createElement('span');
                msgText.textContent = message;
                const muteBtn = document.createElement('button');
                muteBtn.className = 'toast-btn';
                muteBtn.textContent = "今後表示しない";
                muteBtn.onclick = () => { muteUnregisteredWarning = true; toast.style.display = 'none'; };
                toast.appendChild(msgText);
                toast.appendChild(muteBtn);
                toastContainer.appendChild(toast);
                setTimeout(() => { if (toast.parentNode) toast.parentNode.removeChild(toast); }, 3000);
            }

            function logChat(role, text, speakerId = null) {
                const row = document.createElement('div');
                row.className = `row ${role}`;
                const bubble = document.createElement('div');
                if (role === 'system') {
                    bubble.className = 'system-bubble'; bubble.textContent = text;
                } else {
                    bubble.className = 'bubble'; bubble.textContent = text;
                    if (role === 'user' && speakerId) {
                        const nameLabel = document.createElement('div');
                        nameLabel.className = 'speaker-name'; nameLabel.textContent = speakerId; 
                        row.insertBefore(nameLabel, row.firstChild);
                        const idNum = speakerId.replace('User ', '');
                        if (!isNaN(idNum)) { row.classList.add(`user-type-${idNum}`); } else { row.classList.add('user-type-unknown'); }
                    } else if (role === 'ai') {
                         const nameLabel = document.createElement('div');
                        nameLabel.className = 'speaker-name'; nameLabel.textContent = "AI Assistant";
                        row.insertBefore(nameLabel, row.firstChild);
                    }
                }
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
                    logChat('ai', "【システム】マイクに向かって話しかけてください。");
                } catch(e) { console.error(e); }
            };

            async function startRecording() {
                try {
                    statusDiv.textContent = "サーバー接続中...";
                    const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
                    socket = new WebSocket(wsProtocol + window.location.host + '/ws');
                    socket.binaryType = 'arraybuffer';

                    socket.onopen = async () => {
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
                            if (data.status === 'processing') statusDiv.textContent = data.message;
                            if (data.status === 'interrupt') stopAudioPlayback();
                            if (data.status === 'system_info') logChat('ai', data.message);
                            if (data.status === 'system_alert') {
                                if (data.alert_type === 'unregistered') showToast(data.message);
                                else if (data.alert_type === 'irrelevant') logChat('system', data.message);
                                statusDiv.textContent = "待機中...";
                            }
                            if (data.status === 'transcribed') logChat('user', data.question_text, data.speaker_id);
                            if (data.status === 'reply_chunk') {
                                if (!currentAiBubble) currentAiBubble = logChat('ai', ''); 
                                currentAiBubble.textContent += data.text_chunk;
                                chatBox.scrollTop = chatBox.scrollHeight;
                            }
                            if (data.status === 'complete') {
                                if (!currentAiBubble && data.answer_text) logChat('ai', data.answer_text);
                                currentAiBubble = null;
                                statusDiv.textContent = "🎙️ 準備OK";
                            }
                        }
                    };
                    socket.onclose = () => stopRecording();
                } catch (e) { console.error(e); }
            }

            async function initAudioStream() {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
                sourceInput = audioContext.createMediaStreamSource(stream);
                processor = audioContext.createScriptProcessor(512, 1, 1);
                processor.onaudioprocess = (e) => {
                    if (!socket || socket.readyState !== WebSocket.OPEN) return;
                    socket.send(e.inputBuffer.getChannelData(0).buffer);
                };
                sourceInput.connect(processor);
                processor.connect(audioContext.destination);
                isRecording = true;
            }

            function stopRecording() {
                isRecording = false;
                if (sourceInput) sourceInput.disconnect();
                if (processor) processor.disconnect();
                if (audioContext) audioContext.close();
                if (socket) socket.close();
                btnStart.style.display = 'inline-block';
                btnStop.style.display = 'none';
                statusDiv.textContent = "停止中";
            }

            function stopAudioPlayback() {
                if (currentSourceNode) { try { currentSourceNode.stop(); } catch(e){} currentSourceNode = null; }
                audioQueue = [];
                isPlaying = false;
            }

            let nextStartTime = 0;

            async function processAudioQueue() {
                if (audioQueue.length === 0) {
                    isPlaying = false;
                    return;
                }
                isPlaying = true;
                const rawBytes = audioQueue.shift();
                
                try {
                    if (audioContext.state === 'suspended') await audioContext.resume();

                    // 1. Raw Int16 -> Float32
                    const int16Data = new Int16Array(rawBytes);
                    const float32Data = new Float32Array(int16Data.length);
                    for (let i = 0; i < int16Data.length; i++) {
                        float32Data[i] = int16Data[i] / 32768.0;
                    }

                    // 2. Buffer作成 (16kHz)
                    const audioBuffer = audioContext.createBuffer(1, float32Data.length, 16000);
                    audioBuffer.getChannelData(0).set(float32Data);

                    // 3. 再生
                    const source = audioContext.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(audioContext.destination);

                    if (nextStartTime < audioContext.currentTime) nextStartTime = audioContext.currentTime;
                    source.start(nextStartTime);
                    nextStartTime += audioBuffer.duration;

                    processAudioQueue();
                    
                } catch(e) { console.error("Play error:", e); isPlaying = false; }
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