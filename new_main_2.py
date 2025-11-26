# /workspace/new_new_main.py
# Server-Side VAD (Silero) + Streaming Architecture + Speaker Registration + UI Improvements

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
    from new_text_to_speech import synthesize_speech
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
        # ファイル経由で登録 (torchaudio互換性のため)
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
    # メモリ上のTensorで高速判定
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

        logger.info("[TASK] 文字起こし開始")
        segments = await asyncio.to_thread(
            GLOBAL_ASR_MODEL_INSTANCE.transcribe, 
            audio_float32_np
        )
        
        # テキスト抽出
        text = "".join([s[2] for s in GLOBAL_ASR_MODEL_INSTANCE.ts_words(segments)])
        
        if not text.strip():
            logger.info("[TASK] 空の認識結果")
            return

        logger.info(f"[TASK] テキスト: {text}")
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


# --- ヘルパー: 回答生成と音声合成 ---
async def handle_llm_tts(text: str, websocket: WebSocket, chat_history: list):
    text_buffer = ""
    sentence_count = 0
    full_answer = ""
    # 「、」も含めて細かく区切る（体感速度向上）
    split_pattern = r'(?<=[。！？\n、])'

    iterator = generate_answer_stream(text, history=chat_history)

    async def send_audio_chunk(phrase, idx):
        filename = f"resp_{idx}.wav"
        path = os.path.join(PROCESSING_DIR, filename)
        success = await asyncio.to_thread(synthesize_speech, phrase, path)
        if success:
            with open(path, 'rb') as f:
                wav_data = f.read()
            await websocket.send_bytes(wav_data)

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
                        await send_audio_chunk(sent, sentence_count)
                text_buffer = sentences[-1]
        
        if text_buffer.strip():
            sentence_count += 1
            await websocket.send_json({"status": "reply_chunk", "text_chunk": text_buffer})
            await send_audio_chunk(text_buffer, sentence_count)

        chat_history.append({"role": "user", "parts": [text]})
        chat_history.append({"role": "model", "parts": [full_answer]})
        
        await websocket.send_json({"status": "complete", "answer_text": full_answer})

    except Exception as e:
        logger.error(f"LLM/TTS Error: {e}")


# ---------------------------
# WebSocket エンドポイント
# ---------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WS] Client Connected.")
    
    vad_iterator = VADIterator(vad_model)
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
                
                # Tensor化 (1, 512)
                window_tensor = torch.from_numpy(window_np).unsqueeze(0).to(DEVICE)

                # VAD判定
                speech_dict = await asyncio.to_thread(vad_iterator, window_tensor, return_seconds=True)
                
                if speech_dict:
                    if "start" in speech_dict:
                        logger.info("🗣️ Speech START")
                        is_speaking = True
                        # ★ UI更新: 聞いています
                        await websocket.send_json({"status": "processing", "message": "👂 聞いています..."})
                        audio_buffer = [window_np] 
                    
                    elif "end" in speech_dict:
                        logger.info("🤫 Speech END")
                        if is_speaking:
                            is_speaking = False
                            audio_buffer.append(window_np)
                            
                            full_audio = np.concatenate(audio_buffer)
                            
                            # ノイズ判定
                            if len(full_audio) / SAMPLE_RATE < 0.2:
                                logger.info("Noise detected (too short)")
                                await websocket.send_json({"status": "ignored", "message": "..."})
                            else:
                                # ★ UI更新: 処理開始フィードバック
                                await websocket.send_json({"status": "processing", "message": "🧠 AI思考中..."})
                                
                                # パイプライン実行
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
# フロントエンド (話者登録 & UI改善版)
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def get_root():
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device.width, initial-scale=1.0">
        <title>Realtime Voice Chat ⚡</title>
        <style>
            body { font-family: sans-serif; display: grid; place-items: center; min-height: 90vh; background: #222; color: #fff; margin: 0; }
            #container { background: #333; padding: 2rem; border-radius: 12px; text-align: center; width: 90%; max-width: 600px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
            
            button { 
                padding: 1rem 1.5rem; border-radius: 30px; border: none; font-size: 1rem; cursor: pointer; margin: 10px; font-weight: bold; transition: all 0.2s;
            }
            button:active { transform: scale(0.95); }
            
            #btn-start { background: #00d2ff; color: #000; }
            #btn-stop { background: #ff4b4b; color: #fff; display: none; }
            #btn-register { background: #28a745; color: #fff; display: none; font-size: 0.9rem; padding: 0.8rem 1.2rem; }
            
            #status { 
                margin-top: 1rem; font-size: 1.3rem; min-height: 1.5em; font-weight: bold;
                padding: 10px; border-radius: 8px; background: rgba(0,0,0,0.2);
            }
            
            .bubble { text-align: left; padding: 12px 18px; margin: 8px; border-radius: 18px; display: inline-block; max-width: 80%; }
            .row { display: flex; width: 100%; margin-bottom: 10px; }
            .row.ai { justify-content: flex-start; }
            .row.user { justify-content: flex-end; }
            
            .ai .bubble { background: #005c4b; color: #fff; border-bottom-left-radius: 4px; }
            .user .bubble { background: #00d2ff; color: #000; border-bottom-right-radius: 4px; }
            
            #chat-box { 
                height: 400px; overflow-y: auto; margin-top: 20px; border: 1px solid #555; padding: 10px; 
                background: #2a2a2a; border-radius: 8px;
            }
        </style>
    </head>
    <body>
        <div id="container">
            <h1>Realtime Talk (L4)</h1>
            <div>
                <button id="btn-start">会話開始</button>
                <button id="btn-stop">停止</button>
            </div>
            <div>
                <button id="btn-register">➕ メンバーを追加</button>
            </div>
            
            <div id="status">待機中</div>
            <div id="chat-box"></div>
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

            // --- UI Helper ---
            function logChat(role, text) {
                const row = document.createElement('div');
                row.className = `row ${role}`;
                const bubble = document.createElement('div');
                bubble.className = 'bubble';
                bubble.textContent = text;
                row.appendChild(bubble);
                chatBox.appendChild(row);
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            // --- メンバー登録 ---
            btnRegister.onclick = async () => {
                try {
                    await fetch('/enable-registration', { method: 'POST' });
                    statusDiv.textContent = "🆕 新しい人が話してください (登録モード)";
                    statusDiv.style.color = "#28a745";
                    logChat('ai', "システム: 次に話す人の声を登録します。何か話しかけてください。");
                } catch(e) { console.error(e); }
            };

            // --- WebSocket & Audio ---
            async function startRecording() {
                try {
                    statusDiv.textContent = "接続中...";
                    const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
                    socket = new WebSocket(wsProtocol + window.location.host + '/ws');
                    socket.binaryType = 'arraybuffer';

                    socket.onopen = async () => {
                        console.log("WS Connected");
                        statusDiv.textContent = "🎙️ お話しください";
                        statusDiv.style.color = "#fff";
                        
                        btnStart.style.display = 'none';
                        btnStop.style.display = 'inline-block';
                        btnRegister.style.display = 'inline-block';
                        
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
                                if (data.message.includes("聞いて")) statusDiv.style.color = "#ff4b4b"; // 赤
                                else if (data.message.includes("思考中")) statusDiv.style.color = "#00d2ff"; // 青
                            }
                            
                            if (data.status === 'transcribed') logChat('user', data.question_text);
                            if (data.status === 'complete') {
                                statusDiv.textContent = "🎙️ お話しください";
                                statusDiv.style.color = "#fff";
                            }
                            if (data.status === 'ignored') {
                                statusDiv.textContent = data.message;
                                setTimeout(() => {
                                     statusDiv.textContent = "🎙️ お話しください";
                                     statusDiv.style.color = "#fff";
                                }, 2000);
                            }
                        }
                    };

                    socket.onclose = () => stopRecording();

                } catch (e) {
                    console.error(e);
                    statusDiv.textContent = "エラー発生";
                }
            }

            async function initAudioStream() {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: { 
                        channelCount: 1, 
                        echoCancellation: true, 
                        noiseSuppression: true,
                        autoGainControl: true
                    } 
                });
                
                source = audioContext.createMediaStreamSource(stream);
                // バッファサイズ 4096 (約256ms)
                processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = (e) => {
                    if (!socket || socket.readyState !== WebSocket.OPEN) return;
                    const inputData = e.inputBuffer.getChannelData(0);
                    socket.send(inputData.buffer);
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
                btnRegister.style.display = 'none';
                statusDiv.textContent = "停止中";
                statusDiv.style.color = "#fff";
            }

            // --- 再生ロジック ---
            async function processAudioQueue() {
                if (isPlaying || audioQueue.length === 0) return;
                isPlaying = true;
                const wavData = audioQueue.shift();
                
                try {
                    if (!audioContext || audioContext.state === 'closed') {
                         audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    }
                    const audioBuffer = await audioContext.decodeAudioData(wavData);
                    const source = audioContext.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(audioContext.destination);
                    source.onended = () => {
                        isPlaying = false;
                        processAudioQueue();
                    };
                    source.start(0);
                } catch(e) {
                    console.error("再生エラー", e);
                    isPlaying = false;
                }
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