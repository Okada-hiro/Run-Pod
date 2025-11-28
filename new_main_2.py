# /workspace/new_main_2.py
# Server-Side VAD + Streaming + Speaker ID & Colors + Multi-user Context

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


# --- ヘルパー: 音声処理パイプライン ---
async def process_voice_pipeline(audio_float32_np, websocket: WebSocket, chat_history: list):
    global NEXT_AUDIO_IS_REGISTRATION
    
    # SpeakerGuard用に Tensor化 (1, samples)
    voice_tensor = torch.from_numpy(audio_float32_np).float().unsqueeze(0)
    
    speaker_id = "Unknown"
    is_allowed = False

    # ---------------------------
    # 1. 話者判定 / 登録ロジック
    # ---------------------------
    if NEXT_AUDIO_IS_REGISTRATION:
        # --- 新規登録モード ---
        temp_reg_path = f"{PROCESSING_DIR}/reg_{id(audio_float32_np)}.wav"
        import soundfile as sf
        sf.write(temp_reg_path, audio_float32_np, 16000)
        
        # 登録実行してIDを取得
        new_id = await asyncio.to_thread(speaker_guard.register_new_speaker, temp_reg_path)
        NEXT_AUDIO_IS_REGISTRATION = False # フラグを戻す
        
        if new_id:
            speaker_id = new_id
            is_allowed = True
            await websocket.send_json({"status": "system_info", "message": f"✅ {new_id} を登録しました！会話を続けます。"})
            # ★ ここで return せず、そのまま下流へ流す！ ★
        else:
            await websocket.send_json({"status": "error", "message": "登録に失敗しました"})
            return
            
    else:
        # --- 通常モード: 話者特定 ---
        # identify_speaker は (bool, speaker_id) を返すように変更済み
        is_allowed, detected_id = await asyncio.to_thread(speaker_guard.identify_speaker, voice_tensor)
        speaker_id = detected_id

    # ---------------------------
    # 2. アクセス制御
    # ---------------------------
    if not is_allowed:
        logger.info("[Access Denied] 登録されていない話者です。")
        await websocket.send_json({"status": "ignored", "message": "🚫 未登録の声です (ブロック)"})
        return

    # ---------------------------
    # 3. Whisper 文字起こし
    # ---------------------------
    try:
        if GLOBAL_ASR_MODEL_INSTANCE is None:
            raise ValueError("Whisper Model not loaded")

        logger.info("[TASK] 文字起こし開始")
        segments = await asyncio.to_thread(
            GLOBAL_ASR_MODEL_INSTANCE.transcribe, 
            audio_float32_np
        )
        
        text = "".join([s[2] for s in GLOBAL_ASR_MODEL_INSTANCE.ts_words(segments)])
        
        if not text.strip():
            logger.info("[TASK] 空の認識結果")
            return

        # ★ LLMへの入力用に、話者タグを付与
        text_with_context = f"【{speaker_id}】 {text}"
        logger.info(f"[TASK] {text_with_context}")

        # フロントエンドには「表示用テキスト」と「話者ID」を送る
        await websocket.send_json({
            "status": "transcribed",
            "question_text": text,
            "speaker_id": speaker_id  # 色分け用ID
        })

        # ---------------------------
        # 4. LLM & TTS ストリーミング
        # ---------------------------
        # ここで text ではなく text_with_context を渡すことで、LLMは誰の発言か理解する
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

    # LLMにはタグ付きテキスト(text_for_llm)を渡す
    iterator = generate_answer_stream(text_for_llm, history=chat_history)

    async def send_audio_chunk(phrase, idx):
        filename = f"resp_{idx}.wav"
        path = os.path.join(PROCESSING_DIR, filename)
        success = await asyncio.to_thread(synthesize_speech, phrase, path)
        if success:
            with open(path, 'rb') as f:
                wav_data = f.read()
            try:
                await websocket.send_bytes(wav_data)
            except RuntimeError:
                pass

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

        # 履歴にもタグ付きで保存されるので、文脈が維持される
        chat_history.append({"role": "user", "parts": [text_for_llm]})
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
    
    # VAD調整済み
    vad_iterator = VADIterator(
        vad_model, 
        threshold=0.5, 
        sampling_rate=16000, 
        min_silence_duration_ms=1000, 
        speech_pad_ms=50
    )

    audio_buffer = [] 
    is_speaking = False
    interruption_triggered = False 
    
    WINDOW_SIZE_SAMPLES = 512 
    SAMPLE_RATE = 16000
    CHECK_SPEAKER_SAMPLES = 12000
    
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
                        logger.info("🗣️ Speech START")
                        is_speaking = True
                        interruption_triggered = False 
                        audio_buffer = [window_np]
                        await websocket.send_json({"status": "processing", "message": "👂 聞いています..."})
                    
                    elif "end" in speech_dict:
                        logger.info("🤫 Speech END")
                        if is_speaking:
                            is_speaking = False
                            audio_buffer.append(window_np)
                            full_audio = np.concatenate(audio_buffer)
                            
                            if len(full_audio) / SAMPLE_RATE < 0.2:
                                logger.info("Noise detected")
                                await websocket.send_json({"status": "ignored", "message": "..."})
                            else:
                                await websocket.send_json({"status": "processing", "message": "🧠 AI思考中..."})
                                await process_voice_pipeline(full_audio, websocket, chat_history)
                            audio_buffer = [] 
                else:
                    if is_speaking:
                        audio_buffer.append(window_np)
                        
                        # バージイン判定
                        current_len = sum(len(c) for c in audio_buffer)
                        if not interruption_triggered and not NEXT_AUDIO_IS_REGISTRATION and current_len > CHECK_SPEAKER_SAMPLES:
                            temp_audio = np.concatenate(audio_buffer)
                            temp_tensor = torch.from_numpy(temp_audio).float().unsqueeze(0)
                            
                            # バージイン判定でも誰か特定する
                            is_verified, spk_id = await asyncio.to_thread(speaker_guard.identify_speaker, temp_tensor)
                            
                            if is_verified:
                                logger.info(f"⚡ [Barge-in] {spk_id} の声を検知！停止指示。")
                                await websocket.send_json({"status": "interrupt", "message": "🛑 音声停止"})
                                interruption_triggered = True

    except WebSocketDisconnect:
        logger.info("[WS] Disconnected")
    except Exception as e:
        logger.error(f"[WS ERROR] {e}", exc_info=True)
    finally:
        vad_iterator.reset_states()


# ---------------------------
# フロントエンド (色分け & Speaker ID対応)
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def get_root():
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device.width, initial-scale=1.0">
        <title>Multi-User Voice Chat</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; display: grid; place-items: center; min-height: 90vh; background: #202c33; color: #e9edef; margin: 0; }
            #container { background: #111b21; padding: 0; border-radius: 0; text-align: center; width: 100%; max-width: 600px; height: 100vh; display: flex; flex-direction: column; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
            @media (min-width: 600px) {
                #container { height: 90vh; border-radius: 12px; }
            }
            
            header { background: #202c33; padding: 15px; border-bottom: 1px solid #374045; font-weight: bold; font-size: 1.1rem; display: flex; justify-content: space-between; align-items: center; }
            
            #chat-box { 
                flex: 1; overflow-y: auto; padding: 20px; 
                background-image: url("https://user-images.githubusercontent.com/15075759/28719144-86dc0f70-73b1-11e7-911d-60d70fcded21.png");
                background-repeat: repeat;
                background-size: 400px;
                background-color: #0b141a;
            }

            .row { display: flex; width: 100%; margin-bottom: 8px; flex-direction: column; }
            .row.ai { align-items: flex-start; }
            .row.user { align-items: flex-end; }
            
            .speaker-name { font-size: 0.75rem; color: #8696a0; margin-bottom: 2px; margin-left: 5px; margin-right: 5px;}

            .bubble { 
                padding: 8px 12px; border-radius: 8px; max-width: 75%; 
                font-size: 0.95rem; line-height: 1.4; word-wrap: break-word;
                box-shadow: 0 1px 0.5px rgba(0,0,0,0.13);
            }
            .ai .bubble { background: #202c33; color: #e9edef; border-top-left-radius: 0; }
            
            /* ユーザーごとの色分け */
            /* User 0 (Owner) - Green */
            .user-type-0 .bubble { background: #005c4b; color: #e9edef; border-top-right-radius: 0; }
            /* User 1 (Member) - Blue */
            .user-type-1 .bubble { background: #0078d4; color: #fff; border-top-right-radius: 0; }
            /* User 2 (Member) - Purple */
            .user-type-2 .bubble { background: #6b63ff; color: #fff; border-top-right-radius: 0; }
            /* Default/Other - Grey */
            .user-type-unknown .bubble { background: #374045; color: #e9edef; border-top-right-radius: 0; }
            
            #controls { background: #202c33; padding: 15px; border-top: 1px solid #374045; }
            
            button { 
                padding: 10px 20px; border-radius: 24px; border: none; font-size: 1rem; cursor: pointer; margin: 0 5px; font-weight: bold; transition: opacity 0.2s;
            }
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
            let sourceInput;
            let isRecording = false;
            
            const btnStart = document.getElementById('btn-start');
            const btnStop = document.getElementById('btn-stop');
            const btnRegister = document.getElementById('btn-register');
            const statusDiv = document.getElementById('status');
            const chatBox = document.getElementById('chat-box');

            let audioQueue = [];
            let isPlaying = false;
            let currentSourceNode = null;
            let currentAiBubble = null;

            // --- UI Helper: 話者IDに基づいてクラスを付与 ---
            function logChat(role, text, speakerId = null) {
                const row = document.createElement('div');
                row.className = `row ${role}`;
                
                // ユーザーの場合のみ名前を表示
                if (role === 'user' && speakerId) {
                    const nameLabel = document.createElement('div');
                    nameLabel.className = 'speaker-name';
                    nameLabel.textContent = speakerId; // "User 0", "User 1"
                    row.appendChild(nameLabel);
                    
                    // 色分け用クラス (User 0 -> user-type-0)
                    const idNum = speakerId.replace('User ', '');
                    if (!isNaN(idNum)) {
                        row.classList.add(`user-type-${idNum}`);
                    } else {
                        row.classList.add('user-type-unknown');
                    }
                } else {
                    // AIの場合
                    const nameLabel = document.createElement('div');
                    nameLabel.className = 'speaker-name';
                    nameLabel.textContent = "AI Assistant";
                    row.appendChild(nameLabel);
                }

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
                    logChat('ai', "【システム】新しい方の声を登録します。マイクに向かって話しかけてください。");
                } catch(e) { console.error(e); }
            };

            async function startRecording() {
                try {
                    statusDiv.textContent = "サーバー接続中...";
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
                            }
                            if (data.status === 'interrupt') {
                                stopAudioPlayback();
                            }
                            if (data.status === 'system_info') {
                                // 登録完了などのシステム通知
                                logChat('ai', data.message);
                            }

                            // ★ 字幕表示 (話者IDを使用)
                            if (data.status === 'transcribed') {
                                logChat('user', data.question_text, data.speaker_id);
                            }

                            if (data.status === 'reply_chunk') {
                                if (!currentAiBubble) {
                                    currentAiBubble = logChat('ai', ''); 
                                }
                                currentAiBubble.textContent += data.text_chunk;
                                chatBox.scrollTop = chatBox.scrollHeight;
                            }
                            if (data.status === 'complete') {
                                if (!currentAiBubble && data.answer_text) {
                                    logChat('ai', data.answer_text);
                                }
                                currentAiBubble = null;
                                statusDiv.textContent = "🎙️ 準備OK";
                            }
                            if (data.status === 'ignored') {
                                statusDiv.textContent = data.message;
                            }
                        }
                    };
                    socket.onclose = () => stopRecording();
                } catch (e) {
                    console.error(e);
                }
            }

            async function initAudioStream() {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
                sourceInput = audioContext.createMediaStreamSource(stream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);
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
                    currentSourceNode = source;
                    source.onended = () => { currentSourceNode = null; isPlaying = false; processAudioQueue(); };
                    source.start(0);
                } catch(e) { isPlaying = false; currentSourceNode = null; }
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