#new_main_2.pyからの進化系 割り込みができる

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
    # 1. 話者認識 (最終確認)
    # ---------------------------
    # ※ 割り込み判定ですでにOKが出ている場合も多いが、念のため全データで最終確認
    is_allowed = await asyncio.to_thread(speaker_guard.verify_tensor, voice_tensor)

    if not is_allowed:
        logger.info("[Access Denied] 登録されていない話者です（最終判定）。")
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
    split_pattern = r'(?<=[。！？\n、])'

    # 生成開始
    iterator = generate_answer_stream(text, history=chat_history)

    async def send_audio_chunk(phrase, idx):
        filename = f"resp_{idx}.wav"
        path = os.path.join(PROCESSING_DIR, filename)
        success = await asyncio.to_thread(synthesize_speech, phrase, path)
        if success:
            with open(path, 'rb') as f:
                wav_data = f.read()
            # 音声データを送信
            try:
                await websocket.send_bytes(wav_data)
            except RuntimeError:
                # 接続が切れている場合など
                pass

    try:
        for chunk in iterator:
            # ★ ここに「新しい割り込み」があった場合のキャンセル処理を入れることも可能だが、
            # 今回はWebSocketループ側で管理し、クライアントが再生を止める方式を採用する。
            
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
                        # 字幕送信
                        await websocket.send_json({"status": "reply_chunk", "text_chunk": sent})
                        # 音声送信
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
# WebSocket エンドポイント (Barge-in対応)
# ---------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WS] Client Connected.")
    
    vad_iterator = VADIterator(
    vad_model, 
    threshold=0.5, 
    sampling_rate=16000, 
    min_silence_duration_ms=500, 
    speech_pad_ms=50
)
    audio_buffer = [] 
    is_speaking = False
    interruption_triggered = False # 今回の発話ですでに割り込み指示を出したか
    
    # 設定
    WINDOW_SIZE_SAMPLES = 512 
    SAMPLE_RATE = 16000
    CHECK_SPEAKER_SAMPLES = 12000 # 約0.75秒溜まったら話者チェックする
    
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
                        interruption_triggered = False # リセット
                        audio_buffer = [window_np]
                        # UI更新: 聞き取り開始
                        await websocket.send_json({"status": "processing", "message": "👂 聞いています..."})
                    
                    elif "end" in speech_dict:
                        logger.info("🤫 Speech END")
                        if is_speaking:
                            is_speaking = False
                            audio_buffer.append(window_np)
                            
                            full_audio = np.concatenate(audio_buffer)
                            
                            # 短すぎるノイズは無視
                            if len(full_audio) / SAMPLE_RATE < 0.2:
                                logger.info("Noise detected (too short)")
                                await websocket.send_json({"status": "ignored", "message": "..."})
                            else:
                                await websocket.send_json({"status": "processing", "message": "🧠 AI思考中..."})
                                # パイプライン実行（非同期タスクとして投げると並列処理になるが、
                                # ここでは順次処理でチャットの整合性を保つ）
                                await process_voice_pipeline(full_audio, websocket, chat_history)
                            
                            audio_buffer = [] 
                
                else:
                    if is_speaking:
                        audio_buffer.append(window_np)
                        
                        # --- ★★★ バージイン（割り込み）判定ロジック ★★★ ---
                        # まだ割り込み指示を出しておらず、かつ一定量（0.75秒分など）音声が溜まった場合
                        current_len = sum(len(c) for c in audio_buffer)
                        
                        if not interruption_triggered and not NEXT_AUDIO_IS_REGISTRATION and current_len > CHECK_SPEAKER_SAMPLES:
                            # 暫定バッファを結合してチェック
                            temp_audio = np.concatenate(audio_buffer)
                            temp_tensor = torch.from_numpy(temp_audio).float().unsqueeze(0)
                            
                            # 話者チェック (SpeakerGuard)
                            is_verified = await asyncio.to_thread(speaker_guard.verify_tensor, temp_tensor)
                            
                            if is_verified:
                                logger.info("⚡ [Barge-in] 本人の声を検知！再生停止指示を送信します。")
                                # クライアントに「音声停止」を指示
                                await websocket.send_json({"status": "interrupt", "message": "🛑 音声停止"})
                                interruption_triggered = True
                            else:
                                # 本人ではない(雑音の可能性) -> 割り込み指示を送らない（無視して再生継続）
                                # ※ただし、最終的にSpeech ENDまで行ったら再度チェックされてブロックされる
                                pass

    except WebSocketDisconnect:
        logger.info("[WS] Disconnected")
    except Exception as e:
        logger.error(f"[WS ERROR] {e}", exc_info=True)
    finally:
        vad_iterator.reset_states()


# ---------------------------
# フロントエンド (字幕修正 & LINE風UI & 割り込み対応)
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

            .row { display: flex; width: 100%; margin-bottom: 8px; }
            .row.ai { justify-content: flex-start; }
            .row.user { justify-content: flex-end; }
            
            .bubble { 
                padding: 8px 12px; border-radius: 8px; max-width: 75%; 
                font-size: 0.95rem; line-height: 1.4; position: relative; word-wrap: break-word;
                box-shadow: 0 1px 0.5px rgba(0,0,0,0.13);
            }
            .ai .bubble { background: #202c33; color: #e9edef; border-top-left-radius: 0; }
            .user .bubble { background: #005c4b; color: #e9edef; border-top-right-radius: 0; }
            
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
                <span>AI Agent</span>
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

            // --- 再生管理用変数 ---
            let audioQueue = [];
            let isPlaying = false;
            let currentSourceNode = null; // 現在再生中のAudioBufferSourceNode
            let currentAiBubble = null;   // 字幕ストリーミング用DOM

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
                return bubble; 
            }

            // --- メンバー登録 ---
            btnRegister.onclick = async () => {
                try {
                    await fetch('/enable-registration', { method: 'POST' });
                    statusDiv.textContent = "🆕 新規メンバー登録モード";
                    statusDiv.style.color = "#00a884";
                    logChat('ai', "【システム】次に話す人の声を登録します。何か話しかけてください。");
                } catch(e) { console.error(e); }
            };

            // --- WebSocket & Audio ---
            async function startRecording() {
                try {
                    statusDiv.textContent = "サーバー接続中...";
                    const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
                    socket = new WebSocket(wsProtocol + window.location.host + '/ws');
                    socket.binaryType = 'arraybuffer';

                    socket.onopen = async () => {
                        console.log("WS Connected");
                        statusDiv.textContent = "🎙️ 準備OK。話しかけてください";
                        statusDiv.style.color = "#e9edef";
                        
                        btnStart.style.display = 'none';
                        btnStop.style.display = 'inline-block';
                        
                        await initAudioStream();
                    };

                    socket.onmessage = async (event) => {
                        if (event.data instanceof ArrayBuffer) {
                            // 音声データ受信 -> キューに追加して再生処理へ
                            audioQueue.push(event.data);
                            processAudioQueue();
                        } else {
                            const data = JSON.parse(event.data);
                            
                            // ステータス更新
                            if (data.status === 'processing') {
                                statusDiv.textContent = data.message;
                                if (data.message.includes("聞いて")) statusDiv.style.color = "#ef5350"; 
                                else if (data.message.includes("思考中")) statusDiv.style.color = "#00a884";
                            }
                            
                            // ★★★ 割り込み (Interrupt) 処理 ★★★
                            if (data.status === 'interrupt') {
                                console.log("🛑 Interrupt Signal Received!");
                                stopAudioPlayback(); // 音声を即時停止
                                // 字幕(bubble)は維持する（ユーザーの希望）
                            }

                            // ユーザーの発言（即字幕表示）
                            if (data.status === 'transcribed') {
                                logChat('user', data.question_text);
                            }

                            // AIの回答（字幕ストリーミング）
                            if (data.status === 'reply_chunk') {
                                if (!currentAiBubble) {
                                    currentAiBubble = logChat('ai', ''); 
                                }
                                currentAiBubble.textContent += data.text_chunk;
                                chatBox.scrollTop = chatBox.scrollHeight;
                            }

                            // 完了時
                            if (data.status === 'complete') {
                                if (!currentAiBubble && data.answer_text) {
                                    logChat('ai', data.answer_text);
                                }
                                currentAiBubble = null; 
                                statusDiv.textContent = "🎙️ 準備OK。話しかけてください";
                                statusDiv.style.color = "#e9edef";
                            }

                            if (data.status === 'ignored') {
                                statusDiv.textContent = data.message;
                            }
                        }
                    };

                    socket.onclose = () => stopRecording();

                } catch (e) {
                    console.error(e);
                    statusDiv.textContent = "接続エラー";
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
                
                sourceInput = audioContext.createMediaStreamSource(stream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = (e) => {
                    if (!socket || socket.readyState !== WebSocket.OPEN) return;
                    const inputData = e.inputBuffer.getChannelData(0);
                    socket.send(inputData.buffer);
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

            // --- 再生ロジック (割り込み対応版) ---
            
            // 音声を即時停止し、キューをクリアする関数
            function stopAudioPlayback() {
                // 1. 再生中なら止める
                if (currentSourceNode) {
                    try {
                        currentSourceNode.stop();
                    } catch(e) {
                        // すでに止まっている場合などは無視
                    }
                    currentSourceNode = null;
                }
                // 2. 待機中の音声を破棄
                audioQueue = [];
                isPlaying = false;
                console.log("Audio Playback Cleared.");
            }

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
                    
                    // 現在のソースとして保持（stop用）
                    currentSourceNode = source;
                    
                    source.onended = () => {
                        // 正常終了した場合のみ次へ（stopされた場合はここは呼ばれるが、queueは空になっているはず）
                        currentSourceNode = null;
                        isPlaying = false;
                        processAudioQueue();
                    };
                    source.start(0);
                } catch(e) {
                    console.error("再生エラー", e);
                    isPlaying = false;
                    currentSourceNode = null;
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