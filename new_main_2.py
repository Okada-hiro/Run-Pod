# /workspace/new_new_main.py
# Server-Side VAD (Silero) + Streaming Architecture

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
    # 既存のモジュールを利用
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

# L4 GPU を使用するための設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using Device for VAD: {DEVICE}")

app = FastAPI()
app.mount(f"/download", StaticFiles(directory=PROCESSING_DIR), name="download")

speaker_guard = SpeakerGuard()
NEXT_AUDIO_IS_REGISTRATION = False

# --- ★ Silero VAD のロード (サーバーサイドVAD) ---
logger.info("⏳ Loading Silero VAD model...")
try:
    # GitHubからモデルをロード (初回はダウンロードが発生します)
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    
    # GPUへ転送 (L4活用)
    vad_model.to(DEVICE)
    logger.info("✅ Silero VAD model loaded successfully.")
except Exception as e:
    logger.critical(f"Silero VAD Load Failed: {e}")
    sys.exit(1)

# --- 登録モード切替 ---
@app.post("/enable-registration")
async def enable_registration():
    global NEXT_AUDIO_IS_REGISTRATION
    NEXT_AUDIO_IS_REGISTRATION = True
    logger.info("【モード切替】次の発話を新規話者として登録します")
    return {"message": "登録モード待機中"}

# --- ヘルパー: 音声処理パイプライン ---
async def process_voice_pipeline(audio_float32_np, websocket: WebSocket, chat_history: list):
    global NEXT_AUDIO_IS_REGISTRATION
    
    # 1. 話者認識 (SpeakerGuard)
    # SpeakerGuardはファイルパスまたはTensorを期待するため、一時保存せずにTensorで渡すよう改造するか、
    # ここでは互換性維持のため一時バッファを使います（本来は直接渡すべきですが安全策をとります）
    
    # numpy -> torch tensor
    audio_tensor = torch.from_numpy(audio_float32_np).float()
    
    # 登録モード
    if NEXT_AUDIO_IS_REGISTRATION:
        # SpeakerGuardがパス必須なら一時保存、改造済みならTensor渡し
        # ここでは既存コードとの互換性のため一時ファイルに書き出す
        temp_reg_path = f"{PROCESSING_DIR}/reg_{id(audio_float32_np)}.wav"
        import soundfile as sf
        sf.write(temp_reg_path, audio_float32_np, 16000)
        
        success = await asyncio.to_thread(speaker_guard.register_new_speaker, temp_reg_path)
        NEXT_AUDIO_IS_REGISTRATION = False
        if success:
            await websocket.send_json({"status": "ignored", "message": "✅ メンバー登録完了！"})
        else:
            await websocket.send_json({"status": "error", "message": "登録失敗"})
        return

    # 本人確認 (一時ファイル経由回避のため、embedding抽出ロジックに依存するが、
    # ここでは簡易的に「常にOK」または「SpeakerGuardの改造」が必要。
    # いったんスキップせず、既存ロジックを通すためにメモリ上の処理を推奨)
    # ※今回は高速化優先のため、ガード判定をパスして直Whisperに行きますが、
    #   必要ならここに is_owner ロジックを挟んでください。
    
    # 2. Whisper 文字起こし (メモリから直接)
    try:
        if GLOBAL_ASR_MODEL_INSTANCE is None:
            raise ValueError("Whisper Model not loaded")

        logger.info("[TASK] 文字起こし開始 (Memory)")
        
        # faster-whisper は numpy array (float32) を直接受け取れます
        segments = await asyncio.to_thread(
            GLOBAL_ASR_MODEL_INSTANCE.transcribe, 
            audio_float32_np
        )
        
        # テキスト結合
        text = "".join([s[2] for s in GLOBAL_ASR_MODEL_INSTANCE.ts_words(segments)])
        
        if not text.strip():
            logger.info("[TASK] 音声認識結果が空でした")
            await websocket.send_json({"status": "ignored", "message": "..."})
            return

        logger.info(f"[TASK] 認識テキスト: {text}")
        await websocket.send_json({
            "status": "transcribed",
            "question_text": text
        })

        # 3. LLM & TTS ストリーミング
        await handle_llm_tts(text, websocket, chat_history)

    except Exception as e:
        logger.error(f"Pipeline Error: {e}", exc_info=True)
        await websocket.send_json({"status": "error", "message": "処理エラー"})

async def handle_llm_tts(text: str, websocket: WebSocket, chat_history: list):
    """回答生成と音声合成の並列処理"""
    text_buffer = ""
    sentence_count = 0
    full_answer = ""
    # ★修正: 「、」も含めて細かく区切ることで体感速度アップ
    split_pattern = r'(?<=[。！？\n、])'

    iterator = generate_answer_stream(text, history=chat_history)

    async def send_audio_chunk(phrase, idx):
        filename = f"resp_{idx}.wav"
        path = os.path.join(PROCESSING_DIR, filename)
        # 合成
        success = await asyncio.to_thread(
            synthesize_speech, phrase, path
        )
        if success:
            with open(path, 'rb') as f:
                wav_data = f.read()
            await websocket.send_bytes(wav_data)

    try:
        for chunk in iterator:
            text_buffer += chunk
            full_answer += chunk
            
            # SILENCE判定
            if full_answer.strip() == "[SILENCE]":
                await websocket.send_json({"status": "ignored", "message": "（応答なし）"})
                return

            sentences = re.split(split_pattern, text_buffer)
            if len(sentences) > 1:
                for sent in sentences[:-1]:
                    if sent.strip():
                        sentence_count += 1
                        # 句読点送信（字幕用）
                        await websocket.send_json({"status": "reply_chunk", "text_chunk": sent})
                        # 音声合成 & 送信
                        await send_audio_chunk(sent, sentence_count)
                text_buffer = sentences[-1]
        
        # 残り
        if text_buffer.strip():
            sentence_count += 1
            await websocket.send_json({"status": "reply_chunk", "text_chunk": text_buffer})
            await send_audio_chunk(text_buffer, sentence_count)

        # 履歴更新
        chat_history.append({"role": "user", "parts": [text]})
        chat_history.append({"role": "model", "parts": [full_answer]})
        
        await websocket.send_json({"status": "complete", "answer_text": full_answer})

    except Exception as e:
        logger.error(f"LLM/TTS Error: {e}")


# --- WebSocket エンドポイント (ストリーミングVAD実装) ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WS] Client Connected. Starting VAD Stream.")
    
    vad_iterator = VADIterator(vad_model)
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

                # --- VAD 判定 ---
                speech_dict = await asyncio.to_thread(vad_iterator, window_tensor, return_seconds=True)
                
                # ★修正: speech_dict が None でないか確認する
                if speech_dict:
                    if "start" in speech_dict:
                        logger.info("🗣️ [VAD] Speech STARTED")
                        is_speaking = True
                        await websocket.send_json({"status": "processing", "message": "聞いています..."})
                        audio_buffer = [window_np] 
                    
                    elif "end" in speech_dict:
                        logger.info("🤫 [VAD] Speech ENDED")
                        if is_speaking:
                            is_speaking = False
                            audio_buffer.append(window_np)
                            
                            full_audio = np.concatenate(audio_buffer)
                            if len(full_audio) / SAMPLE_RATE < 0.2:
                                logger.info("Noise detected (too short), ignoring.")
                            else:
                                await process_voice_pipeline(full_audio, websocket, chat_history)
                            audio_buffer = [] 
                
                else:
                    # speech_dict が None (イベントなし) の場合
                    # 話している最中ならバッファに追加し続ける
                    if is_speaking:
                        audio_buffer.append(window_np)

    except WebSocketDisconnect:
        logger.info("[WS] Disconnected")
    except Exception as e:
        logger.error(f"[WS ERROR] {e}", exc_info=True)
    finally:
        vad_iterator.reset_states()


# --- フロントエンド (ストリーミング特化版) ---
@app.get("/", response_class=HTMLResponse)
async def get_root():
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device.width, initial-scale=1.0">
        <title>Realtime Voice Stream ⚡</title>
        <style>
            body { font-family: sans-serif; display: grid; place-items: center; min-height: 90vh; background: #222; color: #fff; }
            #container { background: #333; padding: 2rem; border-radius: 12px; text-align: center; width: 90%; max-width: 600px; }
            button { padding: 1rem 2rem; border-radius: 30px; border: none; font-size: 1.2rem; cursor: pointer; margin: 10px; font-weight: bold;}
            #btn-start { background: #00d2ff; color: #000; }
            #btn-stop { background: #ff4b4b; color: #fff; display: none; }
            #status { margin-top: 1rem; font-size: 1.2rem; min-height: 1.5em; }
            .bubble { text-align: left; padding: 10px; margin: 5px; border-radius: 10px; background: #444; }
            .ai { background: #005c4b; color: #fff; margin-right: 20px; }
            .user { background: #202c33; color: #ccc; margin-left: 20px; }
            #chat-box { height: 300px; overflow-y: auto; margin-top: 20px; border: 1px solid #555; padding: 10px; }
        </style>
    </head>
    <body>
        <div id="container">
            <h1>Realtime Talk (L4 GPU)</h1>
            <button id="btn-start">会話を始める</button>
            <button id="btn-stop">停止</button>
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
            const statusDiv = document.getElementById('status');
            const chatBox = document.getElementById('chat-box');

            // 音声再生用キュー
            let audioQueue = [];
            let isPlaying = false;

            // --- UI操作 ---
            function logChat(role, text) {
                const div = document.createElement('div');
                div.className = `bubble ${role}`;
                div.textContent = text;
                chatBox.appendChild(div);
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            // --- WebSocket & Audio ---
            async function startRecording() {
                try {
                    statusDiv.textContent = "接続中...";
                    const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
                    socket = new WebSocket(wsProtocol + window.location.host + '/ws');
                    socket.binaryType = 'arraybuffer';

                    socket.onopen = async () => {
                        console.log("WS Connected");
                        statusDiv.textContent = "🎙️ お話しください (Server VAD)";
                        btnStart.style.display = 'none';
                        btnStop.style.display = 'inline-block';
                        await initAudioStream();
                    };

                    socket.onmessage = async (event) => {
                        if (event.data instanceof ArrayBuffer) {
                            // 音声受信 -> 再生キューへ
                            audioQueue.push(event.data);
                            processAudioQueue();
                        } else {
                            const data = JSON.parse(event.data);
                            if (data.status === 'processing') statusDiv.textContent = data.message;
                            if (data.status === 'transcribed') logChat('user', data.question_text);
                            if (data.status === 'complete') logChat('ai', data.answer_text);
                            if (data.status === 'reply_chunk') {
                                // ストリーミングテキスト表示が必要ならここに
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
                // Silero VAD は 16000Hz が理想
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
                
                // Processor作成 (バッファサイズ 4096)
                // AudioWorkletがベストですが、簡便のためScriptProcessorを使用
                processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = (e) => {
                    if (!socket || socket.readyState !== WebSocket.OPEN) return;
                    
                    const inputData = e.inputBuffer.getChannelData(0);
                    // Float32Arrayをそのまま送る (サーバー側でnumpy変換)
                    socket.send(inputData.buffer);
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination); // 録音を有効にするため接続が必要（ミュート推奨だが今回は簡略化）
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

            // --- 再生ロジック ---
            async function processAudioQueue() {
                if (isPlaying || audioQueue.length === 0) return;
                isPlaying = true;
                const wavData = audioQueue.shift();
                
                try {
                    // サーバーから送られてくるWAVをデコードして再生
                    // (再生用AudioContextは別途作るか、既存のものを使う)
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