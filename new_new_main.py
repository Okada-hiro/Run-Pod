# /workspace/new_new_main.py (完全修正版: Web Audio API対応)
import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketDisconnect
import os
import asyncio
import time
import logging 
import sys 
from pydub import AudioSegment
import io
import re
from speaker_filter import SpeakerGuard



# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] 
)
logger = logging.getLogger(__name__)

# --- 処理モジュールのインポート ---
try:
    from transcribe_func import whisper_text_only
    # supporter_generator を優先してインポート
    try:
        from supporter_generator import generate_answer_stream
    except ImportError:
        from new_answer_generator import generate_answer_stream

    from new_text_to_speech import synthesize_speech
except ImportError as e:
    print(f"[ERROR] 必要なモジュールが見つかりません: {e}")

# --- 設定 ---
PROCESSING_DIR = "incoming_audio" 
LANGUAGE = "ja"

# --- アプリケーション初期化 ---
app = FastAPI()
os.makedirs(PROCESSING_DIR, exist_ok=True)
app.mount(f"/download", StaticFiles(directory=PROCESSING_DIR), name="download")


# ---------------------------
# 1. 文ごとの処理関数
# ---------------------------
async def process_sentence(text: str, base_filename: str, index: int, websocket: WebSocket):
    logger.info(f"[STREAM] 文{index}: {text[:20]}...")
    
    # (A) 字幕送信
    try:
        await websocket.send_json({
            "status": "reply_chunk",
            "text_chunk": text
        })
    except Exception as e:
        logger.error(f"[STREAM ERROR] テキスト送信失敗: {e}")

    # (B) 音声合成
    part_filename = f"{base_filename}.part{index}.wav"
    part_path_abs = os.path.abspath(os.path.join(PROCESSING_DIR, part_filename))

    success = await asyncio.to_thread(
        synthesize_speech,
        text_to_speak=text,
        output_wav_path=part_path_abs
    )
    
    if success:
        try:
            # (C) 音声チェックと変換
            audio_segment = AudioSegment.from_wav(part_path_abs)
            duration_sec = len(audio_segment) / 1000.0
            
            if duration_sec < 0.1:
                logger.warning(f"⚠️ [AUDIO WARNING] 生成音声が短すぎます: {duration_sec}秒")
            else:
                logger.info(f"🔊 [AUDIO OK] 文{index} 長さ: {duration_sec}秒")

            mp3_buffer = io.BytesIO()
            audio_segment.export(mp3_buffer, format="mp3", bitrate="128k")
            audio_data = mp3_buffer.getvalue()

            # (D) 送信
            await websocket.send_bytes(audio_data)
        except Exception as e:
            logger.error(f"[STREAM ERROR] 音声変換・送信中にエラー: {e}", exc_info=True)


# ---------------------------
# 2. バックグラウンド処理 (メインフロー)
# ---------------------------
async def process_audio_file(audio_path: str, original_filename: str, websocket: WebSocket, chat_history: list):
    logger.info(f"[TASK START] ファイル処理開始: {original_filename}")
    # ★★★ ここに追加: 話者判定 ★★★
    # 本人じゃなければ即終了 (WhisperもLLMも回さない)
    is_owner = await asyncio.to_thread(speaker_guard.is_owner, audio_path)
    
    if not is_owner:
        logger.info("[TASK] 他人の声のため無視しました")
        await websocket.send_json({"status": "ignored", "message": "（他人の声を無視）"})
        return 
    # ★★★ ここまで ★★★
    try:
        # --- 文字起こし ---
        output_txt_path = os.path.join(PROCESSING_DIR, original_filename + ".txt")
        
        question_text = await asyncio.to_thread(
            whisper_text_only,
            audio_path, language=LANGUAGE, output_txt=output_txt_path
        )
        logger.info(f"[TASK] 文字起こし完了: {question_text}")

        await websocket.send_json({
            "status": "transcribed",
            "message": "...",
            "question_text": question_text
        })

        # --- ストリーミング回答 & パイプライン処理 ---
        text_buffer = ""
        sentence_count = 0
        full_answer_log = ""
        split_pattern = r'(?<=[。！？\n])'

        iterator = generate_answer_stream(question_text, history=chat_history)

        for chunk_text in iterator:
            text_buffer += chunk_text
            full_answer_log += chunk_text 

            # [SILENCE] チェック
            if full_answer_log.strip() == "[SILENCE]":
                logger.info("[TASK] SILENCE検出。応答をスキップします。")
                await websocket.send_json({"status": "ignored", "message": "（音声を無視しました）"})
                return

            # バッファ分割
            sentences = re.split(split_pattern, text_buffer)
            if len(sentences) > 1:
                for sent in sentences[:-1]:
                    if sent.strip():
                        sentence_count += 1
                        await process_sentence(sent, original_filename, sentence_count, websocket)
                text_buffer = sentences[-1]

        # 残り処理
        if text_buffer.strip():
            if text_buffer.strip() == "[SILENCE]":
                 await websocket.send_json({"status": "ignored", "message": "（音声を無視しました）"})
                 return

            sentence_count += 1
            await process_sentence(text_buffer, original_filename, sentence_count, websocket)
        
        # 履歴更新
        chat_history.append({"role": "user", "parts": [question_text]})
        chat_history.append({"role": "model", "parts": [full_answer_log]})
        
        await websocket.send_json({"status": "complete", "answer_text": full_answer_log})
        logger.info(f"[TASK END] 完了. 現在の履歴数: {len(chat_history)//2}ターン")

    except Exception as e:
        logger.error(f"[TASK ERROR] エラー: {e}", exc_info=True)
        try:
            await websocket.send_json({"status": "error", "message": f"エラー: {e}"})
        except WebSocketDisconnect:
            pass 

# ---------------------------
# 3. WebSocket エンドポイント
# ---------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WS] クライアント接続")
    
    chat_history = []
    
    try:
        while True:
            audio_data = await websocket.receive_bytes()
            audio_io = io.BytesIO(audio_data)
            
            temp_id = f"ws_{int(time.time())}"
            output_wav_filename = f"{temp_id}.wav"
            output_wav_path = os.path.join(PROCESSING_DIR, output_wav_filename)
            
            def convert_audio():
                try:
                    audio = AudioSegment.from_file(audio_io) 
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    audio.export(output_wav_path, format="wav")
                    return True
                except Exception as e:
                    logger.error(f"[WS ERROR] 変換失敗: {e}")
                    return False

            if not await asyncio.to_thread(convert_audio):
                await websocket.send_json({"status": "error", "message": "音声形式エラー"})
                continue
            
            # 処理開始通知
            await websocket.send_json({"status": "processing", "message": "認識中..."})

            asyncio.create_task(process_audio_file(
                output_wav_path, 
                output_wav_filename, 
                websocket,
                chat_history
            ))
            
    except WebSocketDisconnect:
        logger.info("[WS] 切断")
    except Exception as e:
        logger.error(f"[WS ERROR] : {e}", exc_info=True)
    finally:
        try:
            await websocket.close()
        except:
            pass


# ---------------------------
# 4. フロントエンド (修正版 HTML/JS - Web Audio API版)
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def get_root():
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device.width, initial-scale=1.0">
        <title>AI Voice Talk (Web Audio API)</title>
        
        <style>
            body { font-family: sans-serif; display: grid; place-items: center; min-height: 90vh; background: #f0f2f5; }
            #container { background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 8px 20px rgba(0,0,0,0.1); text-align: center; width: 90%; max-width: 600px; }
            
            button {
                font-size: 1rem; padding: 0.8rem 1.5rem; border: none; 
                border-radius: 25px; cursor: pointer; margin: 0.5rem; 
                color: white; transition: transform 0.1s, opacity 0.2s;
                font-weight: bold;
            }
            button:active { transform: scale(0.98); }
            button:disabled { background: #ccc !important; cursor: not-allowed; opacity: 0.6; transform: none; }
            
            #startButton { background: #007bff; }
            #stopButton { background: #6c757d; }

            #status { margin-top: 1.5rem; font-size: 1.1rem; color: #333; min-height: 1.5em; font-weight: bold; }
            #vad-status { font-size: 0.9rem; color: #666; height: 1.5em; margin-bottom: 10px;}
            
            #qa-display { 
                margin: 1rem auto 0 auto; text-align: left; width: 100%; 
                border-top: 2px solid #f0f0f0; padding-top: 1rem; 
                max-height: 400px; overflow-y: auto;
            }
            .bubble {
                padding: 10px 15px; border-radius: 15px; margin-bottom: 10px;
                line-height: 1.5; position: relative;
            }
            .user-bubble { background: #e7f5ff; color: #0056b3; margin-left: 20px; border-bottom-right-radius: 2px;}
            .user-bubble::before { content: 'あなた'; font-size: 0.7rem; position: absolute; top: -18px; right: 0; color: #999; }
            
            .ai-bubble { background: #f0fff4; color: #155724; margin-right: 20px; border-bottom-left-radius: 2px;}
            .ai-bubble::before { content: 'AI'; font-size: 0.7rem; position: absolute; top: -18px; left: 0; color: #999; }
        </style>
    </head>
    <body>
        <div id="container">
            <h1>AI Voice Talk ⚡</h1>
            <p>いつでも話しかけてください（割り込み可能）</p>
            
            <div>
                <button id="startButton">マイクON</button>
                <button id="stopButton" disabled>マイクOFF</button>
            </div>
            
            <div id="status">準備完了</div>
            <div id="vad-status">(待機中)</div>
            
            <div id="qa-display"></div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/bundle.min.js"></script>

        <script>
            // --- グローバル変数 ---
            const startButton = document.getElementById('startButton');
            const stopButton = document.getElementById('stopButton');
            const statusDiv = document.getElementById('status');
            const vadStatusDiv = document.getElementById('vad-status');
            const qaDisplay = document.getElementById('qa-display');

            let ws;
            let vad; 
            let mediaStream; 
            
            // Web Audio API用の変数
            let audioCtx = null;
            let currentSource = null; // 現在再生中のソース
            
            let isSpeaking = false;     
            let audioQueue = [];        
            let isPlaying = false;      
            let ignoreIncomingAudio = false; 

            // --- 1. WebSocket接続 ---
            function connectWebSocket() {
                const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
                ws = new WebSocket(wsProtocol + window.location.host + '/ws');
                ws.binaryType = 'arraybuffer';

                ws.onopen = () => {
                    console.log('WebSocket 接続');
                    statusDiv.textContent = '接続完了。マイクをONにしてください。';
                    startButton.disabled = false;
                };

                ws.onmessage = (event) => {
                    if (event.data instanceof ArrayBuffer) {
                        if (ignoreIncomingAudio) return;
                        const audioBlob = new Blob([event.data], { type: 'audio/mp3' });
                        audioQueue.push(audioBlob);
                        processAudioQueue();
                    } else {
                        try {
                            const data = JSON.parse(event.data);
                            handleJsonMessage(data);
                        } catch (e) { console.error(e); }
                    }
                };

                ws.onclose = () => {
                    statusDiv.textContent = '再接続してください。';
                    stopVAD(); 
                };
            }

            // --- 2. UI操作 ---
            let currentQuestionId = null;
            let currentAnswerId = null;

            function appendBubble(role, text, id) {
                let div = document.getElementById(id);
                if (!div) {
                    div = document.createElement('div');
                    div.id = id;
                    div.className = `bubble ${role === 'user' ? 'user-bubble' : 'ai-bubble'}`;
                    qaDisplay.appendChild(div);
                    qaDisplay.scrollTop = qaDisplay.scrollHeight;
                }
                div.textContent = text;
            }

            function handleJsonMessage(data) {
                if (data.status === 'processing') {
                    statusDiv.textContent = data.message;
                } else if (data.status === 'transcribed') {
                    currentQuestionId = `q-${Date.now()}`;
                    appendBubble('user', data.question_text, currentQuestionId);
                    currentAnswerId = `a-${Date.now()}`;
                    appendBubble('ai', '...', currentAnswerId);
                } else if (data.status === 'reply_chunk') {
                    if (ignoreIncomingAudio) return;
                    const div = document.getElementById(currentAnswerId);
                    if (div) {
                        if (div.textContent === '...') div.textContent = '';
                        div.textContent += data.text_chunk;
                        qaDisplay.scrollTop = qaDisplay.scrollHeight;
                    }
                } else if (data.status === 'ignored') {
                    statusDiv.textContent = "(音声を無視しました)";
                    const div = document.getElementById(currentAnswerId);
                    if(div) div.textContent = "(応答なし)";
                } else if (data.status === 'error') {
                    statusDiv.textContent = `エラー: ${data.message}`;
                }
            }

            // --- 3. Web Audio API 初期化 (最重要) ---
            // ボタンクリック時に1回だけ呼び出し、コンテキストを「再開(resume)」状態にする
            async function initAudioContext() {
                if (!audioCtx) {
                    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                }
                
                // ブラウザによってサスペンドされている場合は再開させる
                if (audioCtx.state === 'suspended') {
                    await audioCtx.resume();
                }

                // 無音を再生して確実にロック解除する
                const buffer = audioCtx.createBuffer(1, 1, 22050);
                const source = audioCtx.createBufferSource();
                source.buffer = buffer;
                source.connect(audioCtx.destination);
                source.start(0);
                
                console.log("🔊 AudioContext unlocked/resumed:", audioCtx.state);
            }

            // --- 4. 音声再生ロジック (Web Audio API版) ---
            function processAudioQueue() {
                if (isPlaying) return;
                if (audioQueue.length === 0) return;
                
                const nextBlob = audioQueue.shift();
                playAudioBlob(nextBlob);
            }

            async function playAudioBlob(blob) {
                if (!audioCtx) return; // 初期化前なら無視

                isPlaying = true;
                statusDiv.textContent = '🔊 AI回答中...';

                try {
                    // Blob -> ArrayBuffer -> AudioBuffer
                    const arrayBuffer = await blob.arrayBuffer();
                    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

                    const source = audioCtx.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(audioCtx.destination);
                    
                    currentSource = source; // 中断できるように保存

                    source.onended = () => {
                        isPlaying = false;
                        currentSource = null;
                        processAudioQueue();
                        
                        if (audioQueue.length === 0) {
                            statusDiv.textContent = '🟢 完了。次の質問をどうぞ。';
                        }
                    };

                    source.start(0);
                    
                } catch (e) {
                    console.error("再生エラー(decode/play):", e);
                    isPlaying = false;
                    processAudioQueue();
                }
            }

            // --- 5. 割り込み処理 ---
            function interruptAudio() {
                // 再生中のソースを停止
                if (currentSource) {
                    try { currentSource.stop(); } catch(e){}
                    currentSource = null;
                }
                
                audioQueue = [];
                isPlaying = false;
                ignoreIncomingAudio = true;
                statusDiv.textContent = '⛔ 中断。あなたの声を聞いています。';
                
                if (currentAnswerId) {
                    const div = document.getElementById(currentAnswerId);
                    if (div) div.textContent += " (中断)";
                }
            }

            // --- 6. VAD & マイク設定 ---
            async function setupVAD() {
                try {
                    startButton.disabled = true;
                    statusDiv.textContent = 'マイク準備中...';

                    while (!window.vad) await new Promise(r => setTimeout(r, 50));
                    
                    mediaStream = await navigator.mediaDevices.getUserMedia({ 
                        audio: {
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true
                        } 
                    });
                    
                    vad = await window.vad.MicVAD.new({
                        stream: mediaStream,
                        positiveSpeechThreshold: 0.9, // 誤検知防止で少し高め
                        minSpeechFrames: 4,
                        preSpeechPadFrames: 20,
                        onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/",
                        baseAssetPath: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/",
                        
                        onSpeechStart: () => {
                            isSpeaking = true;
                            vadStatusDiv.textContent = "🗣️ 感知中...";
                            if (isPlaying || audioQueue.length > 0) {
                                interruptAudio(); // 割り込み
                            }
                        },
                        
                        onSpeechEnd: (audio) => {
                            isSpeaking = false;
                            vadStatusDiv.textContent = "📡 送信中...";
                            if (ws && ws.readyState === WebSocket.OPEN) {
                                ignoreIncomingAudio = false; 
                                sendAudioAsWav(audio);
                                statusDiv.textContent = 'AI思考中...';
                            }
                        }
                    });

                    vad.start();
                    stopButton.disabled = false;
                    statusDiv.textContent = '🟢 準備完了。';
                    vadStatusDiv.textContent = '👂 待機中';

                } catch (err) {
                    console.error('VAD/Mic エラー:', err);
                    statusDiv.textContent = 'マイク初期化失敗。';
                    startButton.disabled = false;
                }
            }

            // --- その他 ---
            function sendAudioAsWav(float32Array) {
                const wavBuffer = encodeWAV(float32Array, 16000); 
                ws.send(wavBuffer);
            }
            function stopVAD() {
                vad?.destroy(); 
                vad = null;
                mediaStream?.getTracks().forEach(track => track.stop());
                startButton.disabled = false;
                stopButton.disabled = true;
                statusDiv.textContent = '停止中';
            }
            function encodeWAV(samples, sampleRate) {
                const buffer = new ArrayBuffer(44 + samples.length * 2);
                const view = new DataView(buffer);
                writeString(view, 0, 'RIFF');
                view.setUint32(4, 36 + samples.length * 2, true);
                writeString(view, 8, 'WAVE');
                writeString(view, 12, 'fmt ');
                view.setUint32(16, 16, true);
                view.setUint16(20, 1, true); 
                view.setUint16(22, 1, true); 
                view.setUint32(24, sampleRate, true);
                view.setUint32(28, sampleRate * 2, true);
                view.setUint16(32, 2, true);
                view.setUint16(34, 16, true);
                writeString(view, 36, 'data');
                view.setUint32(40, samples.length * 2, true);
                floatTo16BitPCM(view, 44, samples);
                return view;
            }
            function writeString(view, offset, string) {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            }
            function floatTo16BitPCM(output, offset, input) {
                for (let i = 0; i < input.length; i++, offset += 2) {
                    let s = Math.max(-1, Math.min(1, input[i]));
                    s = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    output.setInt16(offset, s, true);
                }
            }

            // ★ボタンクリックでAudioContext初期化とVAD起動を同時に行う
            startButton.onclick = async () => {
                await initAudioContext(); // オーディオエンジンの起動
                await setupVAD();         // マイクの起動
            };
            
            stopButton.onclick = stopVAD;
            window.onload = connectWebSocket;
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"サーバーを http://0.0.0.0:{port} で起動します。")
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None)