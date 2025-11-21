# /workspace/new_new_main.py (修正版: バージイン対応)
import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
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
            # (C) WAV -> MP3
            audio_segment = AudioSegment.from_wav(part_path_abs)
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
# 4. フロントエンド (修正版 HTML/JS)
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def get_root():
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device.width, initial-scale=1.0">
        <title>AI Voice Talk (Barge-In)</title>
        
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

            #audioPlayback { margin-top: 1rem; display: none; }
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
            
            <div id="qa-display">
                </div>

            <div id="audioPlayback"></div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/bundle.min.js"></script>

        <script>
            // --- DOM要素 ---
            const startButton = document.getElementById('startButton');
            const stopButton = document.getElementById('stopButton');
            const statusDiv = document.getElementById('status');
            const vadStatusDiv = document.getElementById('vad-status');
            const qaDisplay = document.getElementById('qa-display');
            const audioPlayback = document.getElementById('audioPlayback');

            // --- グローバル変数 ---
            let ws;
            let vad; 
            let mediaStream; 
            
            // 状態管理フラグ
            let isSpeaking = false;     // ユーザーが話しているか
            let isAISpeaking = false;   // AIが喋っているか（再生中か）
            
            let audioQueue = [];        // 再生待ちの音声キュー
            let isPlaying = false;      // 現在音声を再生中か
            let currentAudio = null;    // 現在のAudioオブジェクト
            
            // バージイン制御用: 「前の回答」の残党を無視するためのフラグ
            let ignoreIncomingAudio = false; 

            // UI操作系
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
                return div;
            }

            function connectWebSocket() {
                const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
                ws = new WebSocket(wsProtocol + window.location.host + '/ws');
                ws.binaryType = 'arraybuffer';

                ws.onopen = () => {
                    console.log('WebSocket 接続');
                    statusDiv.textContent = '接続しました。マイクをONにしてください。';
                    startButton.disabled = false;
                };

                ws.onmessage = (event) => {
                    // (A) 音声データ受信
                    if (event.data instanceof ArrayBuffer) {
                        if (ignoreIncomingAudio) {
                            console.log("割り込み済みのため、古い音声パケットを破棄");
                            return;
                        }
                        const audioBlob = new Blob([event.data], { type: 'audio/mp3' });
                        audioQueue.push(audioBlob);
                        processAudioQueue();
                    } 
                    // (B) 制御メッセージ受信
                    else {
                        try {
                            const data = JSON.parse(event.data);
                            handleJsonMessage(data);
                        } catch (e) { console.error(e); }
                    }
                };

                ws.onclose = () => {
                    statusDiv.textContent = 'サーバー切断。リロード推奨。';
                    stopVAD(); 
                };
            }

            // JSONメッセージハンドリング
            let currentQuestionId = null;
            let currentAnswerId = null;

            function handleJsonMessage(data) {
                if (data.status === 'processing') {
                    // 新しいターン開始
                    statusDiv.textContent = data.message;
                    
                    // ★ここ重要: 新しい処理が始まったので、以前の割り込みフラグは解除
                    // ただし、AIが喋っている最中ならそれは「前のターン」なので止める必要があるが
                    // processingが来る＝ユーザーが話し終わって送信した後なので、
                    // 基本的にユーザー発話完了時点でinterruptAudioしてるはず。
                    
                } else if (data.status === 'transcribed') {
                    currentQuestionId = `q-${Date.now()}`;
                    appendBubble('user', data.question_text, currentQuestionId);
                    
                    currentAnswerId = `a-${Date.now()}`;
                    appendBubble('ai', '...', currentAnswerId);

                } else if (data.status === 'reply_chunk') {
                    if (ignoreIncomingAudio) return; // 無視モードならテキストも更新しない
                    
                    const div = document.getElementById(currentAnswerId);
                    if (div) {
                        if (div.textContent === '...') div.textContent = '';
                        div.textContent += data.text_chunk;
                        qaDisplay.scrollTop = qaDisplay.scrollHeight;
                    }

                } else if (data.status === 'ignored') {
                    statusDiv.textContent = "（音声を無視しました）";
                    if (currentAnswerId) {
                         const div = document.getElementById(currentAnswerId);
                         if(div) div.textContent = "(応答なし)";
                    }

                } else if (data.status === 'error') {
                    statusDiv.textContent = `エラー: ${data.message}`;
                }
            }

            // --- VAD & マイク設定 (Barge-Inの中核) ---
            async function setupVAD() {
                try {
                    startButton.disabled = true;
                    statusDiv.textContent = 'VAD準備中...';

                    while (!window.vad) await new Promise(r => setTimeout(r, 50));
                    
                    // ★重要1: エコーキャンセルを有効にする
                    mediaStream = await navigator.mediaDevices.getUserMedia({ 
                        audio: {
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true
                        } 
                    });
                    
                    vad = await window.vad.MicVAD.new({
                        stream: mediaStream,
                        positiveSpeechThreshold: 0.8,
                        minSpeechFrames: 2,
                        preSpeechPadFrames: 20,
                        
                        // ★重要2: 話し始めの検知 (割り込みトリガー)
                        onSpeechStart: () => {
                            isSpeaking = true;
                            vadStatusDiv.textContent = "🗣️ 感知中...";
                            
                            // もしAIが喋っていたり、再生待ちがある場合は「割り込み」とみなす
                            if (isPlaying || audioQueue.length > 0) {
                                console.log("⚡ 割り込み発生！ AIの音声を停止します");
                                interruptAudio();
                            }
                        },
                        
                        // ★重要3: 話し終わりの検知
                        onSpeechEnd: (audio) => {
                            isSpeaking = false;
                            vadStatusDiv.textContent = "📡 送信中...";
                            
                            // サーバーへ送信
                            if (ws && ws.readyState === WebSocket.OPEN) {
                                // 次のAI回答を受け入れる準備
                                ignoreIncomingAudio = false; 
                                sendAudioAsWav(audio);
                                statusDiv.textContent = 'AI思考中...';
                            }
                            
                            // ★以前あった vad.pause() は削除。常に聞き耳を立てる。
                        }
                    });

                    vad.start();
                    stopButton.disabled = false;
                    statusDiv.textContent = '🟢 準備完了。いつでも話しかけてください。';
                    vadStatusDiv.textContent = '👂 待機中';

                } catch (err) {
                    console.error('VADエラー:', err);
                    statusDiv.textContent = 'マイク初期化エラー。';
                    startButton.disabled = false;
                }
            }

            // --- 割り込み処理関数 ---
            function interruptAudio() {
                // 1. 再生中の音声を止める
                if (currentAudio) {
                    currentAudio.pause();
                    currentAudio = null;
                }
                
                // 2. 再生待ちキューを空にする
                audioQueue = [];
                isPlaying = false;
                isAISpeaking = false;
                
                // 3. これから届く「古い回答の続き」を無視するフラグを立てる
                ignoreIncomingAudio = true;
                
                statusDiv.textContent = '⛔ 中断しました。あなたの声を聞いています。';
                
                // UI上のフィードバック（オプション）
                if (currentAnswerId) {
                    const div = document.getElementById(currentAnswerId);
                    if (div) div.textContent += " (中断)";
                }
            }

            // --- 音声再生ロジック ---
            function processAudioQueue() {
                if (isPlaying) return;
                if (audioQueue.length === 0) return;
                
                const nextBlob = audioQueue.shift();
                playAudioBlob(nextBlob);
            }

            function playAudioBlob(blob) {
                isPlaying = true;
                isAISpeaking = true; // AI発話中フラグ
                statusDiv.textContent = '🔊 AI回答中...';

                const url = URL.createObjectURL(blob);
                currentAudio = new Audio(url);
                
                currentAudio.onended = () => {
                    isPlaying = false;
                    processAudioQueue(); // 次の文へ
                    
                    // 全部終わったら
                    if (audioQueue.length === 0) {
                        isAISpeaking = false;
                        statusDiv.textContent = '🟢 完了。次の質問をどうぞ。';
                    }
                };
                
                // エラーハンドリング
                currentAudio.onerror = () => {
                    isPlaying = false;
                    processAudioQueue();
                };

                currentAudio.play().catch(e => {
                    console.error("再生エラー:", e);
                    isPlaying = false;
                    processAudioQueue();
                });
            }

            // --- その他ユーティリティ ---
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
                vadStatusDiv.textContent = '';
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

            startButton.onclick = setupVAD;
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