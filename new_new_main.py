# /workspace/new_new_main.py
import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
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

# --- 既存の処理モジュールをインポート ---
try:
    from transcribe_func import whisper_text_only
    # ファイル名が new_answer_generator ではなく supporter_generator である場合は適宜変更してください
    # ここではユーザーコードのファイル名に従い supporter_generator と仮定して修正します
    # もしファイル名が違う場合は import元を変更してください
    try:
        from supporter_generator import generate_answer, generate_answer_stream
    except ImportError:
        from new_answer_generator import generate_answer, generate_answer_stream

    from new_text_to_speech import synthesize_speech
except ImportError as e:
    print(f"[ERROR] 必要なモジュールが見つかりません: {e}")
    # 動作確認のため一時的にpassしないようにexitします
    # exit(1) 

# --- 設定 ---
PROCESSING_DIR = "incoming_audio" 
LANGUAGE = "ja"

# --- アプリケーション初期化 ---
app = FastAPI()
os.makedirs(PROCESSING_DIR, exist_ok=True)
app.mount(f"/download", StaticFiles(directory=PROCESSING_DIR), name="download")


# ---------------------------
# 1. 文ごとの処理関数 (字幕送信 -> 音声合成 -> 音声送信)
# ---------------------------
async def process_sentence(text: str, base_filename: str, index: int, websocket: WebSocket):
    logger.info(f"[STREAM] 文{index}: {text[:20]}...")
    
    # (A) 先に字幕テキストを送る
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
            # (C) WAV -> MP3 変換 (軽量化)
            audio_segment = AudioSegment.from_wav(part_path_abs)
            mp3_buffer = io.BytesIO()
            audio_segment.export(mp3_buffer, format="mp3", bitrate="128k")
            audio_data = mp3_buffer.getvalue()

            # (D) バイナリ音声送信
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

        # フロントエンドに「聞き取りました」と通知
        await websocket.send_json({
            "status": "transcribed",
            "message": "...", # 最初は何も表示しない、または「思考中」
            "question_text": question_text
        })

        # --- ストリーミング回答 & パイプライン処理 ---
        logger.info(f"[TASK] ストリーミング処理開始...")

        text_buffer = ""
        sentence_count = 0
        full_answer_log = ""
        
        # 句読点(。！？)または改行で分割する正規表現
        split_pattern = r'(?<=[。！？\n])'

        # ★ ここで現在の履歴を渡して生成
        # LLMからストリーミングで文字を受け取る
        # ★ モデル名を修正 (gemini-2.5-flash-lite)
        iterator = generate_answer_stream(question_text, model="gemini-2.5-flash-lite", history=chat_history)

        for chunk_text in iterator:
            text_buffer += chunk_text
            full_answer_log += chunk_text 

            # ★ [SILENCE] 判定ロジック
            # ストリームの途中でも、全体が [SILENCE] になりそうなら検知したい
            # ただし、普通に話しているときも文頭で判定しないように注意が必要
            # ここでは「バッファ全体が [SILENCE] と一致したら即停止」とします
            if full_answer_log.strip() == "[SILENCE]":
                logger.info("[TASK] SILENCE検出。応答をスキップします。")
                await websocket.send_json({"status": "ignored", "message": "（音声を無視しました）"})
                return  # 処理終了

            # バッファ分割チェック
            sentences = re.split(split_pattern, text_buffer)

            if len(sentences) > 1:
                # 確定した文(最後以外)を処理
                for sent in sentences[:-1]:
                    if sent.strip():
                        sentence_count += 1
                        await process_sentence(sent, original_filename, sentence_count, websocket)
                
                # 残りをバッファに戻す
                text_buffer = sentences[-1]

        # ループ終了後の残りかす処理
        if text_buffer.strip():
            # 最後の最後で [SILENCE] だった場合（ストリームが一括で来た場合など）
            if text_buffer.strip() == "[SILENCE]":
                 logger.info("[TASK] SILENCE検出(末尾)。応答をスキップします。")
                 await websocket.send_json({"status": "ignored", "message": "（音声を無視しました）"})
                 return

            sentence_count += 1
            await process_sentence(text_buffer, original_filename, sentence_count, websocket)

        # --- ★ ここが重要：回答が完了したら履歴に追加 ---
        # Geminiの履歴フォーマットに合わせて追加
        chat_history.append({"role": "user", "parts": [question_text]})
        chat_history.append({"role": "model", "parts": [full_answer_log]})
        # 完了通知
        await websocket.send_json({"status": "complete", "answer_text": full_answer_log})
        logger.info(f"[TASK END] ストリーミング完了: {original_filename}")

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
    logger.info("[WS] クライアントが接続しました。")
    # ★ ここで接続ごとの会話履歴を初期化
    chat_history = []
    try:
        while True:
            audio_data = await websocket.receive_bytes()
            audio_io = io.BytesIO(audio_data)
            
            temp_id = f"ws_{int(time.time())}"
            output_wav_filename = f"{temp_id}.wav"
            output_wav_path = os.path.join(PROCESSING_DIR, output_wav_filename)
            
            # pydubでフォーマット自動判別してWAV化
            def convert_audio():
                try:
                    audio = AudioSegment.from_file(audio_io) 
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    audio.export(output_wav_path, format="wav")
                    return True
                except Exception as e:
                    logger.error(f"[WS ERROR] pydub 変換失敗: {e}")
                    return False

            if not await asyncio.to_thread(convert_audio):
                await websocket.send_json({"status": "error", "message": "音声形式の変換に失敗しました。"})
                continue
            
            # 処理開始通知
            await websocket.send_json({"status": "processing", "message": "音声を認識しています..."})

            asyncio.create_task(process_audio_file(
                output_wav_path, 
                output_wav_filename, 
                websocket,
                chat_history  # <--- これが必要です！
            ))
            
    except WebSocketDisconnect:
        logger.info("[WS] クライアントが切断しました。")
    except Exception as e:
        logger.error(f"[WS ERROR] WebSocketエラー: {e}", exc_info=True)
    finally:
        try:
            await websocket.close()
        except:
            pass


# ---------------------------
# 4. フロントエンド (HTML/JS)
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def get_root():
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device.width, initial-scale=1.0">
        <title>AI Voice Talk (Silence Aware)</title>
        
        <style>
            body { font-family: sans-serif; display: grid; place-items: center; min-height: 90vh; background: #f4f4f4; }
            #container { background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; width: 90%; max-width: 600px; }
            
            button {
                font-size: 1rem; padding: 0.8rem 1.5rem; border: none; 
                border-radius: 5px; cursor: pointer; margin: 0.5rem; 
                color: white; transition: opacity 0.2s;
            }
            button:disabled { background: #ccc !important; cursor: not-allowed; opacity: 0.6; }
            
            #startButton { background: #007bff; font-size: 1.2rem; }
            #stopButton { background: #6c757d; }
            #interruptButton { background: #dc3545; display: inline-block; }

            #status { margin-top: 1.5rem; font-size: 1.1rem; color: #333; min-height: 2em; font-weight: bold; }
            #vad-status { font-size: 0.9rem; color: #666; height: 1.5em; }
            
            #qa-display { margin: 1.5rem auto 0 auto; text-align: left; width: 100%; border-top: 1px solid #eee; padding-top: 1rem; }
            #qa-display div { margin-bottom: 1rem; padding: 0.8rem; background: #f9f9f9; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; }
            
            #question-text::before { content: '■ あなたの質問:'; font-weight: bold; display: block; margin-bottom: 0.3rem; color: #007bff;}
            #answer-text::before { content: '■ AIの回答:'; font-weight: bold; display: block; margin-bottom: 0.3rem; color: #28a745;}
            
            #audioPlayback { margin-top: 1rem; }
        </style>
    </head>
    <body>
        <div id="container">
            <h1>AI Voice Talk</h1>
            <p>下のボタンを押してマイクを起動してください。</p>
            
            <div>
                <button id="startButton">マイクを起動する</button>
                <button id="stopButton" disabled>マイクを停止する</button>
            </div>
            <div>
                <button id="interruptButton" disabled>■ 話をさえぎる</button>
            </div>
            
            <div id="status">準備完了</div>
            <div id="vad-status">(VAD待機中)</div>
            
            <div id="qa-display">
                <div id="question-text"></div>
                <div id="answer-text"></div>
            </div>

            <div id="audioPlayback"></div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/bundle.min.js"></script>

        <script>
            // --- DOM要素 ---
            const startButton = document.getElementById('startButton');
            const stopButton = document.getElementById('stopButton');
            const interruptButton = document.getElementById('interruptButton'); 
            const statusDiv = document.getElementById('status');
            const vadStatusDiv = document.getElementById('vad-status');
            const audioPlayback = document.getElementById('audioPlayback');
            const questionTextDiv = document.getElementById('question-text');
            const answerTextDiv = document.getElementById('answer-text');

            // --- グローバル変数 ---
            let ws;
            let vad; 
            let mediaStream; 
            let isSpeaking = false; 
            let isAISpeaking = false; 
            
            let audioQueue = [];       
            let isPlaying = false;     
            let isServerDone = false;  
            let currentAudio = null;   
            let currentAudioUrl = null; 

            function connectWebSocket() {
                const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
                const wsUrl = wsProtocol + window.location.host + '/ws';
                
                ws = new WebSocket(wsUrl);
                ws.binaryType = 'arraybuffer';

                ws.onopen = () => {
                    console.log('WebSocket 接続成功');
                    statusDiv.textContent = '準備完了。マイクを起動してください。';
                    startButton.disabled = false;
                };

                ws.onmessage = (event) => {
                    if (event.data instanceof ArrayBuffer) {
                        console.log(`音声受信: ${event.data.byteLength} bytes`);
                        const audioBlob = new Blob([event.data], { type: 'audio/mp3' });
                        audioQueue.push(audioBlob);
                        processAudioQueue();
                    } else {
                        try {
                            const data = JSON.parse(event.data);
                            handleJsonMessage(data);
                        } catch (e) {
                            console.error("JSONパースエラー", e);
                        }
                    }
                };

                ws.onclose = () => {
                    statusDiv.textContent = 'サーバー切断。リロードしてください。';
                    stopVAD(); 
                };
            }

            function handleJsonMessage(data) {
                if (data.message) statusDiv.textContent = data.message;

                if (data.status === 'processing') {
                    questionTextDiv.textContent = '(聞き取っています...)';
                    answerTextDiv.textContent = '';
                    audioQueue = [];     
                    isServerDone = false; 
                    isPlaying = false;
                    vad?.pause(); 

                } else if (data.status === 'transcribed') {
                    questionTextDiv.textContent = data.question_text;
                    answerTextDiv.textContent = '...'; 

                } else if (data.status === 'reply_chunk') {
                    if (answerTextDiv.textContent === '...') {
                        answerTextDiv.textContent = '';
                    }
                    answerTextDiv.textContent += data.text_chunk;

                } else if (data.status === 'ignored') {
                    // ★ 無視された場合の処理
                    console.log("サーバーにより無視されました");
                    answerTextDiv.textContent = "(応答なし)";
                    isServerDone = true;
                    finishPlayback(); // 即座に待機に戻る

                } else if (data.status === 'complete') {
                    console.log("サーバー生成完了");
                    isServerDone = true;
                    if (!isPlaying && audioQueue.length === 0) {
                        finishPlayback();
                    }
                    
                } else if (data.status === 'error') {
                    answerTextDiv.textContent = `エラー: ${data.message}`;
                    finishPlayback(); 
                }
            }

            async function setupVAD() {
                try {
                    while (!window.vad) await new Promise(r => setTimeout(r, 50));
                    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    
                    vad = await window.vad.MicVAD.new({
                        stream: mediaStream, 
                        onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/",
                        baseAssetPath: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/",
                        positiveSpeechThreshold: 0.8,
                        negativeSpeechThreshold: 0.8,
                        minSpeechFrames: 2,
                        preSpeechPadFrames: 20,
                        redemptionFrames: 30,
                        
                        onSpeechStart: () => {
                            if (isAISpeaking) return; 
                            isSpeaking = true;
                            vadStatusDiv.textContent = "発話中...";
                        },
                        
                        onSpeechEnd: (audio) => {
                            if (isAISpeaking) return;
                            isSpeaking = false;
                            vadStatusDiv.textContent = "送信中...";
                            
                            if (ws && ws.readyState === WebSocket.OPEN) {
                                sendAudioAsWav(audio);
                                statusDiv.textContent = '音声を送信中...';
                                vad?.pause(); 
                            }
                        }
                    });

                    vad.start();
                    startButton.disabled = true;
                    stopButton.disabled = false;
                    interruptButton.disabled = true;
                    statusDiv.textContent = 'マイク起動完了。話しかけてください。';
                    vadStatusDiv.textContent = '待機中...';

                } catch (err) {
                    console.error('VADエラー:', err);
                    statusDiv.textContent = 'VAD初期化失敗';
                }
            }

            function sendAudioAsWav(float32Array) {
                const wavBuffer = encodeWAV(float32Array, 16000); 
                const blob = new Blob([wavBuffer], { type: 'audio/wav' });
                ws.send(blob);
            }

            // --- WAVエンコード関数は変更なし ---
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

            function stopVAD() {
                vad?.destroy(); 
                vad = null;
                mediaStream?.getTracks().forEach(track => track.stop());
                isSpeaking = false;
                startButton.disabled = false;
                stopButton.disabled = true;
                interruptButton.disabled = true;
                statusDiv.textContent = '停止しました。';
            }
            
            function processAudioQueue() {
                if (isPlaying) return;
                if (audioQueue.length === 0) {
                    if (isServerDone) finishPlayback();
                    return;
                }
                const nextBlob = audioQueue.shift();
                playAudioBlob(nextBlob);
            }

            function playAudioBlob(blob) {
                isPlaying = true;
                isAISpeaking = true;
                vad?.pause();
                statusDiv.textContent = 'AI回答中...';
                interruptButton.disabled = false;

                if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl);
                currentAudioUrl = URL.createObjectURL(blob);
                
                if (currentAudio) {
                    currentAudio.pause();
                    currentAudio.onended = null;
                }

                audioPlayback.innerHTML = ''; 
                currentAudio = new Audio(currentAudioUrl);
                currentAudio.controls = true;
                currentAudio.autoplay = true;

                currentAudio.onended = () => {
                    console.log("断片再生完了");
                    isPlaying = false;
                    processAudioQueue();
                };
                
                currentAudio.onerror = (e) => {
                    console.error("再生エラー", e);
                    isPlaying = false;
                    processAudioQueue();
                }

                audioPlayback.appendChild(currentAudio);
            }

            function finishPlayback() {
                console.log("全完了。待機モードへ");
                isAISpeaking = false;
                isPlaying = false;
                isServerDone = false;
                audioQueue = []; 
                interruptButton.disabled = true; 

                if (currentAudio) {
                    currentAudio.pause();
                    currentAudio = null;
                }
                
                vad?.start(); 
                statusDiv.textContent = '待機中... 話しかけてください。';
                vadStatusDiv.textContent = '待機中...';
            }

            function interruptAudio() {
                console.log("中断");
                audioQueue = []; 
                isServerDone = true; 
                finishPlayback();
                statusDiv.textContent = '中断しました。';
            }

            startButton.onclick = setupVAD;
            stopButton.onclick = stopVAD;
            interruptButton.onclick = interruptAudio; 

            window.onload = () => {
                startButton.disabled = true;
                connectWebSocket();
            };
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"サーバーを http://0.0.0.0:{port} で起動します。")
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None)