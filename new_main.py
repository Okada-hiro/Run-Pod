# /workspace/main.py
import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketDisconnect
import os
import asyncio
import time
import subprocess 
import logging 
import sys 
from pydub import AudioSegment
import io
import base64 # ★ Base64エンコードのために追加

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
    from new_answer_generator import generate_answer
    from new_text_to_speech import synthesize_speech
except ImportError:
    print("[ERROR] 必要なモジュール(transcribe_func, answer_generator, new_text_to_speech)が見つかりません。")
    exit(1)

# --- 設定 ---
PROCESSING_DIR = "incoming_audio" 
MODEL_SIZE = "medium"
LANGUAGE = "ja"

# --- アプリケーション初期化 ---
app = FastAPI()
os.makedirs(PROCESSING_DIR, exist_ok=True)
# /download マウントは残しておいても良い (デバッグ用)
app.mount(f"/download", StaticFiles(directory=PROCESSING_DIR), name="download")
logger.info(f"'{PROCESSING_DIR}' ディレクトリを /download としてマウントしました。")


# ---------------------------
# バックグラウンド処理関数 
# ---------------------------
async def process_audio_file(audio_path: str, original_filename: str, websocket: WebSocket):
    logger.info(f"[TASK START] ファイル処理開始: {original_filename}")
    question_text = ""
    answer_text = ""
    
    try:
        # --- 1. 文字起こし ---
        output_txt_path = os.path.join(PROCESSING_DIR, original_filename + ".txt")
        logger.info(f"[TASK] (1/4) 文字起こし中...")
        question_text = await asyncio.to_thread(
            whisper_text_only,
            audio_path, language=LANGUAGE, output_txt=output_txt_path
        )
        logger.info(f"[TASK] (1/4) 文字起こし完了: {question_text}")

        # クライアントに通知 (文字起こし完了)
        await websocket.send_json({
            "status": "transcribed",
            "message": "文字起こし完了。回答を生成中...",
            "question_text": question_text
        })

        # --- 2. 回答生成 (OpenAI) ---
        logger.info(f"[TASK] (2/4) 回答生成中...")
        answer_text = await asyncio.to_thread(generate_answer, question_text)
        logger.info(f"[TASK] (2/4) 回答生成完了: {answer_text[:30]}...")

        # クライアントに通知 (回答生成完了)
        await websocket.send_json({
            "status": "answered",
            "message": "回答生成完了。音声を合成中...",
            "answer_text": answer_text
        })

        # --- 3. 回答の保存 (.txt) ---
        logger.info(f"[TASK] (3/4) 回答テキスト保存中...")
        answer_file_path = os.path.join(PROCESSING_DIR, original_filename + ".ans.txt")
        with open(answer_file_path, "w", encoding="utf-8") as f:
            f.write(answer_text)
        
        # --- 4. 回答の音声合成 (Fish-Speech) ---
        logger.info(f"[TASK] (4/4) 回答の音声合成中...")
        answer_wav_filename = original_filename + ".ans.wav"
        answer_wav_path_abs = os.path.abspath(os.path.join(PROCESSING_DIR, answer_wav_filename))
        
        success_tts = await asyncio.to_thread(
            synthesize_speech,
            text_to_speak=answer_text,
            output_wav_path=answer_wav_path_abs
        )
        
        if success_tts:
            logger.info(f"[TASK] (4/4) 音声合成 完了。クライアントにBase64データを送信します。")
            
            # ★★★ 変更点：URLの代わりにBase64データを送信 ★★★
            try:
                with open(answer_wav_path_abs, "rb") as audio_file:
                    audio_data = audio_file.read()
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                # 最終完了通知 (データを直接送る)
                await websocket.send_json({
                    "status": "complete",
                    "message": "再生を開始します。",
                    "audio_base64": audio_base64, # URLの代わり
                    "question_text": question_text, 
                    "answer_text": answer_text      
                })
                
                # # 送信後、WAVファイルは削除してもよい (ディスク節約)
                # if os.path.exists(answer_wav_path_abs):
                #     os.remove(answer_wav_path_abs)
                # if os.path.exists(audio_path): # 元のWAVも
                #     os.remove(audio_path)

            except FileNotFoundError:
                 logger.error(f"[TASK ERROR] 生成した音声ファイルが見つかりません: {answer_wav_path_abs}")
                 await websocket.send_json({"status": "error", "message": "音声ファイルの読み込みに失敗しました。"})

        else:
            logger.warning(f"[WARN] (4/4) 音声合成に失敗しました。")
            await websocket.send_json({"status": "error", "message": "音声合成に失敗しました。"})
        
        logger.info(f"[TASK END] 全処理完了: {original_filename}")

    except Exception as e:
        logger.error(f"[TASK ERROR] '{original_filename}' の処理中にエラーが発生しました: {e}", exc_info=True)
        try:
            await websocket.send_json({"status": "error", "message": f"処理中にエラーが発生しました: {e}"})
        except WebSocketDisconnect:
            pass 

# ---------------------------
# WebSocket エンドポイント ( /ws )
# ---------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WS] クライアントが接続しました。")
    try:
        while True:
            audio_data = await websocket.receive_bytes()
            
            # メモリ上のデータ(BytesIO)として扱う
            audio_io = io.BytesIO(audio_data)
            
            temp_id = f"ws_{int(time.time())}"
            output_wav_filename = f"{temp_id}.wav"
            output_wav_path = os.path.join(PROCESSING_DIR, output_wav_filename)

            logger.info(f"[WS] メモリ内で変換処理開始...")
            
            # pydubを使ってメモリ上でWebMを読み込み、WAVとしてディスクに保存
            def convert_audio():
                try:
                    audio = AudioSegment.from_file(audio_io, format="webm") 
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    audio.export(output_wav_path, format="wav")
                    return True
                except Exception as e:
                    logger.error(f"[WS ERROR] pydub 変換失敗: {e}")
                    return False

            if not await asyncio.to_thread(convert_audio):
                await websocket.send_json({"status": "error", "message": "音声形式の変換に失敗しました。"})
                continue
            
            logger.info(f"[WS] 変換成功: {output_wav_path}")
            
            # 処理開始通知
            await websocket.send_json({"status": "processing", "message": "音声を認識しています..."})

            asyncio.create_task(process_audio_file(
                output_wav_path, 
                output_wav_filename, 
                websocket
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
# ルート ( / )
# (ブラウザにHTML/JavaScriptを返す)
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def get_root():
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device.width, initial-scale=1.0">
        <title>VAD音声応答 (Base64)</title>
        
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
            #audioPlayback audio { width: 100%; }
            #downloadLink { margin-top: 0.5rem; font-size: 0.9rem; }
        </style>
    </head>
    <body>
        <div id="container">
            <h1>音声応答システム (Base64)</h1>
            <p>下のボタンを押してマイクを起動してください。</p>
            
            <div>
                <button id="startButton">マイクを起動する</button>
                <button id="stopButton" disabled>マイクを停止する</button>
            </div>
            <div>
                <button id="interruptButton" disabled>■ 回答を中断して話す</button>
            </div>
            
            <div id="status">ここにステータスが表示されます</div>
            <div id="vad-status">(VAD待機中)</div>
            
            <div id="qa-display">
                <div id="question-text"></div>
                <div id="answer-text"></div>
            </div>

            <div id="audioPlayback"></div>
            <div id="downloadLink"></div>
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
            const downloadLinkDiv = document.getElementById('downloadLink');
            const questionTextDiv = document.getElementById('question-text');
            const answerTextDiv = document.getElementById('answer-text');

            // --- グローバル変数 ---
            let ws;
            let mediaRecorder;
            let audioChunks = [];
            let vad; 
            let mediaStream; 
            let silenceTimer = null; 
            let isRecording = false; 
            let isSpeaking = false; 
            let isAISpeaking = false; 
            let currentAudio = null; 
            let currentAudioUrl = null; // Blob URLを保持
            
            // ★ VADの待ち時間。800msのまま。短くしたい場合は 600 などに
            const SILENCE_THRESHOLD_MS = 800; 

            // --- 1. WebSocket接続 ---
            function connectWebSocket() {
                const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
                const wsUrl = wsProtocol + window.location.host + '/ws';
                
                ws = new WebSocket(wsUrl);

                ws.onopen = () => {
                    console.log('WebSocket 接続成功');
                    statusDiv.textContent = '準備完了。マイクを起動してください。';
                    startButton.disabled = false;
                };

                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    console.log('サーバーからメッセージ:', data);
                    
                    if (data.message) {
                        statusDiv.textContent = data.message;
                    }

                    if (data.status === 'processing') {
                        questionTextDiv.textContent = '(聞き取っています...)';
                        answerTextDiv.textContent = '';
                        vad?.pause(); 

                    } else if (data.status === 'transcribed') {
                        questionTextDiv.textContent = data.question_text;
                        answerTextDiv.textContent = '(回答を生成しています...)';

                    } else if (data.status === 'answered') {
                        answerTextDiv.textContent = data.answer_text;

                    // ★★★ 変更点： audio_url の代わりに audio_base64 を受信 ★★★
                    } else if (data.status === 'complete' && data.audio_base64) {
                        if(data.question_text) questionTextDiv.textContent = data.question_text;
                        if(data.answer_text) answerTextDiv.textContent = data.answer_text;
                        
                        // Base64から音声再生＆リンク作成
                        playAudioFromBase64(data.audio_base64); 
                        
                    } else if (data.status === 'error') {
                        answerTextDiv.textContent = `エラー: ${data.message}`;
                        statusDiv.textContent = 'エラーが発生しました。待機中に戻ります。';
                        isAISpeaking = false;
                        interruptButton.disabled = true; 
                        vad?.start();
                    }
                };

                ws.onclose = () => {
                    console.log('WebSocket 接続切断');
                    statusDiv.textContent = 'サーバーとの接続が切れました。リロードしてください。';
                    stopVAD(); 
                };
            }

            // --- 2. VADとマイクのセットアップ ---
            async function setupVAD() {
                try {
                    while (!window.vad) {
                        await new Promise(r => setTimeout(r, 50));
                    }
                    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    setupMediaRecorder(mediaStream);

                    vad = await window.vad.MicVAD.new({
                        stream: mediaStream, 
                        onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/",
                        baseAssetPath: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/",
                        
                        onSpeechStart: () => {
                            if (isAISpeaking) return; 
                            isSpeaking = true;
                            vadStatusDiv.textContent = "発話中...";
                            if (silenceTimer) { clearTimeout(silenceTimer); silenceTimer = null; }
                            if (!isRecording) startMediaRecorder(); 
                        },
                        onSpeechEnd: (audio) => {
                            if (isAISpeaking) return;
                            isSpeaking = false;
                            vadStatusDiv.textContent = "発話終了 (無音タイマー起動)";
                            if (isRecording) startSilenceTimer(); 
                        }
                    });

                    vad.start();

                    startButton.disabled = true;
                    stopButton.disabled = false;
                    interruptButton.disabled = true;
                    statusDiv.textContent = 'マイク起動完了。話しかけてください。';
                    vadStatusDiv.textContent = '待機中...';

                } catch (err) {
                    console.error('VADまたはマイクのセットアップに失敗:', err);
                    statusDiv.textContent = 'VADの初期化に失敗しました。';
                }
            }

            // --- 3. MediaRecorder のセットアップ ---
            function setupMediaRecorder(stream) {
                // 圧縮形式 (WebM/Opus) で送信
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) audioChunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    isRecording = false;
                    if (audioChunks.length === 0) {
                        if (!isAISpeaking && vad) vad.start(); 
                        return;
                    }
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    audioChunks = []; 
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(audioBlob);
                        statusDiv.textContent = '音声を送信中...';
                        vadStatusDiv.textContent = 'サーバー処理中...';
                        vad?.pause(); 
                    }
                };
                
                mediaRecorder.onstart = () => {
                    isRecording = true;
                    audioChunks = []; 
                };
            }
            
            function startMediaRecorder() {
                if (mediaRecorder && !isRecording) {
                    if (isAISpeaking) return;
                    mediaRecorder.start(1000); 
                }
            }
            
            function stopMediaRecorder() {
                if (mediaRecorder && isRecording) mediaRecorder.stop();
            }

            // --- 5. 無音タイマー ---
            function startSilenceTimer() {
                if (silenceTimer) clearTimeout(silenceTimer);
                silenceTimer = setTimeout(() => {
                    if (isRecording && !isSpeaking) {
                        vadStatusDiv.textContent = "無音検出。送信します。";
                        stopMediaRecorder();
                    }
                    silenceTimer = null;
                }, SILENCE_THRESHOLD_MS);
            }

            // --- 6. VAD停止 ---
            function stopVAD() {
                vad?.destroy(); 
                vad = null;
                mediaStream?.getTracks().forEach(track => track.stop());
                mediaStream = null;
                if (mediaRecorder && isRecording) mediaRecorder.stop();
                isRecording = false;
                
                startButton.disabled = false;
                stopButton.disabled = true;
                interruptButton.disabled = true;
                statusDiv.textContent = 'マイクが停止しました。';
                vadStatusDiv.textContent = '';
            }
            
            // --- 7. ユーティリティ & 再生制御 ---
            
            // ★★★ 変更点：Base64から再生・リンク作成する関数 ★★★
            function playAudioFromBase64(base64String) {
                vad?.pause(); 
                isAISpeaking = true;
                statusDiv.textContent = '音声回答を再生中...';
                interruptButton.disabled = false;
                
                // 以前のBlob URLがあれば解放
                if (currentAudioUrl) {
                    URL.revokeObjectURL(currentAudioUrl);
                    currentAudioUrl = null;
                }
                
                // Base64をデコードしてBlobを作成
                const binaryString = window.atob(base64String);
                const len = binaryString.length;
                const bytes = new Uint8Array(len);
                for (let i = 0; i < len; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                const audioBlob = new Blob([bytes], { type: 'audio/wav' });
                
                // メモリ上のBlobへのURLを生成 (HTTPリクエスト不要！)
                currentAudioUrl = URL.createObjectURL(audioBlob);
                
                // 1. 音声再生
                audioPlayback.innerHTML = '';
                if (currentAudio) { currentAudio.pause(); }
                
                currentAudio = new Audio(currentAudioUrl);
                currentAudio.controls = true;
                currentAudio.autoplay = true; // 即時再生
                
                currentAudio.onended = () => {
                    console.log("再生完了。");
                    finishPlayback(); // VAD再開などの処理
                };
                
                audioPlayback.appendChild(currentAudio);
                
                // 2. ダウンロードリンク作成
                downloadLinkDiv.innerHTML = '';
                const a = document.createElement('a');
                a.href = currentAudioUrl; // 再生と同じメモリURL
                a.textContent = '回答音声をダウンロード';
                a.download = `answer_${Date.now()}.wav`; 
                downloadLinkDiv.appendChild(a);
            }

            // 再生終了時（自然終了または中断時）の処理
            function finishPlayback() {
                isAISpeaking = false;
                interruptButton.disabled = true; // ボタンを無効化
                if (currentAudio) {
                    currentAudio.pause();
                    currentAudio = null;
                }
                
                // メモリリーク防止のため、URLは保持しておくが
                // 次の再生時に古いのを解放する (interruptAudioとの兼ね合い)
                
                vad?.start(); // VAD再開
                statusDiv.textContent = '待機中... 話しかけてください。';
                vadStatusDiv.textContent = '待機中...';
            }

            // ★ 中断ボタンが押された時の処理
            function interruptAudio() {
                if (currentAudio) {
                    console.log("ユーザー操作により再生を中断します。");
                    currentAudio.pause(); // 音声を停止
                    currentAudio = null;
                }
                
                // UIリセット
                audioPlayback.innerHTML = '';
                // ダウンロードリンクは残しても良い
                
                // finishPlaybackとほぼ同じ処理を実行
                finishPlayback();
                
                statusDiv.textContent = '中断しました。どうぞお話しください。';
                vadStatusDiv.textContent = '聞き取り待機中...';
            }

            // --- 8. イベント ---
            startButton.onclick = setupVAD;
            stopButton.onclick = stopVAD;
            interruptButton.onclick = interruptAudio; // 中断イベント登録

            window.onload = () => {
                startButton.disabled = true;
                connectWebSocket();
            };

        </script>
    </body>
    </html>
    """

# ---------------------------
# サーバー起動
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"サーバーを http://0.0.0.0:{port} で起動します。")
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None)