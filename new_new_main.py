# /workspace/main.py
import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketDisconnect
import os
import asyncio
import time
import subprocess # ffmpeg 実行のため
import logging 
import sys 
from pydub import AudioSegment
import io

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

        # ★★★ 変更点1: 文字起こし完了時点でクライアントに通知 ★★★
        await websocket.send_json({
            "status": "transcribed",
            "message": "文字起こし完了。回答を生成中...",
            "question_text": question_text
        })

        # --- 2. 回答生成 (OpenAI) ---
        logger.info(f"[TASK] (2/4) 回答生成中...")
        answer_text = await asyncio.to_thread(generate_answer, question_text)
        logger.info(f"[TASK] (2/4) 回答生成完了: {answer_text[:30]}...")

        # ★★★ 変更点2: 回答生成完了時点でクライアントに通知 ★★★
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
            logger.info(f"[TASK] (4/4) 音声合成 完了。クライアントに通知します。")
            download_url = f"/download/{answer_wav_filename}"
            
            # 最終完了通知
            await websocket.send_json({
                "status": "complete",
                "message": "再生を開始します。",
                "audio_url": download_url,
                # 念のためここでもテキストを送るが、通常は既に表示済み
                "question_text": question_text, 
                "answer_text": answer_text      
            })
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
            
            # ★ここから変更★
            
            # ファイルに書き込まず、メモリ上のデータ(BytesIO)として扱う
            audio_io = io.BytesIO(audio_data)
            
            # 受信時刻をIDにする
            temp_id = f"ws_{int(time.time())}"
            output_wav_filename = f"{temp_id}.wav"
            output_wav_path = os.path.join(PROCESSING_DIR, output_wav_filename)

            logger.info(f"[WS] メモリ内で変換処理開始...")
            
            # pydubを使ってメモリ上でWebMを読み込み、WAVとしてディスクに保存
            # (subprocessの起動オーバーヘッドを削減)
            def convert_audio():
                audio = AudioSegment.from_file(audio_io, format="webm") # または format="ogg" (opusの場合)
                # 16000Hz, モノラルに変換して保存
                audio = audio.set_frame_rate(16000).set_channels(1)
                audio.export(output_wav_path, format="wav")

            await asyncio.to_thread(convert_audio)
            
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
        <title>VAD音声応答</title>
        
        <style>
            body { font-family: sans-serif; display: grid; place-items: center; min-height: 90vh; background: #f4f4f4; }
            #container { background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; width: 90%; max-width: 600px; }
            #startButton { 
                font-size: 1.2rem; padding: 0.8rem 1.5rem; border: none; 
                border-radius: 5px; cursor: pointer; margin: 0.5rem; 
                background: #007bff; color: white;
            }
            #startButton:disabled { background: #ccc; }
            #stopButton { background: #dc3545; color: white; font-size: 1rem; padding: 0.5rem 1rem; }
            #stopButton:disabled { display: none; }
            #status { margin-top: 1.5rem; font-size: 1.1rem; color: #333; min-height: 2em; font-weight: bold; }
            #vad-status { font-size: 0.9rem; color: #666; height: 1.5em; }
            #qa-display { margin: 1.5rem auto 0 auto; text-align: left; width: 100%; border-top: 1px solid #eee; padding-top: 1rem; }
            #qa-display div { margin-bottom: 1rem; padding: 0.8rem; background: #f9f9f9; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; }
            
            /* 空でも非表示にしない（プレースホルダーが見えるようにする等の調整はお好みで） */
            /* #qa-display div:empty { display: none; } */

            #question-text::before { content: '■ あなたの質問:'; font-weight: bold; display: block; margin-bottom: 0.3rem; color: #007bff;}
            #answer-text::before { content: '■ AIの回答:'; font-weight: bold; display: block; margin-bottom: 0.3rem; color: #28a745;}
            #audioPlayback { margin-top: 1rem; }
            #audioPlayback audio { width: 100%; }
            #downloadLink { margin-top: 0.5rem; font-size: 0.9rem; }
        </style>
    </head>
    <body>
        <div id="container">
            <h1>音声応答システム (VAD)</h1>
            <p>下のボタンを押してマイクを起動してください。</p>
            
            <button id="startButton">マイクを起動する</button>
            <button id="stopButton" disabled>マイクを停止する</button>
            
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
            let vad; // (VADインスタンス)
            let mediaStream; // マイクストリーム
            let silenceTimer = null; // 無音検出タイマー
            let isRecording = false; // MediaRecorder が録音中か
            let isSpeaking = false; // VAD が発話を検知中か
            let isAISpeaking = false; // AIが再生中か
            
            const SILENCE_THRESHOLD_MS = 800; // 2秒間の無音で録音停止

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
                    
                    // 共通してメッセージがあれば表示
                    if (data.message) {
                        statusDiv.textContent = data.message;
                    }

                    // ★★★ 変更点: ステータスごとの分岐処理を追加 ★★★
                    
                    if (data.status === 'processing') {
                        // 音声送信直後
                        questionTextDiv.textContent = '(聞き取っています...)';
                        answerTextDiv.textContent = '';
                        vad?.pause(); // 処理中はVADを一時停止（自分の独り言を拾わないように）

                    } else if (data.status === 'transcribed') {
                        // 1. 文字起こし完了 -> 即表示
                        questionTextDiv.textContent = data.question_text;
                        answerTextDiv.textContent = '(回答を生成しています...)';

                    } else if (data.status === 'answered') {
                        // 2. 回答生成完了 -> 即表示
                        answerTextDiv.textContent = data.answer_text;
                        // 音声合成待ちであることを表示
                        // statusDivは上の共通処理で更新済み

                    } else if (data.status === 'complete' && data.audio_url) {
                        // 3. 音声合成完了 -> 再生開始
                        // 念のためテキストも上書き（内容は同じはず）
                        if(data.question_text) questionTextDiv.textContent = data.question_text;
                        if(data.answer_text) answerTextDiv.textContent = data.answer_text;
                        
                        playAudio(data.audio_url); 
                        createDownloadLink(data.audio_url);
                        
                    } else if (data.status === 'error') {
                        answerTextDiv.textContent = `エラー: ${data.message}`;
                        statusDiv.textContent = 'エラーが発生しました。待機中に戻ります。';
                        // エラー時はVADを再開させる
                        isAISpeaking = false;
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
                        console.log("VADロード待機中...");
                        await new Promise(r => setTimeout(r, 50));
                    }
                    console.log("VADライブラリ ロード完了。");

                    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    setupMediaRecorder(mediaStream);

                    vad = await window.vad.MicVAD.new({
                        stream: mediaStream, 
                        onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/",
                        baseAssetPath: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/",
                        
                        onSpeechStart: () => {
                            if (isAISpeaking) return; // AI再生中は無視
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
                    statusDiv.textContent = 'マイク起動完了。話しかけてください。';
                    vadStatusDiv.textContent = '待機中...';

                } catch (err) {
                    console.error('VADまたはマイクのセットアップに失敗:', err);
                    statusDiv.textContent = 'VADの初期化に失敗しました。';
                }
            }

            // --- 3. MediaRecorder のセットアップ ---
            function setupMediaRecorder(stream) {
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = () => {
                    console.log("MediaRecorder: 録音停止。");
                    isRecording = false;

                    if (audioChunks.length === 0) {
                        console.log("録音データが空です。");
                        // AI再生中でなければVAD再開
                        if (!isAISpeaking && vad) vad.start(); 
                        return;
                    }

                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    audioChunks = []; 

                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(audioBlob);
                        statusDiv.textContent = '音声を送信中...';
                        vadStatusDiv.textContent = 'サーバー処理中...';
                        // ここでVADを止めて、サーバーからのレスポンスを待つ
                        vad?.pause(); 
                    } else {
                        statusDiv.textContent = 'サーバーに接続されていません。';
                    }
                };
                
                mediaRecorder.onstart = () => {
                    console.log("MediaRecorder: 録音開始。");
                    isRecording = true;
                    audioChunks = []; 
                    // 画面クリアはまだしない（前の会話を残したい場合はここを調整）
                    // clearResults(); 
                };
            }
            
            // --- 4. 録音制御 ---
            function startMediaRecorder() {
                if (mediaRecorder && !isRecording) {
                    if (isAISpeaking) return;
                    mediaRecorder.start(1000); 
                }
            }
            
            function stopMediaRecorder() {
                if (mediaRecorder && isRecording) {
                    mediaRecorder.stop();
                }
            }

            // --- 5. 無音タイマー ---
            function startSilenceTimer() {
                if (silenceTimer) clearTimeout(silenceTimer);
                silenceTimer = setTimeout(() => {
                    console.log(`無音判定。録音停止します。`);
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
                statusDiv.textContent = 'マイクが停止しました。';
                vadStatusDiv.textContent = '';
            }
            
            // --- 7. ユーティリティ ---
            function clearResults() {
                audioPlayback.innerHTML = '';
                downloadLinkDiv.innerHTML = '';
                questionTextDiv.textContent = '';
                answerTextDiv.textContent = '';
            }

            function playAudio(url) {
                // ★再生開始時に確実にVADを止める
                vad?.pause(); 
                isAISpeaking = true;
                statusDiv.textContent = '音声回答を再生中...';
                
                audioPlayback.innerHTML = '';
                const audio = new Audio(url);
                audio.controls = true;
                audio.autoplay = true;
                
                audio.onended = () => {
                    console.log("AIの再生完了。VADを再開します。");
                    isAISpeaking = false;
                    
                    // ★再生終了後にユーザー入力を受け付け再開
                    vad?.start(); 
                    
                    statusDiv.textContent = '待機中... 話しかけてください。';
                    vadStatusDiv.textContent = '待機中...';
                };
                
                audioPlayback.appendChild(audio);
            }
            
            function createDownloadLink(url) {
                downloadLinkDiv.innerHTML = '';
                const a = document.createElement('a');
                a.href = url;
                a.textContent = '回答音声をダウンロード';
                a.download = url.split('/').pop(); 
                downloadLinkDiv.appendChild(a);
            }

            // --- 8. イベント ---
            startButton.onclick = setupVAD;
            stopButton.onclick = stopVAD;

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