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
import logging # ★ 優先度1: ロギングモジュールをインポート
import sys # ★ 優先度1: ロギング出力先指定のため

# --- 既存の処理モジュールをインポート ---
try:
    from transcribe_func import whisper_text_only
    from new_answer_generator import generate_answer
    from new_text_to_speech import synthesize_speech
except ImportError:
    print("[ERROR] 必要なモジュール(transcribe_func, answer_generator, new_text_to_speech)が見つかりません。")
    exit(1)

# --- ★ 優先度1: ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # コンソールに標準出力する
)
logger = logging.getLogger(__name__)

# --- 設定 ---
PROCESSING_DIR = "incoming_audio" # アップロード/処理結果の保存場所
MODEL_SIZE = "medium"
LANGUAGE = "ja"

# --- アプリケーション初期化 ---
app = FastAPI()
os.makedirs(PROCESSING_DIR, exist_ok=True)

# 1. /download エンドポイント (生成された .ans.wav をブラウザに返す)
app.mount(f"/download", StaticFiles(directory=PROCESSING_DIR), name="download")
logger.info(f"'{PROCESSING_DIR}' ディレクトリを /download としてマウントしました。")


# ---------------------------
# バックグラウンド処理関数 (WebSocketオブジェクトを追加)
# ---------------------------
async def process_audio_file(audio_path: str, original_filename: str, websocket: WebSocket):
    """
    アップロードされた音声ファイルを受け取り、一連の処理を実行し、
    完了したらWebSocketでクライアントに通知する
    """
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

        # --- 2. 回答生成 (OpenAI) ---
        logger.info(f"[TASK] (2/4) 回答生成中...")
        answer_text = await asyncio.to_thread(generate_answer, question_text)
        logger.info(f"[TASK] (2/4) 回答生成完了: {answer_text[:30]}...")

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
            # ★ 優先度2 & 3: 完了通知にテキストデータを追加
            download_url = f"/download/{answer_wav_filename}"
            await websocket.send_json({
                "status": "complete",
                "message": "回答の準備ができました。",
                "audio_url": download_url,
                "question_text": question_text, # ★ 優先度3
                "answer_text": answer_text      # ★ 優先度3
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
            pass # クライアントが切断済みなら何もしない

# ---------------------------
# WebSocket エンドポイント ( /ws )
# (ブラウザからのマイク音声データ受信)
# ---------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WS] クライアントが接続しました。")
    try:
        while True:
            # 1. ブラウザから音声データ (Blob) を受信
            audio_data = await websocket.receive_bytes()
            
            # 2. 一時ファイルとして保存 (ブラウザからは .webm 形式が多い)
            temp_id = f"ws_{int(time.time())}"
            temp_input_path = os.path.join(PROCESSING_DIR, f"{temp_id}.webm") 
            
            with open(temp_input_path, "wb") as f:
                f.write(audio_data)
            
            # 3. ffmpeg で .wav に変換 (Whisperが処理できる形式へ)
            output_wav_filename = f"{temp_id}.wav"
            output_wav_path = os.path.join(PROCESSING_DIR, output_wav_filename)
            
            # 16kHz モノラル 16bit PCM に変換
            cmd = [
                "ffmpeg",
                "-i", temp_input_path,
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                "-y", # 常に上書き
                output_wav_path
            ]
            
            logger.info(f"[WS] ffmpeg 変換実行: {temp_input_path} -> {output_wav_path}")
            # 非同期でサブプロセスを実行
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                logger.error(f"[WS ERROR] ffmpeg 変換失敗: {stderr.decode()}")
                await websocket.send_json({"status": "error", "message": "音声形式の変換に失敗しました。"})
                continue
            
            logger.info(f"[WS] ffmpeg 変換成功。")
            
            # 4. バックグラウンド処理タスクを開始 (websocketオブジェクトを渡す)
            asyncio.create_task(process_audio_file(
                output_wav_path, 
                output_wav_filename, # .wav ファイル名を基準にする
                websocket
            ))
            
            # 5. クライアントに「処理中」を通知
            await websocket.send_json({"status": "processing", "message": "文字起こしと回答生成を開始しました..."})

            # 6. 一時入力ファイル (webm) を削除
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)

    except WebSocketDisconnect:
        logger.info("[WS] クライアントが切断しました。")
    except Exception as e:
        logger.error(f"[WS ERROR] WebSocketエラー: {e}", exc_info=True)
    finally:
        await websocket.close()


# ---------------------------
# ルート ( / )
# (ブラウザにHTML/JavaScriptを返す)
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def get_root():
    # ★★★ HTML部分を修正 ★★★
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
            #status { margin-top: 1.5rem; font-size: 1.1rem; color: #333; min-height: 2em; }
            #vad-status { font-size: 0.9rem; color: #666; height: 1.5em; }
            #qa-display { margin: 1.5rem auto 0 auto; text-align: left; width: 100%; border-top: 1px solid #eee; padding-top: 1rem; }
            #qa-display div { margin-bottom: 1rem; padding: 0.5rem; background: #f9f9f9; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; }
            #qa-display div:empty { display: none; }
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

        <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@latest/dist/bundle.min.js"></script>

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
            
            const SILENCE_THRESHOLD_MS = 2000; // 2秒間の無音で録音停止

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
                    
                    statusDiv.textContent = data.message; 

                    if (data.status === 'complete' && data.audio_url) {
                        // AIの回答を再生
                        playAudio(data.audio_url); 
                        questionTextDiv.textContent = data.question_text || '（質問を聞き取れませんでした）';
                        answerTextDiv.textContent = data.answer_text || '（回答を生成できませんでした）';
                        createDownloadLink(data.audio_url);
                        
                    } else if (data.status === 'error') {
                        answerTextDiv.textContent = `エラー: ${data.message}`;
                        // エラー時も VAD を再開
                        vad?.start();
                        statusDiv.textContent = 'エラーが発生しました。待機中に戻ります。';
                    }
                };

                ws.onclose = () => {
                    console.log('WebSocket 接続切断');
                    statusDiv.textContent = 'サーバーとの接続が切れました。リロードしてください。';
                    stopVAD(); // VADも停止
                };
            }

            // --- 2. VADとマイクのセットアップ ---
            async function setupVAD() {
                try {
                    // 1. VADライブラリ (bundle.min.js) がグローバルに 'vad' (小文字) を定義するのを待つ
                    while (!window.vad) {
                        console.log("VADロード待機中...");
                        await new Promise(r => setTimeout(r, 50));
                    }
                    console.log("VADライブラリ ロード完了。");

                    // 2. ユーザー操作でマイクを取得
                    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

                    // 3. MediaRecorder のセットアップ (VADとは別に録音を管理)
                    setupMediaRecorder(mediaStream);

                    // 4. VADライブラリを初期化
                    vad = await window.vad.MicVAD.new({
                        stream: mediaStream, 
                        
                        // ★★★ 修正: ログ(404)に基づき、モデルのURLを明示的に指定 ★★★
                        // ライブラリが /silero_vad_legacy.onnx を探しに行っていたため、
                        // CDNのフルパスを指定します。
                        modelURL: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@latest/dist/silero_vad_legacy.onnx",
                        // workletのパスも念のため明示的に指定
                        workletURL: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@latest/dist/vad.worklet.min.js",
                        
                        onSpeechStart: () => {
                            isSpeaking = true;
                            vadStatusDiv.textContent = "発話中...";
                            if (silenceTimer) { clearTimeout(silenceTimer); silenceTimer = null; }
                            // VADが発話を開始したら、MediaRecorder の録音を開始
                            if (!isRecording) startMediaRecorder(); 
                        },
                        onSpeechEnd: (audio) => {
                            // VADが返す生のオーディオデータ(audio)は *無視* する
                            isSpeaking = false;
                            vadStatusDiv.textContent = "発話終了 (無音タイマー起動)";
                            // VADが発話終了したら、MediaRecorder を停止するためのタイマーを開始
                            if (isRecording) startSilenceTimer(); 
                        }
                    });

                    // 5. VAD を開始
                    vad.start();

                    // 6. ボタン・ステータス更新
                    startButton.disabled = true;
                    stopButton.disabled = false;
                    statusDiv.textContent = 'マイク起動完了。話しかけてください。';
                    vadStatusDiv.textContent = '待機中...';

                } catch (err) {
                    console.error('VADまたはマイクのセットアップに失敗:', err);
                    statusDiv.textContent = 'マイクへのアクセスが許可されていません。';
                }
            }

            
            // --- 3. MediaRecorder (録音機能) のセットアップ ---
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
                        console.log("録音データが空です。送信をスキップします。");
                        // AIが喋っていなければ VAD を再開
                        if (!isAISpeaking) vad?.start(); 
                        return;
                    }

                    // 録音データをBlobとして結合
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    audioChunks = []; // チャンクをクリア

                    // WebSocketでサーバーに送信
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(audioBlob);
                        statusDiv.textContent = '音声を送信中... サーバー処理を待っています。';
                        vadStatusDiv.textContent = 'サーバー処理中...';
                    } else {
                        statusDiv.textContent = 'サーバーに接続されていません。';
                    }
                };
                
                mediaRecorder.onstart = () => {
                    console.log("MediaRecorder: 録音開始。");
                    isRecording = true;
                    audioChunks = []; // チャンクをリセット
                    clearResults(); // 以前の結果をクリア
                };
            }
            
            // --- 4. 録音の開始/停止制御 ---
            
            function startMediaRecorder() {
                if (mediaRecorder && !isRecording) {
                    if (isAISpeaking) {
                        console.log("AI再生中のため、録音を開始しません。");
                        return;
                    }
                    mediaRecorder.start(1000); // 1秒ごとにチャンクを生成
                }
            }
            
            function stopMediaRecorder() {
                if (mediaRecorder && isRecording) {
                    mediaRecorder.stop();
                    // VADも一時停止 (サーバーの応答を待つため)
                    vad?.pause(); 
                }
            }

            // --- 5. 無音タイマー ---
            function startSilenceTimer() {
                if (silenceTimer) {
                    clearTimeout(silenceTimer);
                }
                silenceTimer = setTimeout(() => {
                    console.log(`無音時間が ${SILENCE_THRESHOLD_MS}ms に達しました。`);
                    if (isRecording && !isSpeaking) {
                        // 発話中でなく、録音中であれば、録音を停止（＝サーバーへ送信）
                        vadStatusDiv.textContent = "無音検出。サーバーへ送信します。";
                        stopMediaRecorder();
                    }
                    silenceTimer = null;
                }, SILENCE_THRESHOLD_MS);
            }

            // --- 6. VADの停止 (クリーンアップ) ---
            function stopVAD() {
                vad?.destroy(); 
                vad = null;
                mediaStream?.getTracks().forEach(track => track.stop());
                mediaStream = null;
                if (mediaRecorder && isRecording) {
                    mediaRecorder.stop();
                }
                isRecording = false;
                
                startButton.disabled = false;
                stopButton.disabled = true;
                statusDiv.textContent = 'マイクが停止しました。';
                vadStatusDiv.textContent = '';
            }
            
            // --- 7. ユーティリティ関数 ---
            function clearResults() {
                audioPlayback.innerHTML = '';
                downloadLinkDiv.innerHTML = '';
                questionTextDiv.textContent = '';
                answerTextDiv.textContent = '';
            }

            function playAudio(url) {
                // AIが話し始めるので VAD を一時停止 (AIの声を拾わないように)
                vad?.pause(); 
                isAISpeaking = true;
                
                audioPlayback.innerHTML = '';
                const audio = new Audio(url);
                audio.controls = true;
                audio.autoplay = true;
                
                // ★ 再生が終了したら VAD を再開
                audio.onended = () => {
                    console.log("AIの再生完了。VADを再開します。");
                    isAISpeaking = false;
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

            // --- 8. イベントリスナー ---
            startButton.onclick = setupVAD;
            stopButton.onclick = stopVAD;

            // ページ読み込み時にWebSocket接続を開始
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
    # RunPodのデフォルトHTTPポート (8000など) に合わせる
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"サーバーを http://0.0.0.0:{port} で起動します。")
    # ★ 優先度1: Uvicornのロギングをカスタムロガーに合わせる
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None)