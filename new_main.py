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
    from new_answer_generator import generate_answer, generate_answer_stream
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
    
    try:
        # --- 1. 文字起こし (ここは同じ) ---
        output_txt_path = os.path.join(PROCESSING_DIR, original_filename + ".txt")
        logger.info(f"[TASK] (1/4) 文字起こし中...")
        question_text = await asyncio.to_thread(
            whisper_text_only, audio_path, language=LANGUAGE, output_txt=output_txt_path
        )
        logger.info(f"[TASK] (1/4) 文字起こし完了: {question_text}")

        await websocket.send_json({
            "status": "transcribed",
            "message": "回答を生成しながら話します...",
            "question_text": question_text
        })

        # --- 2 & 3 & 4. ストリーミング回答・合成・送信 ---
        logger.info(f"[TASK] ストリーミング処理開始...")

        # バッファとカウンターの準備
        text_buffer = ""
        sentence_count = 0
        full_answer_log = ""

        # 正規表現: 読点(、)では切らず、句点(。)や感嘆符(！)、改行で切る
        # (?<=...) は「後読み」アサーションで、区切り文字を文末に含めるため
        split_pattern = r'(?<=[。！？\n])'

        # ジェネレータからテキストを少しずつ取得
        iterator = generate_answer_stream(question_text)

        for chunk_text in iterator:
            text_buffer += chunk_text
            full_answer_log += chunk_text # ログ用

            # バッファ内に句読点があるかチェックして分割
            sentences = re.split(split_pattern, text_buffer)

            # sentences の最後以外は「確定した文」なので処理する
            # (最後の一つはまだ続きがあるかもしれないのでバッファに戻す)
            if len(sentences) > 1:
                for sent in sentences[:-1]:
                    if sent.strip(): # 空文字でなければ処理
                        sentence_count += 1
                        await process_sentence(sent, original_filename, sentence_count, websocket)
                
                # 未確定の末尾をバッファに戻す
                text_buffer = sentences[-1]

        # ループ終了後、バッファに残っているテキストがあれば処理
        if text_buffer.strip():
            sentence_count += 1
            await process_sentence(text_buffer, original_filename, sentence_count, websocket)
        
        # 全送信完了を通知
        await websocket.send_json({"status": "complete", "answer_text": full_answer_log})
        logger.info(f"[TASK END] ストリーミング完了: {original_filename}")

    except Exception as e:
        logger.error(f"[TASK ERROR] エラー: {e}", exc_info=True)
        await websocket.send_json({"status": "error", "message": f"エラー: {e}"})


# ★ ヘルパー関数: 1文を音声化して送信する
async def process_sentence(text: str, base_filename: str, index: int, websocket: WebSocket):
    logger.info(f"[STREAM] 文{index}: {text[:20]}...")
    
    # ファイル名: original.wav.part1.wav
    part_filename = f"{base_filename}.part{index}.wav"
    part_path_abs = os.path.abspath(os.path.join(PROCESSING_DIR, part_filename))

    # 音声合成
    success = await asyncio.to_thread(
        synthesize_speech, text_to_speak=text, output_wav_path=part_path_abs
    )

    if success:
        # MP3変換 (pydub)
        try:
            audio_segment = AudioSegment.from_wav(part_path_abs)
            mp3_buffer = io.BytesIO()
            audio_segment.export(mp3_buffer, format="mp3", bitrate="128k")
            audio_data = mp3_buffer.getvalue()

            # 送信 (メタデータなしでバイナリだけ送る運用でもOKだが、
            # フロント側でテキスト表示したいならJSONも送ると良い。今回はシンプルにバイナリ送信)
            await websocket.send_bytes(audio_data)
            
            # 生成したWAVは削除しても良いが、デバッグ用に残してもOK
            # os.remove(part_path_abs) 

        except Exception as e:
            logger.error(f"[STREAM ERROR] MP3変換失敗: {e}")

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
                    # ★変更前: format="webm" と指定していた
                    # audio = AudioSegment.from_file(audio_io, format="webm") 
                    
                    # ★変更後: format指定を削除 (自動判別させる)
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
            let vad; 
            let mediaStream; 
            let isSpeaking = false; 
            let isAISpeaking = false; 
            
            // ストリーミング再生制御用
            let audioQueue = [];       
            let isPlaying = false;     
            let isServerDone = false;  
            let currentAudio = null;   
            let currentAudioUrl = null; 

            // --- 1. WebSocket接続 ---
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
                    console.log('WebSocket 接続切断');
                    statusDiv.textContent = 'サーバーとの接続が切れました。リロードしてください。';
                    stopVAD(); 
                };
            }

            // --- 2. メッセージ処理 ---
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
                    answerTextDiv.textContent = '(回答生成中...)';

                } else if (data.status === 'answered') {
                    answerTextDiv.textContent = data.answer_text;

                } else if (data.status === 'complete') {
                    console.log("サーバー処理完了通知を受信");
                    isServerDone = true;
                    if(data.answer_text) answerTextDiv.textContent = data.answer_text;
                    
                    if (!isPlaying && audioQueue.length === 0) {
                        finishPlayback();
                    }
                    
                } else if (data.status === 'error') {
                    answerTextDiv.textContent = `エラー: ${data.message}`;
                    statusDiv.textContent = 'エラーが発生しました。待機中に戻ります。';
                    finishPlayback(); 
                }
            }

            // --- 3. VADセットアップ (★ここが重要★) ---
            async function setupVAD() {
                try {
                    while (!window.vad) {
                        await new Promise(r => setTimeout(r, 50));
                    }
                    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    
                    // MediaRecorderは廃止し、VADの内部バッファを使います
                    
                    vad = await window.vad.MicVAD.new({
                        stream: mediaStream, 
                        onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/",
                        baseAssetPath: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/",
                        
                        // ★ 感度調整
                        positiveSpeechThreshold: 0.8,
                        negativeSpeechThreshold: 0.8,
                        minSpeechFrames: 2,           // 発話検知を少し早める

                        // ★★★ 最重要設定 ★★★
                        preSpeechPadFrames: 20,       // 【重要】声と判定される「前」の20フレーム(約0.6秒)も含める
                        redemptionFrames: 30,         // 無音になってから1秒弱待ってから送信（文の切れ目を待つ）
                        
                        onSpeechStart: () => {
                            if (isAISpeaking) return; 
                            isSpeaking = true;
                            vadStatusDiv.textContent = "発話中...";
                        },
                        
                        onSpeechEnd: (audio) => {
                            // audio は Float32Array (-1.0〜1.0) で、preSpeechPadFrames 分も含んでいる
                            if (isAISpeaking) return;
                            isSpeaking = false;
                            vadStatusDiv.textContent = "発話終了。送信します...";
                            
                            // WebSocketが繋がっているならWAVに変換して送信
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
                    console.error('VAD設定エラー:', err);
                    statusDiv.textContent = 'VADの初期化に失敗しました。';
                }
            }

            // --- 4. ヘルパー: Float32ArrayをWAVに変換して送信 ---
            function sendAudioAsWav(float32Array) {
                // 16kHzにリサンプリングしたいが、ブラウザのAudioContext依存になるため
                // ここではVADが取得したレート(通常16kHzか44.1kHzか48kHz)のまま送る。
                // pydub側で16kHzに変換するので問題なし。
                
                const wavBuffer = encodeWAV(float32Array, 16000); // VADはデフォルト16kHzで動作することが多いが環境による
                const blob = new Blob([wavBuffer], { type: 'audio/wav' });
                ws.send(blob);
            }

            // 簡易WAVエンコーダー (Float32 -> Int16 PCM WAV)
            function encodeWAV(samples, sampleRate) {
                const buffer = new ArrayBuffer(44 + samples.length * 2);
                const view = new DataView(buffer);

                // RIFF chunk
                writeString(view, 0, 'RIFF');
                view.setUint32(4, 36 + samples.length * 2, true);
                writeString(view, 8, 'WAVE');
                
                // fmt sub-chunk
                writeString(view, 12, 'fmt ');
                view.setUint32(16, 16, true);
                view.setUint16(20, 1, true); // PCM
                view.setUint16(22, 1, true); // Mono
                view.setUint32(24, sampleRate, true);
                view.setUint32(28, sampleRate * 2, true);
                view.setUint16(32, 2, true);
                view.setUint16(34, 16, true);

                // data sub-chunk
                writeString(view, 36, 'data');
                view.setUint32(40, samples.length * 2, true);

                // Write PCM samples
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

            // --- 5. VAD停止 ---
            function stopVAD() {
                vad?.destroy(); 
                vad = null;
                mediaStream?.getTracks().forEach(track => track.stop());
                mediaStream = null;
                isSpeaking = false;
                startButton.disabled = false;
                stopButton.disabled = true;
                interruptButton.disabled = true;
                statusDiv.textContent = 'マイクが停止しました。';
            }
            
            // --- 6. 音声再生制御 (ストリーミング版と同じ) ---
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
                statusDiv.textContent = '音声回答を再生中...';
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
                };

                audioPlayback.appendChild(currentAudio);
            }

            function finishPlayback() {
                console.log("全再生完了。待機状態に戻ります。");
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
                console.log("ユーザー操作により再生を中断します。");
                audioQueue = []; 
                isServerDone = true; 
                finishPlayback();
                statusDiv.textContent = '中断しました。どうぞお話しください。';
            }

            // --- イベント ---
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

# ---------------------------
# サーバー起動
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"サーバーを http://0.0.0.0:{port} で起動します。")
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None)