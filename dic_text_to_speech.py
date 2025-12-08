# /workspace/text_to_speech.py (v-FineTuned, v2 - パスフラット化)
import torch
from pathlib import Path
from scipy.io.wavfile import write
import scipy.signal
import os
import numpy as np
import sys

# --- ★ここから修正 (ここから) ---

# --- 1. Style-Bert-VITS2 リポジトリのルートを sys.path に追加 ---
# このファイル (new_text_to_speech.py) があるディレクトリ (= /workspace)
WORKSPACE_DIR = os.getcwd()
# git clone したリポジトリのパス
REPO_PATH = os.path.join(WORKSPACE_DIR, "Style_Bert_VITS2") 

# sys.path にリポジトリのルートを追加
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)
    print(f"[INFO] Added to sys.path: {REPO_PATH}")
# ---

# --- 2. Style-Bert-TTS のインポート (パス修正) ---
try:
    # 変更前: from Style_Bert_VITS2.style_bert_vits2.nlp import bert_models
    # 変更後:
    from style_bert_vits2.nlp import bert_models
    
    # 変更前: from Style_Bert_VITS2.style_bert_vits2.constants import Languages
    # 変更後:
    from style_bert_vits2.constants import Languages
    
    # 変更前: from Style_Bert_VITS2.style_bert_vits2.tts_model import TTSModel
    # 変更後:
    from style_bert_vits2.tts_model import TTSModel
    
except ImportError as e:
    print(f"[ERROR] Style-Bert-TTS のインポートに失敗しました。")
    print(f"       REPO_PATH ({REPO_PATH}) が 'Style_Bert_VITS2' として存在するか確認してください。")
    print(f"       エラー詳細: {e}")
    # プログラムを停止させるためにエラーを再送出
    raise

# --- ★ここまで修正 (ここまで) ---

# --- グローバル変数の準備 (変更なし) ---
GLOBAL_TTS_MODEL = None
GLOBAL_SPEAKER_ID = None
GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- ファインチューニングモデルの設定 ---
# 話者名 (config.json の spk2id 内) は "Ref_voice" のまま
FT_SPEAKER_NAME = "Ref_voice" 
# model_assets/ 直下に置くファイル名
FT_MODEL_FILE = "Ref_voice_e3_s936.safetensors"
#FT_MODEL_FILE = "こんにちは_Ref_voice_e3_s3.safetensors"
#FT_MODEL_FILE = "demo_Ref_voice_e3_s9.safetensors"
FT_CONFIG_FILE = "config.json"
FT_STYLE_FILE = "style_vectors.npy"
# ---

try:
    # --- BERTモデルのグローバルロード (変更なし) ---
    print(f"[INFO] Style-Bert-TTS (FT): Loading BERT models...")
    bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    print("[INFO] Style-Bert-TTS (FT): BERT models loaded.")

    # --- ★ 修正: TTSモデルのロードパス ---
    print("[INFO] Style-Bert-TTS (FT): Loading Fine-Tuned TTSModel...")
    assets_root = Path(REPO_PATH) / "model_assets"
    
    # のご要望通り、サブディレクトリ (Ref_voice/) を使わないパスに変更
    model_path = assets_root / FT_MODEL_FILE
    config_path = assets_root / FT_CONFIG_FILE
    style_vec_path = assets_root / FT_STYLE_FILE

    if not all([model_path.exists(), config_path.exists(), style_vec_path.exists()]):
        print(f"[DEBUG] Check Failed (Path: {assets_root}):")
        print(f"  Model:  {model_path} - Exists: {model_path.exists()}")
        print(f"  Config: {config_path} - Exists: {config_path.exists()}")
        print(f"  Style:  {style_vec_path} - Exists: {style_vec_path.exists()}")
        raise FileNotFoundError(f"モデルファイルが 'model_assets/' 直下に見つかりません。")

    GLOBAL_TTS_MODEL = TTSModel(
        model_path=model_path,
        config_path=config_path,
        style_vec_path=style_vec_path,
        device=GLOBAL_DEVICE
    )
    print("[INFO] Style-Bert-TTS (FT): TTSModel loaded.")

    # --- ★ 話者IDの取得 (話者名は 'Ref_voice' で固定) ---
    try:
        GLOBAL_SPEAKER_ID = GLOBAL_TTS_MODEL.spk2id[FT_SPEAKER_NAME]
        print(f"[INFO] Style-Bert-TTS (FT): Found speaker: {FT_SPEAKER_NAME} (ID: {GLOBAL_SPEAKER_ID})")
    except KeyError:
        print(f"[ERROR] 話者 '{FT_SPEAKER_NAME}' が {config_path} (spk2id) に見つかりません。")
        print(f"利用可能な話者: {list(GLOBAL_TTS_MODEL.spk2id.keys())}")
        raise

    # --- ★追加: ウォームアップ処理 (ここから) ---
    print("[INFO] Style-Bert-TTS (FT): Performing Warm-up (dummy inference)...")
    try:
        # 「あ」と一瞬だけ生成させて、CUDAの初期化コストをここで払っておく
        # 結果は使わないので捨てる
        _ = GLOBAL_TTS_MODEL.infer(
            text="あ",
            language=Languages.JP,
            speaker_id=GLOBAL_SPEAKER_ID,
            style="Neutral",
            style_weight=0.7,
            sdp_ratio=0.2,
            noise=0.6,
            noise_w=0.8,
            length=0.1 # 最短で終わらせる
        )
        print("[INFO] Style-Bert-TTS (FT): Warm-up complete! (Ready for fast inference)")
    except Exception as wu_e:
        print(f"[WARNING] Warm-up failed (will proceed anyway): {wu_e}")
    # --- ★追加: ウォームアップ処理 (ここまで) ---
    print("[INFO] Style-Bert-TTS (FT): All models ready.")

except Exception as e:
    print(f"[ERROR] Style-Bert-TTS (FT) モデルのグローバルロードに失敗しました: {e}")
    import traceback
    traceback.print_exc()

# --- synthesize_speech 関数 (変更なし) ---
def synthesize_speech(text_to_speak: str, output_wav_path: str, prompt_text: str = None):
    if GLOBAL_TTS_MODEL is None or GLOBAL_SPEAKER_ID is None:
        print("[ERROR] Style-Bert-TTS (FT) モデルがロードされていません。")
        return False
        
    try:
        print(f"[DEBUG] Style-Bert-TTS (FT): 音声合成開始... '{text_to_speak[:20]}...'")
        
        sr, audio_data = GLOBAL_TTS_MODEL.infer(
            text=text_to_speak,
            language=Languages.JP,
            speaker_id=GLOBAL_SPEAKER_ID, # グローバルな話者IDを使用
            style="Neutral",
            style_weight=0.7,
            sdp_ratio=0.2,
            noise=0.6,
            noise_w=0.8,
            length=1.0
        )
        
        # 16-bit PCM への変換
        if audio_data.dtype != np.int16:
            audio_norm = audio_data / np.abs(audio_data).max()
            audio_int16 = (audio_norm * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data

        # WAV ファイルとして保存
        write(output_wav_path, sr, audio_int16)
        
        print(f"[SUCCESS] Style-Bert-TTS (FT) 音声を保存しました: {output_wav_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Style-Bert-TTS (FT) 音声生成中に例外: {e}")
        import traceback
        traceback.print_exc()
        return False

import io
from scipy.io.wavfile import write as scipy_write

def synthesize_speech_to_memory(text_to_speak: str) -> bytes:
    """
    音声をファイルに保存せず、バイトデータとして直接返す（Scipy高速化版）
    """
    if GLOBAL_TTS_MODEL is None or GLOBAL_SPEAKER_ID is None:
        return None
        
    try:
        # 1. 推論実行
        sr, audio_data = GLOBAL_TTS_MODEL.infer(
            text=text_to_speak,
            language=Languages.JP,
            speaker_id=GLOBAL_SPEAKER_ID,
            style="Neutral",
            style_weight=0.7,
            sdp_ratio=0.2,
            noise=0.6,
            noise_w=0.8,
            length=1.0
        )
        
        # 2. 16bit PCMに変換 (正規化)
        if audio_data.dtype != np.int16:
            audio_norm = audio_data / np.abs(audio_data).max()
            # floatのままリサンプリングするためにここではまだint16にしない
            audio_float = audio_norm
        else:
            audio_float = audio_data.astype(np.float32) / 32768.0

        # --- ★高速化ポイント: Scipyでリサンプリング (エラー回避版) ---
        target_sr = 16000
        if sr > target_sr:
            # サンプル数を計算
            num_samples = int(len(audio_float) * float(target_sr) / sr)
            # Scipyでリサンプリング (librosaよりトラブルが少ない)
            audio_resampled = scipy.signal.resample(audio_float, num_samples)
            
            # int16に変換
            audio_int16 = (audio_resampled * 32767).astype(np.int16)
            sr = target_sr
        else:
            # リサンプリング不要な場合
            audio_int16 = (audio_float * 32767).astype(np.int16)
        # -------------------------------------------------------
        # 新しいコード（Raw PCM返却）:
        # tobytes() でメモリ上の配列をそのままバイナリ化します
        return audio_int16.tobytes()
        

    except Exception as e:
        print(f"[ERROR] Memory Synthesis Error: {e}")
        # エラー時はNoneを返すか、元のファイルを返すなど安全策をとる
        return None
    

def synthesize_with_manual_tone(text: str, target_word: str, correct_tone_pattern: list):
    """
    特定の単語のアクセント(tone_ids)を強制的に書き換えて音声合成する関数
    
    Args:
        text: 全文 (例: "こんにちは。今日は雨です。")
        target_word: 修正したい単語 (例: "こんにちは")
        correct_tone_pattern: 強制するトーン配列 (例: [0, 1, 1, 1, 1]) 
                              0=低, 1=高
    """
    model = GLOBAL_TTS_MODEL
    if model is None:
        print("[ERROR] Model not loaded.")
        return None

    # 1. テキスト解析 (OpenJTalk + BERT)
    # 通常の infer 内部で行われている処理を分解して呼び出します
    with torch.no_grad():
        # テキストから音素(phoneme_ids)とアクセント(tone_ids)などを取得
        phoneme_ids, dist_info, tone_ids, language_ids, bert, ja_bert, en_bert, emo_embedding, style_embedding = \
            model.get_text(text, Languages.JP, model.spk2id[GLOBAL_SPEAKER_ID], "Neutral", 0.1)

        # 2. アクセント(tone_ids)の強制書き換え
        # tone_ids は tensor([0, 1, 0, ...]) の形をしています
        
        # 音素列（文字列）を取得して、ターゲット単語の位置を探す
        # ※ model.frontend.phone_id_to_text のような機能が必要ですが、
        #    簡易的に音素数の一致や、OpenJTalkの解析結果との照合で特定します。
        
        # 【簡易実装】
        # ここでは「こんにちは」が文頭にあると仮定し、
        # 最初の5モーラ（「こ」「ん」「に」「ち」「は」）を書き換えます。
        # target_word の長さと tone_ids の対応は厳密には音素数依存ですが、
        # 日本語は概ね 1モーラ=1トーン です。
        
        if text.startswith(target_word):
            # 修正用パターンの長さ
            patch_len = len(correct_tone_pattern)
            
            # 元のトーン配列を表示（デバッグ用）
            original_tones = tone_ids[0, :patch_len].tolist()
            print(f"[DEBUG] Original Tones: {original_tones}")
            
            # 上書き実行
            new_tones = torch.tensor(correct_tone_pattern, device=model.device)
            tone_ids[0, :patch_len] = new_tones
            
            print(f"[DEBUG] Patched Tones : {tone_ids[0, :patch_len].tolist()}")
        else:
            print(f"[WARNING] 文頭が '{target_word}' ではないため、安全のため修正をスキップしました。")

        # 3. 音声生成 (書き換えた tone_ids を使用)
        # SDP(ランダム性)を0にして、指定したトーンを確実に守らせる
        x_tst = phoneme_ids.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phoneme_ids.size(0)]).to(model.device)
        
        audio = model.net_g.infer(
            x=x_tst,
            x_lengths=x_tst_lengths,
            bert=bert,
            tone=tone_ids, # ★書き換えたトーン
            language=language_ids,
            ja_bert=ja_bert,
            en_bert=en_bert,
            emo_embedding=emo_embedding,
            style_embedding=style_embedding,
            noise_scale=0.1,    # noise (0.6 -> 0.1)
            length_scale=1.0,
            noise_scale_w=0.1,  # noise_w (0.8 -> 0.1)
            sdp_ratio=0.0,      # ★ランダム性を排除
        )[0][0, 0].data.cpu().float().numpy()

        return audio

# --- 呼び出し例 ---
# 実際に音声合成を行う箇所で以下のように使います
if __name__ == "__main__":
    # アクセント定義: 0=低, 1=高
    # こんにちは (平板) = 低 高 高 高 高
    FLAT_ACCENT = [0, 1, 1, 1, 1] 

    audio_data = synthesize_with_manual_tone(
        "こんにちは。今日は雨です。",
        "こんにちは",
        FLAT_ACCENT
    )
    
    # WAV保存
    if audio_data is not None:
        # float -> int16変換
        audio_int16 = (audio_data * 32767).astype(np.int16)
        write("/workspace/fixed_accent.wav", 44100, audio_int16) # srはモデルに合わせてください
        print("保存しました: /workspace/fixed_accent.wav")
    
