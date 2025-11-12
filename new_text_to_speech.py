# /workspace/text_to_speech.py (v-FineTuned: Ref_voice)
import torch
from pathlib import Path
from scipy.io.wavfile import write
import os
import numpy as np
import sys

# --- Style-Bert-TTS のインポート ---
from Style_Bert_VITS2.style_bert_vits2.nlp import bert_models
from Style_Bert_VITS2.style_bert_vits2.constants import Languages
from Style_Bert_VITS2.style_bert_vits2.tts_model import TTSModel

# ---
# グローバル変数の準備
# ---
GLOBAL_TTS_MODEL = None
GLOBAL_SPEAKER_ID = None # ★ ファインチューニングモデル用の話者ID
GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- ★ 1. ファインチューニングモデルの設定 ---
FT_MODEL_NAME = "Ref_voice"
FT_MODEL_FILE = "Ref_voice_e3_s192.safetensors"
# ---

try:
    # --- 2. BERTモデルのグローバルロード ---
    print(f"[INFO] Style-Bert-TTS (FT): Loading BERT models...")
    bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    print("[INFO] Style-Bert-TTS (FT): BERT models loaded.")

    # --- 3. TTSモデルのグローバルロード ---
    print("[INFO] Style-Bert-TTS (FT): Loading Fine-Tuned TTSModel...")
    assets_root = Path("model_assets")
    
    model_path = assets_root / FT_MODEL_NAME / FT_MODEL_FILE
    config_path = assets_root / FT_MODEL_NAME / "config.json"
    style_vec_path = assets_root / FT_MODEL_NAME / "style_vectors.npy"

    if not all([model_path.exists(), config_path.exists(), style_vec_path.exists()]):
        print(f"[DEBUG] Check Failed (Path: {assets_root / FT_MODEL_NAME}):")
        print(f"  Model:  {model_path} - Exists: {model_path.exists()}")
        print(f"  Config: {config_path} - Exists: {config_path.exists()}")
        print(f"  Style:  {style_vec_path} - Exists: {style_vec_path.exists()}")
        raise FileNotFoundError(f"ファインチューニングモデル '{FT_MODEL_NAME}' のファイルが見つかりません。")

    GLOBAL_TTS_MODEL = TTSModel(
        model_path=model_path,
        config_path=config_path,
        style_vec_path=style_vec_path,
        device=GLOBAL_DEVICE
    )
    print("[INFO] Style-Bert-TTS (FT): TTSModel loaded.")

    # --- 4. ★ 話者IDの取得 ---
    try:
        # ご要望 通り、話者名 'Ref_voice' で固定します
        speaker_name = FT_MODEL_NAME
        GLOBAL_SPEAKER_ID = GLOBAL_TTS_MODEL.spk2id[speaker_name]
        print(f"[INFO] Style-Bert-TTS (FT): Found speaker: {speaker_name} (ID: {GLOBAL_SPEAKER_ID})")
    except KeyError:
        print(f"[ERROR] 話者 '{speaker_name}' が config.json (spk2id) に見つかりません。")
        print(f"利用可能な話者: {list(GLOBAL_TTS_MODEL.spk2id.keys())}")
        raise

    print("[INFO] Style-Bert-TTS (FT): All models ready.")

except Exception as e:
    print(f"[ERROR] Style-Bert-TTS (FT) モデルのグローバルロードに失敗しました: {e}")
    import traceback
    traceback.print_exc()

# ---
# watch_and_transcribe.py から呼ばれるメイン関数
# ---
def synthesize_speech(text_to_speak: str, output_wav_path: str, prompt_text: str = None):
    """
    ファインチューニング済み Style-Bert-TTS でテキストをwavに変換する
    """
    if GLOBAL_TTS_MODEL is None or GLOBAL_SPEAKER_ID is None:
        print("[ERROR] Style-Bert-TTS (FT) モデルがロードされていません。")
        return False
        
    try:
        print(f"[DEBUG] Style-Bert-TTS (FT): 音声合成開始... '{text_to_speak[:20]}...'")
        
        # の推論コード
        sr, audio_data = GLOBAL_TTS_MODEL.infer(
            text=text_to_speak,
            language=Languages.JP,
            speaker_id=GLOBAL_SPEAKER_ID, # ★ ロード済みのIDを使用
            style="Neutral",      # スタイルは適宜調整
            style_weight=0.7,
            sdp_ratio=0.2,
            noise=0.6,
            noise_w=0.8,
            length=1.0
        )
        
        # 16-bit PCM への変換 (以前の修正 と同様)
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

# --- ★ 追加: 単体テスト ---
if __name__ == "__main__":
    print("\n--- Style-Bert-TTS (FineTuned) 単体テスト ---")
    
    if GLOBAL_TTS_MODEL is None:
        print("[FAIL] モデルのグローバルロードに失敗したため、テストを中止します。")
    else:
        #
        TEST_TEXT = "こんにちは。これは、ファインチューニングしたモデルによる音声合成のテストです。"
        TEST_OUTPUT = "/workspace/test_finetuned_output.wav"
        
        print(f"テキスト: {TEST_TEXT}")
        print(f"出力先: {TEST_OUTPUT}")
        
        if os.path.exists(TEST_OUTPUT):
            os.remove(TEST_OUTPUT)
            
        success = synthesize_speech(TEST_TEXT, TEST_OUTPUT)
        
        if success and os.path.exists(TEST_OUTPUT):
            print(f"[SUCCESS] テストファイルが正常に生成されました。")
        else:
            print("[FAIL] テストに失敗しました。")