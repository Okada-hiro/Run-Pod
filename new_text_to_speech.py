# /workspace/new_text_to_speech.py (v4 - Instant Switch & Pre-load)
import torch
from pathlib import Path
from scipy.io.wavfile import write
import scipy.signal
import os
import numpy as np
import sys
import json
import io

# ★追加: アクセント解析用
import pyopenjtalk

# --- 1. Style-Bert-VITS2 リポジトリのルートを sys.path に追加 ---
WORKSPACE_DIR = os.getcwd()
REPO_PATH = os.path.join(WORKSPACE_DIR, "Style_Bert_VITS2") 

if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)
    print(f"[INFO] Added to sys.path: {REPO_PATH}")

# --- 2. Style-Bert-TTS のインポート ---
try:
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.tts_model import TTSModel
except ImportError as e:
    print(f"[ERROR] Style-Bert-TTS のインポートに失敗しました。")
    print(f"       REPO_PATH ({REPO_PATH}) が 'Style_Bert_VITS2' として存在するか確認してください。")
    raise

# --- グローバル変数 ---
LOADED_MODELS = {}      # { "default": TTSModel, "second": TTSModel, ... }
LOADED_SPEAKER_IDS = {} # { "default": 0, "second": 0, ... }
CURRENT_MODEL_KEY = "default"
GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GLOBAL_ACCENT_RULES = {} 

ACCENT_JSON_FILE = "accents.json"

# --- ★モデルカタログ (ここで話者を定義します) ---
MODEL_CATALOG = {
    "default": {
        "model_file": "Ref_voice_e56_s500.safetensors",
        "config_file": "config.json",
        "style_file": "style_vectors.npy",
        "speaker_name": "Ref_voice", # モデル内の話者名(ID特定用)
        "params": {
            "pitch": 1.2, 
            "intonation": 1.3,
            "length": 0.9,
            "assist_text": "アナウンサーです。はきはきと、明瞭に喋ります。",
            "lpf_cutoff": 9000
        }
    },
    # ★追加したい2つ目のモデル
    "second": {
        "model_file": "Second_Voice.safetensors", # ★実際のファイル名に変更してください
        "config_file": "config.json",             # 違うなら変更
        "style_file": "style_vectors.npy",        # 違うなら変更
        "speaker_name": "No.2",                   # ★モデル内の正しい話者名
        "params": {
            "pitch": 1.2, 
            "intonation": 1.0,
            "length": 0.8,
            "assist_text": "落ち着いた声で喋ります。",
            "lpf_cutoff": 8000
        }
    }
}

# --- ヘルパー関数群 ---

def load_accent_dict(json_path):
    global GLOBAL_ACCENT_RULES
    if not os.path.exists(json_path):
        # print(f"[WARNING] Accent JSON not found: {json_path}")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        count = 0
        for word, tones in data.items():
            phones = pyopenjtalk.g2p(word, kana=False).split(" ")
            phones = [p for p in phones if p not in ('pau', 'sil')]
            GLOBAL_ACCENT_RULES[word] = {"phones": phones, "tones": tones}
            count += 1
        print(f"[INFO] Loaded {count} accent rules from {json_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load accent dict: {e}")

def _parse_openjtalk_accent(labels):
    phones = []
    tones = []
    for label in labels:
        parts = label.split('/')
        p3 = label.split('-')[1].split('+')[0]
        if p3 == 'sil': p3 = 'pau'
        phones.append(p3)
        if p3 == 'pau':
            tones.append(0)
            continue
        try:
            a_part = parts[1]
            if 'A:' not in a_part:
                tones.append(0)
                continue
            nums = a_part.split(':')[1].split('+')
            a1 = int(nums[0])
            a2 = int(nums[1])
            is_high = 0
            if a1 == 0:
                if a2 == 1: is_high = 0
                else:       is_high = 1
            else:
                if a2 <= a1:
                    if a2 == 1 and a1 > 1: is_high = 0
                    else: is_high = 1
                else:
                    is_high = 0
            tones.append(is_high)
        except:
            tones.append(0)
    return phones, tones

def _g2p_and_patch(text):
    labels = pyopenjtalk.extract_fullcontext(text)
    phones, tones = _parse_openjtalk_accent(labels)

    for word, rule in GLOBAL_ACCENT_RULES.items():
        target_phones = rule['phones']
        target_tones = rule['tones']
        if len(target_phones) != len(target_tones): continue

        seq_len = len(target_phones)
        for i in range(len(phones) - seq_len + 1):
            if phones[i : i + seq_len] == target_phones:
                for j, t_val in enumerate(target_tones):
                    tones[i + j] = t_val
    return phones, tones

def _apply_lowpass_scipy(audio_numpy, sr, cutoff):
    if cutoff <= 0 or cutoff >= sr / 2: return audio_numpy
    audio_numpy = np.squeeze(audio_numpy)
    try:
        nyquist = 0.5 * sr
        normal_cutoff = cutoff / nyquist
        sos = scipy.signal.butter(5, normal_cutoff, btype='low', analog=False, output='sos')
        filtered = scipy.signal.sosfilt(sos, audio_numpy)
        return filtered
    except:
        return audio_numpy

# --- 初期化処理 (一括ロード) ---
def initialize_all_models():
    global LOADED_MODELS, LOADED_SPEAKER_IDS, CURRENT_MODEL_KEY

    # 1. 共通BERTのロード
    print(f"[INFO] Loading BERT models...")
    bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    
    # 2. アクセント辞書
    load_accent_dict(os.path.join(WORKSPACE_DIR, ACCENT_JSON_FILE))

    assets_root = Path(REPO_PATH) / "model_assets"

    # 3. カタログ全ロード
    for key, conf in MODEL_CATALOG.items():
        print(f"[INFO] Pre-loading model: '{key}' ...")
        try:
            model_path = assets_root / conf["model_file"]
            config_path = assets_root / conf["config_file"]
            style_vec_path = assets_root / conf["style_file"]

            if not model_path.exists():
                print(f"[WARNING] Model file not found: {model_path}. Skipping '{key}'.")
                continue

            # モデルインスタンス作成 (VRAM消費)
            model = TTSModel(
                model_path=model_path,
                config_path=config_path,
                style_vec_path=style_vec_path,
                device=GLOBAL_DEVICE
            )
            
            # 話者ID特定
            spk_name = conf["speaker_name"]
            if spk_name in model.spk2id:
                spk_id = model.spk2id[spk_name]
            else:
                print(f"[WARNING] Speaker '{spk_name}' not found in {key}. Using ID 0.")
                spk_id = 0

            LOADED_MODELS[key] = model
            LOADED_SPEAKER_IDS[key] = spk_id
            
            # ウォームアップ (初回推論の遅延を消す)
            _ = model.infer(text="あ", speaker_id=spk_id, length=0.1)

        except Exception as e:
            print(f"[ERROR] Failed to load {key}: {e}")

    print(f"[INFO] All models loaded. Models: {list(LOADED_MODELS.keys())}")
    print(f"[INFO] Current selection: {CURRENT_MODEL_KEY}")


# --- ★モデル切り替え関数 (0秒スイッチ) ---
def switch_model(model_key: str):
    global CURRENT_MODEL_KEY
    if model_key in LOADED_MODELS:
        CURRENT_MODEL_KEY = model_key
        print(f"[INFO] Switched to model: {model_key} (Instant)")
        return True
    else:
        print(f"[ERROR] Model '{model_key}' is not loaded.")
        return False

# --- 実行時に初期化 ---
initialize_all_models()


# --- 音声合成 (メモリ版: Chatbot用) ---
def synthesize_speech_to_memory(text_to_speak: str) -> bytes:
    # 現在選択されているモデルを取得
    model = LOADED_MODELS.get(CURRENT_MODEL_KEY)
    spk_id = LOADED_SPEAKER_IDS.get(CURRENT_MODEL_KEY)
    conf = MODEL_CATALOG.get(CURRENT_MODEL_KEY, {}).get("params", {})
    
    if model is None:
        print("[ERROR] No model loaded (Current Key is invalid).")
        return None
        
    try:
        # パラメータ取得
        pitch = conf.get("pitch", 1.2)
        intonation = conf.get("intonation", 1.2)
        length = conf.get("length", 0.8)
        assist_text = conf.get("assist_text", "")
        lpf_cutoff = conf.get("lpf_cutoff", 9000)

        # 1. アクセント修正
        phones, tones = _g2p_and_patch(text_to_speak)

        # 2. 推論
        sr, audio_data = model.infer(
            text=text_to_speak,
            given_phone=phones,
            given_tone=tones,
            speaker_id=spk_id,
            pitch_scale=pitch,
            intonation_scale=intonation,
            length=length,
            assist_text=assist_text,
            use_assist_text=True,
            style="Neutral", style_weight=0.1, sdp_ratio=0, noise=0.6, noise_w=0.8
        )

        # 正規化
        if not isinstance(audio_data, np.ndarray):
            audio_data = audio_data.cpu().numpy()
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.5: audio_data = audio_data / 32768.0

        # 3. LPF
        audio_data = _apply_lowpass_scipy(audio_data, sr, lpf_cutoff)

        # 4. リサンプリング (16kHzへ) -> 通信量削減のため必須
        target_sr = 16000
        if sr > target_sr:
            num_samples = int(len(audio_data) * float(target_sr) / sr)
            audio_data = scipy.signal.resample(audio_data, num_samples)

        # int16変換
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)

        return audio_int16.tobytes()

    except Exception as e:
        print(f"[ERROR] Synthesis Error: {e}")
        return None

# --- 音声合成 (ファイル保存版: 互換性のため残す) ---
def synthesize_speech(text_to_speak: str, output_wav_path: str, prompt_text: str = None):
    # メモリ版と同じロジックでバイナリを作り、ファイルに書くだけにする
    wav_bytes = synthesize_speech_to_memory(text_to_speak)
    if wav_bytes:
        try:
            write(output_wav_path, 16000, np.frombuffer(wav_bytes, dtype=np.int16))
            print(f"[SUCCESS] Saved to {output_wav_path}")
            return True
        except Exception as e:
            print(f"[ERROR] File Write Error: {e}")
            return False
    return False

if __name__ == "__main__":
    print("\n--- Test ---")
    # テスト
    synthesize_speech("これはテストです。", "test_output.wav")
    
    # 切り替えテスト
    if "second" in MODEL_CATALOG:
        switch_model("second")
        synthesize_speech("これは2番目の声です。", "test_output_2.wav")