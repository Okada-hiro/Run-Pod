import sys
from pathlib import Path
import torch
from scipy.io.wavfile import write

# 必要なモジュールをインポート
# app.py や tts_model.py に倣っています
from Style_Bert_VITS2.style_bert_vits2.tts_model import TTSModel
from Style_Bert_VITS2.style_bert_vits2.constants import Languages
import Style_Bert_VITS2.style_bert_vits2.nlp.bert_models as bert_models

# --- 1. 設定項目 (ご自身の環境に合わせて変更) ---

# ファインチューニングしたモデル名 (Data/ や model_assets/ 以下のフォルダ名)
MODEL_NAME = "Ref_voice"

# 使用するモデルファイル (model_assets/Ref_voice/ の中にあるファイル)
# 学習ログから "Ref_voice_e3_s192.safetensors" だったと推測
MODEL_FILE = "Ref_voice_e3_s192.safetensors"

# 推論に使用するデバイス (GPUがある場合は "cuda")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 読み上げさせたいテキスト
TEXT_TO_SPEAK = "こんにちは。これは、ファインチューニングしたモデルによる音声合成のテストです。"

# 保存するWAVファイル名
OUTPUT_WAV_PATH = "test_output.wav"

# --- 2. BERTモデルの明示的なロード ---
# bert_models.py の設計思想に基づき、
# 推論前に使用する言語のBERTモデルをロードします。
try:
    print(f"Loading BERT model for JP...")
    bert_models.load_model(
        Languages.JP, 
        device_map=DEVICE, 
        pretrained_model_name_or_path="ku-nlp/deberta-v2-large-japanese-char-wwm"
    )
    print(f"Loading BERT tokenizer for JP...")
    bert_models.load_tokenizer(
        Languages.JP, 
        pretrained_model_name_or_path="ku-nlp/deberta-v2-large-japanese-char-wwm"
    )
    print("BERT models loaded successfully.")

except Exception as e:
    print(f"Error loading BERT models: {e}")
    print("Please ensure 'ku-nlp/deberta-v2-large-japanese-char-wwm' is accessible or downloaded.")
    sys.exit(1)


# --- 3. TTSモデルの初期化 ---
# tts_model.py の TTSModel クラスを初期化します
print("Initializing TTSModel...")

# モデルへのパスを解決 (app.py の model_holder の動作を簡易的に再現)
model_assets_dir = Path("model_assets")
model_path = model_assets_dir / MODEL_NAME / MODEL_FILE
config_path = model_assets_dir / MODEL_NAME / "config.json"
style_vec_path = model_assets_dir / MODEL_NAME / "style_vectors.npy"

# ファイルの存在チェック
if not all([model_path.exists(), config_path.exists(), style_vec_path.exists()]):
    print(f"Error: Required files not found in {model_assets_dir / MODEL_NAME}")
    print(f"  model_path: {model_path} (Exists: {model_path.exists()})")
    print(f"  config_path: {config_path} (Exists: {config_path.exists()})")
    print(f"  style_vec_path: {style_vec_path} (Exists: {style_vec_path.exists()})")
    sys.exit(1)

# TTSModel クラスのインスタンスを作成
model = TTSModel(
    model_path=model_path,
    config_path=config_path,
    style_vec_path=style_vec_path,
    device=DEVICE
)

# モデルをGPU/CPUにロード (infer時に自動で呼ばれますが、明示的に呼ぶことも可能です)
# model.load()
print("TTSModel initialized.")


# --- 4. 話者IDの取得 ---
# config.json から spk2id マップを読み込み、話者名からIDを取得します
try:
    # "Ref_voice" という名前の話者IDを取得
    speaker_name = MODEL_NAME 
    speaker_id = model.spk2id[speaker_name]
    print(f"Found speaker: {speaker_name} (ID: {speaker_id})")
except KeyError:
    print(f"Error: Speaker '{speaker_name}' not found in config.json (spk2id).")
    print(f"Available speakers: {list(model.spk2id.keys())}")
    sys.exit(1)


# --- 5. 音声合成の実行 ---
print(f"Generating audio for text: {TEXT_TO_SPEAK}")

# TTSModel.infer メソッドを実行して音声合成
sampling_rate, audio_data = model.infer(
    text=TEXT_TO_SPEAK,
    language=Languages.JP,
    speaker_id=speaker_id,
    # --- 以下はオプション (app.py で設定可能な項目) ---
    sdp_ratio=0.2,       # デフォルト値
    noise=0.6,           # デフォルト値
    noise_w=0.8,         # デフォルト値
    length=1.0,          # デフォルト値
    line_split=False,    # デフォルト値
    style="Neutral",     # デフォルト値
    style_weight=0.7     # デフォルト値
)

print("Audio generation complete.")


# --- 6. 音声データをファイルに保存 ---
# infer メソッドは 16bit PCM のNumpy配列を返します
write(OUTPUT_WAV_PATH, sampling_rate, audio_data)
print(f"Audio saved successfully to {OUTPUT_WAV_PATH}")