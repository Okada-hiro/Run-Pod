import os
from tts_wrapper import TTSWrapper

# --- 設定 ---
WORKSPACE_DIR = os.getcwd()
REPO_PATH = os.path.join(WORKSPACE_DIR, "Style_Bert_VITS2")
ASSETS_DIR = os.path.join(REPO_PATH, "model_assets")

# モデルファイル名
MODEL_FILE = "こんにちは_Ref_voice_e3_s3.safetensors"
CONFIG_FILE = "config.json"
STYLE_FILE = "style_vectors.npy"

# アクセント辞書
ACCENT_JSON = "accents.json"

def main():
    # 1. ラッパーの初期化 (モデルロード)
    tts = TTSWrapper(
        repo_path=REPO_PATH,
        model_assets_dir=ASSETS_DIR,
        model_file=MODEL_FILE,
        config_file=CONFIG_FILE,
        style_file=STYLE_FILE
    )

    # 2. アクセント辞書の読み込み
    tts.load_accent_dict(ACCENT_JSON)

    # 3. 音声合成
    text = "こんにちは。今日は、午後から雨が降る予報です。傘は持って行ったほうが安心ですよ。"
    output = "final_output.wav"
    
    tts.infer(text, output, pitch=1.2)

if __name__ == "__main__":
    main()