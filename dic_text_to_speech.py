import os
from tts_wrapper import TTSWrapper

# --- 設定 ---
WORKSPACE_DIR = os.getcwd()
REPO_PATH = os.path.join(WORKSPACE_DIR, "Style_Bert_VITS2")
ASSETS_DIR = os.path.join(REPO_PATH, "model_assets")

# モデルファイル名
MODEL_FILE = "Ref_voice_e300_s2100.safetensors"
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
    text = """こんにちは。まずは、今日の天気と生活に役立つ話題をお伝えします。
本日は、全国的に雲の多い空模様となり、午後にかけて雨の降る地域が増える予報です。午前中は一時的に日差しが届くところもありますが、次第に空が暗くなり、弱い雨が降り始める見込みです。"""
    output = "final_output.wav"

    assist_directive = "コールセンターの自動音声です。はきはきと、明瞭に喋ります。"

    # パラメータを少し緩めて、「辞書優先」にします
    tts.infer(text, output, 
        pitch=1.2, 
        intonation=1.3,        # 1.4は強すぎます。1.0で十分です。
        style_weight=0.1,      # 0.6は強すぎます。0.1に下げてください。
        
        # ★以下の2行を追加してください
        assist_text=assist_directive, 
        assist_text_weight=0.2 # 控えめに効かせる
    )

if __name__ == "__main__":
    main()