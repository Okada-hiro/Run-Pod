import os
from tts_wrapper import TTSWrapper

# --- 設定 ---
WORKSPACE_DIR = os.getcwd()
REPO_PATH = os.path.join(WORKSPACE_DIR, "Style_Bert_VITS2")
ASSETS_DIR = os.path.join(REPO_PATH, "model_assets")

# モデルファイル名 (JP-Extra構造を使っているもの)
MODEL_FILE = "female_Ref_voice_e10_s50.safetensors"
CONFIG_FILE = "config.json"
STYLE_FILE = "style_vectors.npy"

def main():
    # 1. 初期化
    tts = TTSWrapper(
        repo_path=REPO_PATH,
        model_assets_dir=ASSETS_DIR,
        model_file=MODEL_FILE,
        config_file=CONFIG_FILE,
        style_file=STYLE_FILE
    )

    # 2. 音声合成 (Plan C)
    # 難しいことは考えず、テキストを渡すだけ。
    # F0の高さや抑揚は f0_controler.py と wrapper 内の設定で決まります。
    
    text = "みなさん、おはようございます。本日は全国的に雲の多い空模様となり、午後にかけて雨の降る地域が増える予報です。"
    output = "final_output_plan_c.wav"

    tts.infer(text, output, 
        pitch=1.0,       # ベースピッチは wrapper 内の base_freq=220.0 で決まるため 1.0 でOK
        intonation=1.0,  # F0を強制上書きするので、モデルの推論係数は無視されます
        style_weight=0.01, # スタイル変動を極力抑える
        
        length=1.0,      # 標準速度
        sdp_ratio=0.0,   # リズムのランダム性を排除
        
        lpf_cutoff=14000 # 高域を残しつつノイズカット
    )

if __name__ == "__main__":
    main()