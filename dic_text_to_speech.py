import sys
import os
import torch
import numpy as np
import pyopenjtalk
from scipy.io.wavfile import write
from pathlib import Path

# --- 1. パス設定 ---
WORKSPACE_DIR = os.getcwd()
REPO_PATH = os.path.join(WORKSPACE_DIR, "Style_Bert_VITS2")
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

try:
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.tts_model import TTSModel
except ImportError as e:
    print(f"[ERROR] ライブラリのインポートに失敗しました: {e}")
    sys.exit(1)

# --- 設定 ---
FT_MODEL_FILE = "こんにちは_Ref_voice_e3_s3.safetensors"
FT_CONFIG_FILE = "config.json"
FT_STYLE_FILE = "style_vectors.npy"
GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. 独自のアクセント解析関数 (ライブラリ非依存) ---
def custom_g2p_with_patch(text, target_word="こんにちは", patch_tones=None):
    """
    pyopenjtalkを使って音素とトーンを生成し、特定の単語だけトーンを強制置換する
    """
    # 1. OpenJTalkで音素とアクセント情報を抽出
    labels = pyopenjtalk.extract_fullcontext(text)
    
    phones = []
    tones = []
    
    # OpenJTalkのラベルから音素と簡易トーンを生成
    for label in labels:
        # ラベル例: "xx^xx-sil+pau=xx..."
        parts = label.split('/')
        p3 = label.split('-')[1].split('+')[0] # 音素
        
        # Style-Bert-VITS2用の音素変換
        if p3 == 'sil': p3 = 'pau'
        
        phones.append(p3)
        
        # トーンの生成（簡易ロジック: とりあえず全て0にする）
        # ※本来は複雑な計算が必要ですが、今回は「修正対象以外」はどうでもよく、
        #   「修正対象」だけ直せれば良いため、ベースはモデルの予測に任せたいところですが、
        #   given_toneを渡す以上、全埋めする必要があります。
        #   ここでは「平板(0111...)」基調のダミートーンを作ります。
        #   （厳密なイントネーションが必要な場合、ここを作り込む必要がありますが、
        #     挨拶などの短文ならこれで十分通じます）
        tones.append(0) 

    # 2. 「こんにちは」のアクセントを強制適用
    # こんにちは (5モーラ) の音素: k, o, N, n, i, ch, i, w, a (9音素)
    # 平板アクセント: [0, 0, 1, 1, 1, 1, 1, 1, 1] (1拍目低、2拍目以降高)
    
    # ターゲットの音素列を定義
    if target_word == "こんにちは":
        target_phones = ['k', 'o', 'N', 'n', 'i', 'ch', 'i', 'w', 'a']
        if patch_tones is None:
            patch_tones = [0, 0, 1, 1, 1, 1, 1, 1, 1] # 平板
    else:
        # 必要なら他も定義
        target_phones = []

    # 検索と置換
    if target_phones:
        # リスト内でターゲット音素列が始まる場所を探す
        match_idx = -1
        seq_len = len(target_phones)
        for i in range(len(phones) - seq_len + 1):
            if phones[i : i + seq_len] == target_phones:
                match_idx = i
                break
        
        if match_idx != -1:
            print(f"[INFO] '{target_word}' を検出 (Index: {match_idx})。トーンを強制修正します。")
            for j, tone_val in enumerate(patch_tones):
                tones[match_idx + j] = tone_val
        else:
            print(f"[WARNING] '{target_word}' の音素列が見つかりませんでした。")
            print(f"解析された音素: {phones}")

    return phones, tones

# --- モデルロード ---
def load_model():
    print("[INFO] Loading BERT...")
    # BERTロード
    bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

    assets_root = Path(REPO_PATH) / "model_assets"
    model_path = assets_root / FT_MODEL_FILE
    config_path = assets_root / FT_CONFIG_FILE
    style_vec_path = assets_root / FT_STYLE_FILE

    print(f"[INFO] Loading TTS Model from {model_path}...")
    model = TTSModel(
        model_path=model_path,
        config_path=config_path,
        style_vec_path=style_vec_path,
        device=GLOBAL_DEVICE
    )
    return model

# --- メイン処理 ---
def main():
    try:
        model = load_model()
        
        # 修正したいテキスト
        TEXT = "こんにちは。今日は雨ですね。"
        OUTPUT_WAV = "/workspace/Run-Pod/fixed_accent_final.wav"
        
        print(f"--- 音声合成開始: {TEXT} ---")
        
        # 1. 自前のロジックで音素と「修正済みトーン」を作成
        phones, tones = custom_g2p_with_patch(TEXT)
        
        print(f"[DEBUG] 使用する音素: {phones}")
        print(f"[DEBUG] 使用するトーン: {tones}")
        
        # 2. 推論 (given_phone / given_tone を指定)
        # BERTは内部で text から計算されるので、文脈理解は生きています。
        # アクセントだけが tones 配列に従います。
        sr, audio_data = model.infer(
            text=TEXT,
            language=Languages.JP,
            
            # ★強制指定★
            given_phone=phones,
            given_tone=tones,
            
            style="Neutral",
            style_weight=0.1,
            sdp_ratio=0.0, # 揺らぎをゼロにする
            noise=0.1,
            noise_w=0.1,
            length=1.0
        )

        if audio_data.dtype != np.int16:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data
            
        if os.path.exists(OUTPUT_WAV):
            os.remove(OUTPUT_WAV)
            
        write(OUTPUT_WAV, sr, audio_int16)
        print(f"[SUCCESS] 保存しました: {OUTPUT_WAV}")
        
    except Exception as e:
        print(f"[ERROR] エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()