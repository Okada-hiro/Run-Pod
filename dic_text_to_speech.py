import sys
import os
import torch
import numpy as np
from scipy.io.wavfile import write
from pathlib import Path

# --- 1. パス設定 ---
WORKSPACE_DIR = os.getcwd()
REPO_PATH = os.path.join(WORKSPACE_DIR, "Style_Bert_VITS2")
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

try:
    # モデル関連
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.tts_model import TTSModel
    
    # ★重要: 音素変換機能 (g2p) をインポート
    # これを使ってテキストを「音素」と「トーン」のリストに変換します
    from style_bert_vits2.nlp.japanese.phoneme import g2p
    
except ImportError as e:
    print(f"[ERROR] ライブラリのインポートに失敗: {e}")
    sys.exit(1)

# --- 設定 ---
# モデルパス等はご自身の環境に合わせてください
FT_MODEL_FILE = "こんにちは_Ref_voice_e3_s3.safetensors" 
FT_CONFIG_FILE = "config.json"
FT_STYLE_FILE = "style_vectors.npy"
GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- モデルロード (前回と同じ) ---
def load_model():
    print("[INFO] Loading BERT...")
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

# --- ★ここが修正版の核心です ---
def synthesize_with_manual_accent(model, text: str, output_path: str):
    """
    g2pで音素とトーンを取得し、「こんにちは」の部分だけ強制的に平板アクセントに書き換える
    """
    print(f"--- テキスト: {text} ---")

    # 1. テキストを音素(phones)とアクセント(tones)に変換
    # phones: ['pau', 'k', 'o', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', 'pau', ...]
    # tones:  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, ...]
    phones, tones, _ = g2p(text)

    print("[DEBUG] 元の音素:", phones)
    print("[DEBUG] 元のトーン:", tones)

    # 2. 「こんにちは」の音素パターンを探す
    # 「こんにちは」は通常 ['k', 'o', 'N', 'n', 'i', 'ch', 'i', 'w', 'a'] の並びになります
    target_phones = ['k', 'o', 'N', 'n', 'i', 'ch', 'i', 'w', 'a']
    
    # リストの中で target_phones が始まる位置を探す
    start_index = -1
    for i in range(len(phones) - len(target_phones) + 1):
        if phones[i : i + len(target_phones)] == target_phones:
            start_index = i
            break
    
    if start_index != -1:
        print(f"[INFO] 「こんにちは」をインデックス {start_index} で検出しました。アクセントを修正します。")
        
        # 3. アクセントを平板型に書き換え (Low-High-High-High...)
        # 日本語の平板: 1拍目(k,o)は低(0), それ以降は高(1)
        # k(0), o(0), N(1), n(1), i(1), ch(1), i(1), w(1), a(1)
        # ※トーンは音素ごとに付きます
        
        # 書き換えるトーン列 (平板)
        # k, o, N, n, i, ch, i, w, a
        new_tones_segment = [0, 0, 1, 1, 1, 1, 1, 1, 1]
        
        # 上書き実行
        for j, new_tone in enumerate(new_tones_segment):
            tones[start_index + j] = new_tone
            
        print("[DEBUG] 修正後のトーン:", tones)
    else:
        print("[WARNING] 「こんにちは」の音素列が見つかりませんでした。修正をスキップします。")

    # 4. 修正した phone と tone を使って音声を生成
    # infer メソッドの given_phone, given_tone 引数を使います
    sr, audio_data = model.infer(
        text=text, # テキストも渡しますが、given_phoneがあればそちらが優先されます
        language=Languages.JP,
        
        # ★ここで修正済みのリストを渡す
        given_phone=phones,
        given_tone=tones,
        
        style="Neutral",
        style_weight=0.1, # 癖を抑えるため低めに
        sdp_ratio=0.0,    # 揺らぎをゼロにする（アクセント遵守）
        noise=0.1,
        noise_w=0.1,
        length=1.0
    )

    # 保存
    if audio_data.dtype != np.int16:
        audio_int16 = (audio_data * 32767).astype(np.int16)
    else:
        audio_int16 = audio_data
        
    write(output_path, sr, audio_int16)
    print(f"[SUCCESS] Saved to {output_path}")

# --- メイン処理 ---
if __name__ == "__main__":
    try:
        model = load_model()
        
        # テスト実行
        target_text = "こんにちは。今日は雨予報ですね。"
        output_file = "/workspace/fixed_accent_final.wav"
        
        synthesize_with_manual_accent(model, target_text, output_file)
        
    except Exception as e:
        print(f"[ERROR] 実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()