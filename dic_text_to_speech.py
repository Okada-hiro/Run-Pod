import sys
import os
import torch
import numpy as np
from scipy.io.wavfile import write
from pathlib import Path

# --- 1. パス設定 ---
# 実行ディレクトリ (/workspace/Run-Pod/)
WORKSPACE_DIR = os.getcwd()
# リポジトリのパス
REPO_PATH = os.path.join(WORKSPACE_DIR, "Style_Bert_VITS2")

# パスが通っていなければ追加
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)
    print(f"[INFO] Added to sys.path: {REPO_PATH}")

try:
    # 必要なライブラリのインポート
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.tts_model import TTSModel
    
    # ★重要: get_textの代わりに、この g2p 機能を使います
    from style_bert_vits2.nlp.japanese.phoneme import g2p
    
except ImportError as e:
    print(f"[ERROR] ライブラリのインポートに失敗しました: {e}")
    sys.exit(1)

# --- 設定 (環境に合わせて変更してください) ---
FT_MODEL_FILE = "こんにちは_Ref_voice_e3_s3.safetensors" 
FT_CONFIG_FILE = "config.json"
FT_STYLE_FILE = "style_vectors.npy"
GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- モデルロード関数 ---
def load_model():
    print("[INFO] Loading BERT...")
    bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

    assets_root = Path(REPO_PATH) / "model_assets"
    
    # モデルファイルのパス
    # ※ユーザー様の環境に合わせてパス構造を調整してください（model_assets直下か、サブフォルダか）
    model_path = assets_root / FT_MODEL_FILE
    config_path = assets_root / FT_CONFIG_FILE
    style_vec_path = assets_root / FT_STYLE_FILE
    
    # ファイル存在確認
    if not model_path.exists():
        # もし直下にない場合、Ref_voiceフォルダの中を探すなどのロジックを入れるか、
        # 正しいパスを直接指定してください。
        # 今回は一旦エラーを出さず、ログだけ出して続行させます（パスが合っている前提）
        pass

    print(f"[INFO] Loading TTS Model from {model_path}...")
    model = TTSModel(
        model_path=model_path,
        config_path=config_path,
        style_vec_path=style_vec_path,
        device=GLOBAL_DEVICE
    )
    return model

# --- ★アクセント強制修正ロジック ---
def synthesize_with_manual_accent(model, text: str, output_path: str):
    """
    g2pで音素とトーンを取得し、「こんにちは」の部分だけ強制的に平板アクセントに書き換える
    """
    print(f"--- テキスト: {text} ---")

    # 1. テキストを音素(phones)とアクセント(tones)に変換
    # ここで get_text は使いません。g2pを使います。
    phones, tones, _ = g2p(text)

    print(f"[DEBUG] 元の音素: {phones}")
    print(f"[DEBUG] 元のトーン: {tones}")

    # 2. 「こんにちは」の音素パターンを探す
    # 通常の音素列: ['k', 'o', 'N', 'n', 'i', 'ch', 'i', 'w', 'a']
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
        # 書き換えるトーン列: 0, 0, 1, 1, 1, 1, 1, 1, 1
        new_tones_segment = [0, 0, 1, 1, 1, 1, 1, 1, 1]
        
        # 上書き実行
        for j, new_tone in enumerate(new_tones_segment):
            tones[start_index + j] = new_tone
            
        print(f"[DEBUG] 修正後のトーン: {tones}")
    else:
        print("[WARNING] 「こんにちは」の音素列が見つかりませんでした。修正をスキップします。")

    # 4. 音声生成 (inferメソッドを使用)
    # model.infer は given_phone, given_tone を受け取れます（tts_model.py で確認済み）
    print("[INFO] 音声を生成しています...")
    sr, audio_data = model.infer(
        text=text, 
        language=Languages.JP,
        
        # ★ここで修正済みのリストを渡します★
        given_phone=phones,
        given_tone=tones,
        
        style="Neutral",
        style_weight=0.1, 
        sdp_ratio=0.0,    # 揺らぎゼロ（アクセント遵守）
        noise=0.1,
        noise_w=0.1,
        length=1.0
    )

    # 保存処理
    if audio_data.dtype != np.int16:
        audio_int16 = (audio_data * 32767).astype(np.int16)
    else:
        audio_int16 = audio_data
        
    if os.path.exists(output_path):
        os.remove(output_path)
        
    write(output_path, sr, audio_int16)
    print(f"[SUCCESS] 音声を保存しました: {output_path}")

# --- メイン処理 ---
if __name__ == "__main__":
    try:
        # モデルをロード
        model = load_model()
        
        # テスト実行
        # ここを変えれば好きなテキストを試せます
        TARGET_TEXT = "こんにちは。今日は雨予報ですね。"
        OUTPUT_FILE = "/workspace/Run-Pod/fixed_accent_final.wav"
        
        synthesize_with_manual_accent(model, TARGET_TEXT, OUTPUT_FILE)
        
    except Exception as e:
        print(f"[ERROR] 実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()