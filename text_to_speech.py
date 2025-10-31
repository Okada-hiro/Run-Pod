# /workspace/text_to_speech.py
import subprocess
import os
import shutil

# --- 設定 ---
FISH_SPEECH_DIR = "/workspace/fish-speech"
# create_voice_prompt.py で作成したファイル名
PROMPT_TOKENS = "fake.npy" 
# text2semantic の中間ファイル (fish-speech/temp/ 以下に作られる)
INTERMEDIATE_SEMANTIC_FILE = "temp/codes_0.npy"
# dac のデフォルト出力ファイル (fish-speech/temp/ 以下に作られる)
DEFAULT_OUTPUT_WAV = "temp/codes_0.wav"
# ---

def synthesize_speech(text_to_speak: str, output_wav_path: str, prompt_text: str = "はっきりと丁寧な音声で読み上げてください。"):
    """
    テキストを受け取り、fish-speechで音声合成して指定パスにwavを保存する
    
    Args:
        text_to_speak (str): 読み上げるテキスト
        output_wav_path (str): 保存先のwavファイルの絶対パス
        prompt_text (str): 音声のスタイルを指示するプロンプト
        
    Returns:
        bool: 成功したかどうか
    """
    
    print(f"[DEBUG] 音声合成開始... テキスト: '{text_to_speak[:20]}...'")
    
    original_dir = os.getcwd()
    if not os.path.isdir(FISH_SPEECH_DIR):
        print(f"[ERROR] fish-speechディレクトリが見つかりません: {FISH_SPEECH_DIR}")
        return False

    # 1. text2semantic (テキスト -> セマンティックトークン)
    cmd1 = [
        "python",
        "fish_speech/models/text2semantic/inference.py",
        "--text", text_to_speak,
        "--prompt-text", prompt_text,
        "--prompt-tokens", PROMPT_TOKENS,
        "--compile"
    ]
    
    # 2. dac (セマンティックトークン -> wav)
    cmd2 = [
        "python",
        "fish_speech/models/dac/inference.py",
        "-i", INTERMEDIATE_SEMANTIC_FILE
    ]
    
    try:
        # fish-speechのコマンドはリポジトリのルートで実行する必要がある
        os.chdir(FISH_SPEECH_DIR)
        print(f"[DEBUG] CWDを {FISH_SPEECH_DIR} に変更")
        
        # --- コマンド1実行 ---
        print(f"[DEBUG] 実行 (1/2): text2semantic")
        result1 = subprocess.run(cmd1, capture_output=True, text=True, encoding='utf-8')
        if result1.returncode != 0:
            print(f"[ERROR] text2semantic (cmd1) 失敗:")
            print(result1.stderr)
            return False
        print("[DEBUG] text2semantic (cmd1) 成功。")

        # --- コマンド2実行 ---
        # 中間ファイルの存在チェック
        if not os.path.exists(INTERMEDIATE_SEMANTIC_FILE):
            print(f"[ERROR] 中間ファイルが見つかりません: {INTERMEDIATE_SEMANTIC_FILE}")
            return False
            
        print(f"[DEBUG] 実行 (2/2): dac inference")
        result2 = subprocess.run(cmd2, capture_output=True, text=True, encoding='utf-8')
        if result2.returncode != 0:
            print(f"[ERROR] dac inference (cmd2) 失敗:")
            print(result2.stderr)
            return False
        print("[DEBUG] dac inference (cmd2) 成功。")

        # --- 3. 生成されたファイルを移動 ---
        # デフォルトの出力パス (CWDからの相対パス)
        generated_file_path = DEFAULT_OUTPUT_WAV
        
        if os.path.exists(generated_file_path):
            print(f"[DEBUG] 生成されたファイルを移動: {generated_file_path} -> {output_wav_path}")
            # output_wav_path は絶対パスで受け取る想定
            shutil.move(generated_file_path, output_wav_path)
            return True
        else:
            print(f"[ERROR] 期待された出力ファイルが見つかりません: {generated_file_path}")
            return False

    except Exception as e:
        print(f"[ERROR] 音声合成中に予期せぬエラー: {e}")
        return False
    finally:
        # 必ず元のディレクトリに戻る
        os.chdir(original_dir)
        print(f"[DEBUG] CWDを {original_dir} に戻しました")

if __name__ == "__main__":
    # モジュールの単体テスト
    print("--- 音声合成 単体テスト ---")
    TEST_TEXT = "こんにちは。これはfish-speechのテストです。"
    TEST_OUTPUT = "/workspace/test_output.wav"
    
    if not os.path.exists(os.path.join(FISH_SPEECH_DIR, PROMPT_TOKENS)):
        print(f"[WARN] プロンプトファイル ({PROMPT_TOKENS}) が見つかりません。")
        print("       create_voice_prompt.py を先に実行してください。")
    else:
        success = synthesize_speech(TEST_TEXT, TEST_OUTPUT)
        if success:
            print(f"[SUCCESS] テストファイルが生成されました: {TEST_OUTPUT}")
        else:
            print("[FAIL] テストに失敗しました。")