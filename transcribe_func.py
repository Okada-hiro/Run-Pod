import subprocess

def whisper_text_only(audio_path, model="medium", language="ja", output_txt=None):
    """音声ファイルをWhisperで文字起こししてテキストだけ返す（ログ出力付き）"""

    cmd = [
        "python3",
        "whisper_streaming/whisper_online.py",
        "--model", model,
        "--language", language,
        audio_path
    ]

    print(f"[DEBUG] 実行コマンド: {' '.join(cmd)}")

    # 標準出力・標準エラーを取得しつつログ出力
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("[DEBUG] stdout:")
    print(result.stdout)
    print("[DEBUG] stderr:")
    print(result.stderr)

    lines = result.stdout.splitlines()

    # 各行のテキスト部分だけ取り出す
    text_only = ""
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4:
            text_only += "".join(parts[3:])  # 4列目以降を結合

    text_only = text_only.strip()

    # ファイル保存
    if output_txt:
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(text_only)
        print(f"[DEBUG] 文字起こし結果を保存: {output_txt}")

    return text_only

if __name__ == "__main__":
    text = whisper_text_only("/content/drive/MyDrive/temp_record.wav",
                             output_txt="/content/result.txt")
    print("[RESULT]")
    print(text)