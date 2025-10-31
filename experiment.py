# /workspace/watch_and_transcribe.py
import time
import os
from transcribe_func import whisper_text_only
from answer_generator import generate_answer

WATCH_DIR = "incoming_audio"

already_seen = set()

while True:
    files = os.listdir(WATCH_DIR)
    for f in files:
        if f not in already_seen and f.endswith(".wav"):
            audio_path = os.path.join(WATCH_DIR, f)
            print(f"[INFO] 新しい音声を検知: {audio_path}")

            # --- 1. 文字起こし ---
            output_txt_path = os.path.join(WATCH_DIR, f + ".txt")
            print(f"[INFO] 文字起こしを開始... )")
            question_text = whisper_text_only(
                audio_path,
                model="medium",
                language="ja",
                output_txt=output_txt_path
            )
            print(f"[INFO] 文字起こし完了: {output_txt_path}")
            print(f"[INFO] 質問内容: {question_text}")

          

            already_seen.add(f)

    time.sleep(1)  # 1秒ごとにチェック
