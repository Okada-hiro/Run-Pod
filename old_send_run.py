# send_and_run.py
import sounddevice as sd
from scipy.io.wavfile import write
import subprocess
import os
from move_files import move_file

# ---------------------------
# ① 音声を録音する関数
# ---------------------------
def record_audio(filename="record.wav", duration=20, fs=44100):
    print("録音中...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, recording)
    print(f"録音完了: {filename}")
    return filename

#

# ---------------------------
# メイン処理
# ---------------------------
if __name__ == "__main__":
    # 録音ファイル作成
    wav_file = record_audio(duration=5)
    move_file(wav_file)
    
