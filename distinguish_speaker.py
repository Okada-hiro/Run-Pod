#pip install whisperx pydub
# ffmpegもシステムにインストールされている必要があります
# apt-get install ffmpeg (Linuxの場合)

import whisperx
import os
from pydub import AudioSegment
import gc
import torch

# =================設定項目=================
AUDIO_FILE = "long_audio.mp3"     # 元の長い音声ファイル名
OUTPUT_DIR = "separated_dataset"  # 出力先のフォルダ名
HF_TOKEN = "YOUR_HUGGING_FACE_TOKEN" # ここにHugging FaceのTokenを入れてください
DEVICE = "cuda" # GPUを使用
BATCH_SIZE = 16
# =========================================

def main():
    print("1. モデルをロード中...")
    # Whisperモデルのロード (Large-v2 または v3 推奨)
    compute_type = "float16" 
    model = whisperx.load_model("large-v3", DEVICE, compute_type=compute_type)

    print("2. 音声の文字起こしを実行中...")
    audio = whisperx.load_audio(AUDIO_FILE)
    result = model.transcribe(audio, batch_size=BATCH_SIZE)

    # アライメント（タイミング補正）
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
    result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)

    print("3. 話者分離 (Diarization) を実行中...")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
    diarize_segments = diarize_model(audio)
    
    # 文字起こし結果に話者情報を付与
    result = whisperx.assign_word_speakers(diarize_segments, result)

    print("4. 音声をカットしてフォルダに保存中...")
    # 元音声をpydubで読み込む
    original_audio = AudioSegment.from_file(AUDIO_FILE)

    # 出力ディレクトリ作成
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # セグメントごとに処理
    count_dict = {} # ファイル名連番用

    for segment in result["segments"]:
        # 話者が特定できていない場合はスキップ、または "Unknown" フォルダへ
        speaker = segment.get("speaker", "Unknown_Speaker")
        
        # テキストとタイミング
        text = segment["text"].strip()
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)

        # 話者ごとのフォルダ作成
        speaker_dir = os.path.join(OUTPUT_DIR, speaker)
        if not os.path.exists(speaker_dir):
            os.makedirs(speaker_dir)
            count_dict[speaker] = 0
        
        # ファイル名の決定 (例: SPEAKER_00_001.wav)
        count_dict[speaker] += 1
        file_name_base = f"{speaker}_{count_dict[speaker]:04d}"
        wav_path = os.path.join(speaker_dir, f"{file_name_base}.wav")
        lab_path = os.path.join(speaker_dir, f"{file_name_base}.lab")

        # 音声の切り出しと保存 (44.1kHz, 16bit, monoに変換推奨)
        audio_slice = original_audio[start_ms:end_ms]
        audio_slice = audio_slice.set_frame_rate(44100).set_channels(1).set_sample_width(2)
        audio_slice.export(wav_path, format="wav")

        # テキスト(.lab)の保存
        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        print(f"Saved: {wav_path} ({text})")

    print("完了しました！")

    # メモリ解放
    del model, diarize_model, original_audio
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()