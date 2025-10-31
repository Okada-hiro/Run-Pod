#!/bin/bash
# /workspace/setup_fish_speech.sh

# 1. 必要なパッケージのインストール
echo "[SETUP] 必要なaptパッケージをインストールします..."
apt update
apt install -y portaudio19-dev libsox-dev ffmpeg 

# 2. 必要なpipパッケージのインストール
echo "[SETUP] 必要なpipパッケージをインストールします..."
pip install huggingface_hub[cli]
# 3. Hugging Face ログイン (対話型)
echo "[SETUP] Hugging Faceにログインしてください..."
echo "!!! HF Token (read) を入力してください !!!"
huggingface-cli login
# (ここで手動でトークンをペーストする必要があります)

# 4. リポジトリのクローン
if [ -d "/workspace/fish-speech" ]; then
    echo "[SETUP] fish-speech ディレクトリは既に存在します。スキップします。"
else
    echo "[SETUP] fish-speech リポジトリをクローンします..."
    cd /workspace
    git clone https://github.com/fishaudio/fish-speech.git
fi

# 5. fish-speech のセットアップとモデルのダウンロード
cd /workspace/fish-speech
echo "[SETUP] fish-speech の依存関係をインストールします..."
# (cu129の部分はRunPodのCUDAバージョンに合わせて変更が必要な場合があります)
pip install -e .[cu129] 

echo "[SETUP] 事前学習済みモデルをダウンロードします..."
hf download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini

echo "[SETUP] セットアップ完了。"