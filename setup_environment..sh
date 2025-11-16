#!/bin/bash

# エラーが起きたらそこで停止
set -e

# -------------------------------------
# 1. 音声処理ライブラリ
# -------------------------------------
pip install librosa scipy pyworld pyopenjtalk num2words

# -------------------------------------
# 2. Web・AI・音声認識系
# -------------------------------------
pip install fastapi uvicorn[standard] google-generativeai openai huggingface_hub loguru transformers faster-whisper
