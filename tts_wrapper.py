import sys
import os
import json
import torch
import numpy as np
import pyopenjtalk
from scipy.io.wavfile import write
import scipy.signal  # ★これを追加（標準的な信号処理ライブラリ）
from pathlib import Path

class TTSWrapper:
    def __init__(self, repo_path, model_assets_dir, model_file, config_file, style_file, device=None):
        self.repo_path = repo_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path) 
            print(f"[INFO] Inserted '{self.repo_path}' to sys.path[0] to force load local modules.")

        try:
            from style_bert_vits2.nlp import bert_models
            from style_bert_vits2.constants import Languages
            from style_bert_vits2.tts_model import TTSModel
            
            self.TTSModel = TTSModel
            self.bert_models = bert_models
            self.Languages = Languages
           
        except ImportError as e:
            raise ImportError(f"Style-Bert-VITS2ライブラリが見つかりません: {e}")

        self._load_bert()
        self.model = self._load_tts_model(model_assets_dir, model_file, config_file, style_file)
        
        self.accent_rules = {}

    def _load_bert(self):
        print("[INFO] Loading BERT...")
        self.bert_models.load_model(self.Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
        self.bert_models.load_tokenizer(self.Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

    def _load_tts_model(self, assets_dir, model_file, config_file, style_file):
        assets_root = Path(assets_dir)
        print(f"[INFO] Loading TTS Model from {assets_root / model_file}...")
        return self.TTSModel(
            model_path=assets_root / model_file,
            config_path=assets_root / config_file,
            style_vec_path=assets_root / style_file,
            device=self.device
        )

    def load_accent_dict(self, json_path):
        if not os.path.exists(json_path):
            print(f"[WARNING] Accent JSON not found: {json_path}")
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        count = 0
        for word, tones in data.items():
            phones = pyopenjtalk.g2p(word, kana=False).split(" ")
            phones = [p for p in phones if p not in ('pau', 'sil')]
            self.accent_rules[word] = {"phones": phones, "tones": tones}
            count += 1
            
        print(f"[INFO] Loaded {count} accent rules from {json_path}")

    def _parse_openjtalk_accent(self, labels):
        phones = []
        tones = []
        
        for label in labels:
            parts = label.split('/')
            p3 = label.split('-')[1].split('+')[0]
            if p3 == 'sil': p3 = 'pau'
            phones.append(p3)
            
            if p3 == 'pau':
                tones.append(0)
                continue

            try:
                a_part = parts[1]
                if 'A:' not in a_part:
                    tones.append(0)
                    continue
                    
                nums = a_part.split(':')[1].split('+')
                a1 = int(nums[0])
                a2 = int(nums[1])
                
                is_high = 0
                if a1 == 0:
                    if a2 == 1: is_high = 0
                    else:       is_high = 1
                else:
                    if a2 <= a1:
                        if a2 == 1 and a1 > 1: is_high = 0
                        else: is_high = 1
                    else:
                        is_high = 0
                
                tones.append(is_high)
            except:
                tones.append(0)

        return phones, tones

    def _g2p_and_patch(self, text):
        labels = pyopenjtalk.extract_fullcontext(text)
        phones, tones = self._parse_openjtalk_accent(labels)

        for word, rule in self.accent_rules.items():
            target_phones = rule['phones']
            target_tones = rule['tones']
            
            if len(target_phones) != len(target_tones):
                print(f"[WARNING] Skip '{word}': 音素数不一致")
                continue

            seq_len = len(target_phones)
            for i in range(len(phones) - seq_len + 1):
                if phones[i : i + seq_len] == target_phones:
                    print(f"[PATCH] Applying accent fix for '{word}' at index {i}")
                    for j, t_val in enumerate(target_tones):
                        tones[i + j] = t_val

        return phones, tones

    # ★ 変更点: PyTorchの手計算をやめて、Scipyを使います (CPU処理で安全確実)
    def _apply_lowpass_scipy(self, audio_numpy, sr, cutoff):
        if cutoff <= 0 or cutoff >= sr / 2:
            return audio_numpy

        try:
            # ナイキスト周波数
            nyquist = 0.5 * sr
            normal_cutoff = cutoff / nyquist
            
            # バターワースフィルタの設計 (5次)
            # sos (Second-Order Sections) 形式を使うと数値的に安定します
            sos = scipy.signal.butter(5, normal_cutoff, btype='low', analog=False, output='sos')
            
            # フィルタ適用
            filtered = scipy.signal.sosfilt(sos, audio_numpy)
            return filtered
        except Exception as e:
            print(f"[ERROR] Scipy filter failed: {e}")
            return audio_numpy

    def infer(self, text, output_path, style_weight=0.1, pitch=1.0, assist_text_weight=0.0, intonation=1.3, assist_text=None ,length=1.0, sdp_ratio=0.2, lpf_cutoff=9000):
        print(f"--- Synthesizing: {text[:20]}... ---")
        
        phones, tones = self._g2p_and_patch(text)
        use_assist = True if assist_text else False
        
        # 1. 音声生成 (Numpy配列)
        sr, audio_data = self.model.infer(
            text=text,
            language=self.Languages.JP,
            given_phone=phones,
            given_tone=tones,
            style="Neutral",
            style_weight=style_weight,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            use_assist_text=use_assist,
            pitch_scale=pitch,
            intonation_scale=intonation,
            sdp_ratio=sdp_ratio,
            noise=0.1,
            noise_w=0.1,
            length=length
        )

        # 2. Scipyでフィルタ適用 (GPU->CPU変換などが不要で、Numpyのまま処理できるのでバグりません)
        if lpf_cutoff > 0:
            original_max = np.max(np.abs(audio_data))
            
            audio_data = self._apply_lowpass_scipy(audio_data, sr, lpf_cutoff)
            
            print(f"[INFO] Applied Scipy Low-pass filter at {lpf_cutoff}Hz")

        # 3. 音量正規化と保存
        # フィルタをかけると音量が下がることがあるので、少し持ち上げる(オプション)
        if audio_data.dtype != np.int16:
            # クリップ処理 (バリバリ音防止)
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data
            
        write(output_path, sr, audio_int16)
        print(f"[SUCCESS] Saved to {output_path}")