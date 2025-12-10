import sys
import os
import json
import torch
import numpy as np
import pyopenjtalk
from scipy.io.wavfile import write
from pathlib import Path

class TTSWrapper:
    def __init__(self, repo_path, model_assets_dir, model_file, config_file, style_file, device=None):
        self.repo_path = repo_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 【修正】 append ではなく insert(0, ...) を使う
        # これにより、インストール済みのライブラリよりも先に、手元のフォルダを探しに行きます
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

    # ★ここが最重要：OpenJTalkの標準アクセントを計算して復元する機能
    def _parse_openjtalk_accent(self, labels):
        phones = []
        tones = []
        
        for label in labels:
            parts = label.split('/')
            p3 = label.split('-')[1].split('+')[0]
            if p3 == 'sil': p3 = 'pau'
            phones.append(p3)
            
            # pau（無音）は0でOK
            if p3 == 'pau':
                tones.append(0)
                continue

            # ラベルからアクセント情報を解析して、正しい0/1を計算する
            try:
                a_part = parts[1] # A:xx+xx+xx
                if 'A:' not in a_part:
                    tones.append(0)
                    continue
                    
                nums = a_part.split(':')[1].split('+')
                a1 = int(nums[0]) # アクセント核
                a2 = int(nums[1]) # 現在のモーラ番号
                
                is_high = 0
                if a1 == 0: # 平板型
                    if a2 == 1: is_high = 0
                    else:       is_high = 1
                else: # 起伏型
                    if a2 <= a1:
                        if a2 == 1 and a1 > 1: is_high = 0
                        else: is_high = 1
                    else:
                        is_high = 0
                
                tones.append(is_high)
            except:
                tones.append(0) # 解析失敗時のみ0

        return phones, tones

    def _g2p_and_patch(self, text):
        # 1. まず標準的なアクセントを計算（これで0埋めを回避）
        labels = pyopenjtalk.extract_fullcontext(text)
        phones, tones = self._parse_openjtalk_accent(labels)

        # 2. その上で辞書を適用
        for word, rule in self.accent_rules.items():
            target_phones = rule['phones']
            target_tones = rule['tones']
            
            if len(target_phones) != len(target_tones):
                print(f"[WARNING] Skip '{word}': 音素数({len(target_phones)})とトーン数({len(target_tones)})が不一致")
                continue

            seq_len = len(target_phones)
            for i in range(len(phones) - seq_len + 1):
                if phones[i : i + seq_len] == target_phones:
                    print(f"[PATCH] Applying accent fix for '{word}' at index {i}")
                    for j, t_val in enumerate(target_tones):
                        tones[i + j] = t_val

        return phones, tones

    def infer(self, text, output_path, style_weight=0.1, pitch=1.0, assist_text_weight=0.0, intonation=1.3, assist_text=None ,length=1.0, sdp_ratio=0.2):
        print(f"--- Synthesizing: {text[:20]}... ---")
        
        # 修正版のアクセント生成
        phones, tones = self._g2p_and_patch(text)
        
        use_assist = True if assist_text else False
        
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
            
            # ★受け取った値をここで渡すように変更
            sdp_ratio=sdp_ratio,
            noise=0.6,
            noise_w=0.8,
            length=length   # ← ここで渡す！


        )

        if audio_data.dtype != np.int16:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data
            
        write(output_path, sr, audio_int16)
        print(f"[SUCCESS] Saved to {output_path}")