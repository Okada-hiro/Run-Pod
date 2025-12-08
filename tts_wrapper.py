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
        
        # パス設定
        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)
            
        # ライブラリインポート
        try:
            from style_bert_vits2.nlp import bert_models
            from style_bert_vits2.constants import Languages
            from style_bert_vits2.tts_model import TTSModel
            self.TTSModel = TTSModel
            self.bert_models = bert_models
            self.Languages = Languages
        except ImportError as e:
            raise ImportError(f"Style-Bert-VITS2ライブラリが見つかりません: {e}")

        # モデルロード
        self._load_bert()
        self.model = self._load_tts_model(model_assets_dir, model_file, config_file, style_file)
        
        # アクセント辞書の初期化
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
        """
        JSONファイルからアクセント辞書を読み込み、
        検索用に「音素列」を事前計算しておく
        """
        if not os.path.exists(json_path):
            print(f"[WARNING] Accent JSON not found: {json_path}")
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        count = 0
        for word, tones in data.items():
            # 単語から音素列(target_phones)を生成
            # ※pyopenjtalkを使って「その単語がどう分解されるか」を取得
            phones = pyopenjtalk.g2p(word, kana=False).split(" ")
            
            # pau(無音)が含まれる場合があるため除去
            phones = [p for p in phones if p not in ('pau', 'sil')]
            
            # 登録
            self.accent_rules[word] = {
                "phones": phones,
                "tones": tones
            }
            count += 1
            
        print(f"[INFO] Loaded {count} accent rules from {json_path}")

    def _g2p_and_patch(self, text):
        """
        テキスト全体の音素を取得し、辞書にある単語のアクセントを上書きする
        """
        # 1. 全体の音素取得
        labels = pyopenjtalk.extract_fullcontext(text)
        phones = []
        tones = [] # 初期値は全て0

        for label in labels:
            parts = label.split('/')
            p3 = label.split('-')[1].split('+')[0]
            if p3 == 'sil': p3 = 'pau'
            phones.append(p3)
            tones.append(0) # デフォルト0

        # 2. 辞書ルールに基づいてパッチ当て
        for word, rule in self.accent_rules.items():
            target_phones = rule['phones']
            target_tones = rule['tones']
            
            # 音素数が合わない場合はスキップ（安全装置）
            if len(target_phones) != len(target_tones):
                print(f"[WARNING] Skip '{word}': 音素数({len(target_phones)})とトーン数({len(target_tones)})が不一致")
                continue

            # 全体の中からターゲット音素列を探す（全箇所置換）
            seq_len = len(target_phones)
            for i in range(len(phones) - seq_len + 1):
                # 音素列が一致するか？
                if phones[i : i + seq_len] == target_phones:
                    # 一致したらトーンを上書き
                    print(f"[PATCH] Applying accent fix for '{word}' at index {i}")
                    for j, t_val in enumerate(target_tones):
                        tones[i + j] = t_val

        return phones, tones

    assist_directive = "プロのニュースキャスターです。落ち着いたトーンで、正確に、明瞭に原稿を読み上げます。"

    def infer(self, text, output_path, style_weight=0.1, pitch=1.0, assist_text_weight=1.5, intonation=1.0, assist_text=None):
        """
        音声合成実行
        """
        print(f"--- Synthesizing: {text[:20]}... ---")
        
        # アクセント修正処理
        phones, tones = self._g2p_and_patch(text)
        
        # 推論
        sr, audio_data = self.model.infer(
            text=text,
            language=self.Languages.JP,
            given_phone=phones,
            given_tone=tones, # 修正済みトーン
            style="Neutral",
            style_weight=style_weight,

            # ★ここが魔法のスパイスです
            assist_text=assist_text,     # 「嬉しい」「悲しい」などの文章
            assist_text_weight=assist_text_weight,      # どれくらいその口調に寄せるか
            use_assist_text=True if assist_text else False,

            pitch_scale=pitch,
            intonation_scale=intonation,

            sdp_ratio=0.0,    # 揺らぎなし（必須）
            noise=0.1,
            noise_w=0.1,
            length=1.0
        )

        # 保存
        if audio_data.dtype != np.int16:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data
            
        write(output_path, sr, audio_int16)
        print(f"[SUCCESS] Saved to {output_path}")