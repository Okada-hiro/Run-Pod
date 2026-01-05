import sys
import os
import torch
import numpy as np
from scipy.io.wavfile import write
import scipy.signal
from pathlib import Path
import re
import MeCab
import unidic_lite

# mecab_utils のインポートは削除

class TTSWrapper:
    def __init__(self, repo_path, model_assets_dir, model_file, config_file, style_file, device=None):
        self.repo_path = repo_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path) 
            print(f"[INFO] Inserted '{self.repo_path}' to sys.path[0].")

        try:
            from style_bert_vits2.nlp import bert_models
            from style_bert_vits2.nlp.symbols import SYMBOLS 
            from style_bert_vits2.constants import Languages
            from style_bert_vits2.tts_model import TTSModel
            
            self.TTSModel = TTSModel
            self.bert_models = bert_models
            self.Languages = Languages
            self.SYMBOLS = SYMBOLS
        except ImportError as e:
            raise ImportError(f"Style-Bert-VITS2ライブラリが見つかりません: {e}")

        self.pause_symbol = self._determine_pause_symbol()
        print(f"[INFO] Automatically selected pause symbol: '{self.pause_symbol}'")

        self._init_dictionaries()

        print("[INFO] Initializing MeCab (for phonemes)...")
        self.tagger = MeCab.Tagger(f"-d {unidic_lite.DICDIR}")

        print("[INFO] Loading BERT...")
        self.bert_models.load_model(self.Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
        self.bert_models.load_tokenizer(self.Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

        assets_root = Path(model_assets_dir)
        print(f"[INFO] Loading TTS Model from {assets_root / model_file}...")
        self.model = self.TTSModel(
            model_path=assets_root / model_file,
            config_path=assets_root / config_file,
            style_vec_path=assets_root / style_file,
            device=self.device
        )

    def _determine_pause_symbol(self):
        candidates = ['pau', 'sil', 'sp', 'SP', 'sl', '_']
        for sym in candidates:
            if sym in self.SYMBOLS:
                return sym
        if len(self.SYMBOLS) > 0:
            return self.SYMBOLS[-1]
        return 'pau'

    def _init_dictionaries(self):
        # ... (辞書定義は変更なし。長いので省略します) ...
        p = self.pause_symbol
        self.KANA_TO_PHONEME = {
            'ア':['a'], 'イ':['i'], 'ウ':['u'], 'エ':['e'], 'オ':['o'],
            # ... (中略) ...
            '、':[p], '。':[p], '！':[p], '？':[p], '　':[p], ' ':[p]
        }
        self.DIGRAPHS = {
            'キャ':['ky','a'], 'キュ':['ky','u'], 'キョ':['ky','o'],
            # ... (中略) ...
            'チェ':['ch','e'], 'シェ':['sh','e'], 'ジェ':['j','e']
        }

    def _katakana_to_moras(self, text):
        small = set("ャュョァィゥェォ")
        moras = []
        i = 0
        while i < len(text):
            if i + 1 < len(text) and text[i + 1] in small:
                moras.append(text[i:i+2])
                i += 2
            else:
                moras.append(text[i])
                i += 1
        return moras

    def _accent_to_HL(self, pronunciation, accent_type):
        moras = self._katakana_to_moras(pronunciation)
        n = len(moras)
        HL = []
        if n == 0: return moras, []

        if accent_type == 0:
            if n == 1: HL = ["L"]
            else: HL = ["L"] + ["H"] * (n - 1)
        elif accent_type == 1:
            HL = ["H"] + ["L"] * (n - 1)
        else:
            for i in range(n):
                if i == 0: HL.append("L")
                elif i < accent_type: HL.append("H")
                else: HL.append("L")
        return moras, HL

    def _analyze_text_custom(self, text):
        node = self.tagger.parseToNode(text)
        full_phonemes = []
        full_tones = []
        
        full_phonemes.append(self.pause_symbol)
        full_tones.append(0) 
        
        print(f"\n{'='*20} 詳細解析ログ {'='*20}")
        is_katakana = re.compile(r'[\u30A1-\u30F4]+')
        
        while node:
            if node.feature.startswith("BOS/EOS"):
                node = node.next
                continue
            
            features = node.feature.split(",")
            surface = node.surface
            print(f"[DEBUG RAW] 表層:{surface} -> Features:{features}") 

            reading = surface 
            if len(features) > 6 and is_katakana.fullmatch(features[6]):
                reading = features[6]
            elif len(features) > 9 and is_katakana.fullmatch(features[9]):
                reading = features[9]

            acc_type = 0
            try:
                # ★ここが重要：クォート削除処理を残す
                for f in reversed(features):
                    f_clean = f.strip().replace('"', '')
                    if f_clean.isdigit():
                        acc_type = int(f_clean)
                        break
            except:
                acc_type = 0

            if surface in ['、', '。', '！', '？', ' ', '　'] or reading == '*':
                phs = [self.pause_symbol]
                tones = [0]
            else:
                moras, hl_list = self._accent_to_HL(reading, acc_type)
                phs = []
                tones = []
                for m_idx, mora in enumerate(moras):
                    if mora in self.DIGRAPHS: m_phs = self.DIGRAPHS[mora]
                    elif mora in self.KANA_TO_PHONEME: m_phs = self.KANA_TO_PHONEME[mora]
                    else: m_phs = [self.pause_symbol]
                    phs.extend(m_phs)
                    # H=1, L=0
                    val = 1 if hl_list[m_idx] == 'H' else 0
                    tones.extend([val] * len(m_phs))
                print(f"   └-> 採用読み: {reading} / 型: {acc_type} / Tone: {hl_list}")

            full_phonemes.extend(phs)
            full_tones.extend(tones)
            node = node.next

        full_phonemes.append(self.pause_symbol)
        full_tones.append(0)
        full_tones = self._apply_boundary_drop(full_phonemes, full_tones)
        return full_phonemes, full_tones

    def _apply_boundary_drop(self, phones, tones):
        p = self.pause_symbol
        new_tones = tones[:]
        for i in range(len(phones) - 1):
            if phones[i+1] == p and phones[i] != p:
                new_tones[i] = 0
        return new_tones

    def _apply_lowpass_scipy(self, audio_numpy, sr, cutoff):
        if cutoff <= 0 or cutoff >= sr / 2: return audio_numpy
        audio_numpy = np.squeeze(audio_numpy)
        try:
            sos = scipy.signal.butter(5, cutoff / (0.5 * sr), btype='low', analog=False, output='sos')
            return scipy.signal.sosfilt(sos, audio_numpy)
        except:
            return audio_numpy

    def infer(self, text, output_path, style_weight=0.1, pitch=1.0, intonation=1.0, assist_text=None, assist_text_weight=1.0, length=1.0, sdp_ratio=0.0, lpf_cutoff=9000):
        print(f"--- Synthesizing (BERT+Tone Fusion): {text[:20]}... ---")
        
        # 1. 音素とトーンの取得 (これで正確なH/L情報が入手できる)
        phones, tones = self._analyze_text_custom(text)
        use_assist = True if assist_text else False
        
        # 2. 推論実行 (mecab_idsは渡さない)
        with torch.no_grad():
            sr, audio_data = self.model.infer(
                text=text,
                language=self.Languages.JP,
                given_phone=phones,
                given_tone=tones,
                line_split=False, 
                style="Neutral",
                style_weight=style_weight,
                pitch_scale=pitch,
                intonation_scale=intonation,
                assist_text=assist_text,
                assist_text_weight=assist_text_weight,
                use_assist_text=use_assist,
                sdp_ratio=sdp_ratio,
                length=length
            )

        if not isinstance(audio_data, np.ndarray):
            audio_data = audio_data.cpu().numpy()
        
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.5: audio_data = audio_data / 32768.0

        if lpf_cutoff > 0:
            audio_data = self._apply_lowpass_scipy(audio_data, sr, lpf_cutoff)

        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)
            
        write(output_path, sr, audio_int16)
        print(f"[SUCCESS] Saved to {output_path}")