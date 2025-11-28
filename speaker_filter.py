# /workspace/new_speaker_filter.py (話者ID識別対応版)
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
import os
import logging

logger = logging.getLogger(__name__)

def load_audio(path: str, target_sample_rate=16000):
    if not os.path.exists(path):
        raise FileNotFoundError(f"音声ファイルが見つかりません: {path}")

    signal, fs = torchaudio.load(path)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if fs != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=target_sample_rate)
        signal = resampler(signal)
    return signal

class SpeakerGuard:
    def __init__(self):
        print("⏳ [SpeakerGuard] モデルをロード中... (SpeechBrain)")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )
        # 構造変更: リストではなく、辞書のリスト [{'id': 'User 0', 'emb': tensor}, ...]
        self.known_speakers = [] 
        self.threshold = 0.35 
        print(f"✅ [SpeakerGuard] 準備完了 (Device: {self.device})")

    def extract_embedding(self, audio_tensor):
        audio_tensor = audio_tensor.to(self.device)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        wav_lens = torch.ones(audio_tensor.shape[0]).to(self.device)
        with torch.no_grad():
            embedding = self.classifier.encode_batch(audio_tensor, wav_lens)
        return embedding

    def identify_speaker(self, audio_tensor) -> tuple[bool, str]:
        """
        Tensorを受け取り、(登録済みか, 話者ID) を返す
        """
        try:
            current_embedding = self.extract_embedding(audio_tensor)
            
            # 初回登録 (オーナー)
            if not self.known_speakers:
                print("🔒 [SpeakerGuard] 最初の話者を 'User 0' (オーナー) として登録")
                self.known_speakers.append({'id': 'User 0', 'emb': current_embedding})
                return True, "User 0"

            max_score = -1.0
            best_match_id = "Unknown"
            is_match = False

            # 全登録者と比較して、最も似ている人を探す
            for speaker in self.known_speakers:
                score = torch.nn.functional.cosine_similarity(
                    speaker['emb'], current_embedding, dim=-1
                )
                score_val = score.item()
                
                if score_val > max_score:
                    max_score = score_val
                    if score_val > self.threshold:
                        is_match = True
                        best_match_id = speaker['id']

            if is_match:
                logger.info(f"✅ [SpeakerGuard] 認証成功: {best_match_id} (スコア: {max_score:.4f})")
                return True, best_match_id
            else:
                logger.info(f"🚫 [SpeakerGuard] 未知の話者 (最大スコア: {max_score:.4f})")
                return False, "Unknown"
                
        except Exception as e:
            print(f"[SpeakerGuard Error] 識別失敗: {e}")
            return False, "Error"

    def register_new_speaker(self, audio_path: str) -> str:
        """
        新規登録し、割り当てたIDを返す
        """
        try:
            audio_tensor = load_audio(audio_path)
            new_emb = self.extract_embedding(audio_tensor)
            
            # 新しいIDを生成 (User 1, User 2...)
            new_id = f"User {len(self.known_speakers)}"
            
            self.known_speakers.append({'id': new_id, 'emb': new_emb})
            print(f"📝 [SpeakerGuard] 新規登録完了: {new_id}")
            return new_id
        except Exception as e:
            print(f"[SpeakerGuard Error] 登録失敗: {e}")
            return None

    # 後方互換性のため残す（今回は使わない）
    def verify_tensor(self, audio_tensor):
        is_ok, _ = self.identify_speaker(audio_tensor)
        return is_ok