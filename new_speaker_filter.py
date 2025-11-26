# /workspace/speaker_filter.py
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
import os

# --- 1. 音声読み込み関数 ---
def load_audio(path: str, target_sample_rate=16000):
    if not os.path.exists(path):
        raise FileNotFoundError(f"音声ファイルが見つかりません: {path}")

    signal, fs = torchaudio.load(path)

    # ステレオ→モノラル変換
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    # リサンプリング (16kHz必須)
    if fs != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=target_sample_rate)
        signal = resampler(signal)

    return signal

# --- 2. 声紋フィルタークラス ---
class SpeakerGuard:
    def __init__(self):
        print("⏳ [SpeakerGuard] モデルをロード中... (初回は数分かかります)")
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        
        # ★変更点: 1人だけでなく、複数の埋め込みベクトルを保存するリストにする
        self.allowed_embeddings = [] 
        
        # 閾値 (ご提示の通り0.35で設定。厳しい場合は0.25へ)
        self.threshold = 0.35 
        print("✅ [SpeakerGuard] 準備完了")

    def extract_embedding(self, audio_tensor):
        with torch.no_grad():
            embedding = self.classifier.encode_batch(audio_tensor)
        return embedding

    def register_new_speaker(self, audio_path: str) -> bool:
        """
        ★追加機能: 指定された音声を新しい話者としてリストに追加する
        """
        try:
            audio_tensor = load_audio(audio_path)
            new_emb = self.extract_embedding(audio_tensor)
            self.allowed_embeddings.append(new_emb)
            print(f"📝 [SpeakerGuard] 新しい話者を登録しました (現在 {len(self.allowed_embeddings)} 人)")
            return True
        except Exception as e:
            print(f"[SpeakerGuard Error] 登録失敗: {e}")
            return False

    def is_owner(self, audio_path: str) -> bool:
        """
        入力音声が登録済みリストの誰かと一致するか判定
        """
        try:
            audio_tensor = load_audio(audio_path)
        except Exception as e:
            print(f"[SpeakerGuard Error] 読み込み失敗: {e}")
            return False

        current_embedding = self.extract_embedding(audio_tensor)

        # ★変更点: まだ誰も登録されていなければ、最初の1人を自動登録
        if not self.allowed_embeddings:
            print("🔒 [SpeakerGuard] 最初の話者をオーナーとして自動登録しました")
            self.allowed_embeddings.append(current_embedding)
            return True

        # ★変更点: リスト内の全員と比較し、一人でも閾値を超えればOK
        max_score = -1.0
        is_match = False

        for saved_emb in self.allowed_embeddings:
            score = torch.nn.functional.cosine_similarity(
                saved_emb, current_embedding, dim=-1
            )
            score_val = score.item()
            
            if score_val > max_score:
                max_score = score_val
            
            if score_val > self.threshold:
                is_match = True
                break # 一人でも一致すればループ終了

        if is_match:
            print(f"✅ [SpeakerGuard] 本人確認OK (スコア: {max_score:.4f})")
        else:
            print(f"🚫 [SpeakerGuard] ブロック (最大スコア: {max_score:.4f})")
            
        return is_match