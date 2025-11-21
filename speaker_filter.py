# /workspace/speaker_filter.py
import torch
import torchaudio
from speechbrain.inference.speakers import EncoderClassifier
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
        # ECAPA-TDNN という非常に精度の高いモデルを使用
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"} # GPUがあれば使う
        )
        self.owner_embedding = None
        # 類似度の閾値 (0.0〜1.0)。0.25〜0.35あたりが一般的
        # 値を小さくすると厳しくなり、大きくすると緩くなる
        self.threshold = 0.35 
        print("✅ [SpeakerGuard] 準備完了")

    def extract_embedding(self, audio_tensor):
        """
        音声データから「声の特徴ベクトル」を抽出する
        """
        with torch.no_grad():
            # 推論実行
            embedding = self.classifier.encode_batch(audio_tensor)
        return embedding

    def is_owner(self, audio_path: str) -> bool:
        """
        入力された音声ファイルがオーナーか判定する
        """
        # 音声をロード
        try:
            audio_tensor = load_audio(audio_path)
        except Exception as e:
            print(f"[SpeakerGuard Error] 読み込み失敗: {e}")
            return False

        # 現在の声の特徴を取得
        current_embedding = self.extract_embedding(audio_tensor)

        # まだオーナーがいなければ、この人をオーナーにする
        if self.owner_embedding is None:
            print("🔒 [SpeakerGuard] 最初の話者をオーナーとして登録しました")
            self.owner_embedding = current_embedding
            return True

        # 類似度判定 (コサイン類似度)
        # score は -1.0(別人) 〜 1.0(本人) の範囲
        score = torch.nn.functional.cosine_similarity(
            self.owner_embedding, current_embedding, dim=-1
        )
        
        # スコアを取り出す
        score_val = score.item()
        
        # ここでは「閾値よりスコアが高ければ本人」と判定
        is_match = score_val > self.threshold
        
        if is_match:
            print(f"✅ [SpeakerGuard] 本人確認OK (スコア: {score_val:.4f})")
        else:
            print(f"🚫 [SpeakerGuard] 他人の声をブロック (スコア: {score_val:.4f})")
            
        return is_match