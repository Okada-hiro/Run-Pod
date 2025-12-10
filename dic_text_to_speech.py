import os
from tts_wrapper import TTSWrapper

# --- 設定 ---
WORKSPACE_DIR = os.getcwd()
REPO_PATH = os.path.join(WORKSPACE_DIR, "Style_Bert_VITS2")
ASSETS_DIR = os.path.join(REPO_PATH, "model_assets")

# モデルファイル名
MODEL_FILE = "Ref_voice_e300_s2100.safetensors"
CONFIG_FILE = "config.json"
STYLE_FILE = "style_vectors.npy"

# アクセント辞書
ACCENT_JSON = "accents.json"

def main():
    # 1. ラッパーの初期化 (モデルロード)
    tts = TTSWrapper(
        repo_path=REPO_PATH,
        model_assets_dir=ASSETS_DIR,
        model_file=MODEL_FILE,
        config_file=CONFIG_FILE,
        style_file=STYLE_FILE
    )

    # 2. アクセント辞書の読み込み
    tts.load_accent_dict(ACCENT_JSON)

    # 3. 音声合成
    text = """みなさん、おはようございます。まずは、今日の天気と生活に役立つ話題をお伝えします。
本日は、全国的に雲の多い空模様となり、午後にかけて雨の降る地域が増える予報です。午前中は一時的に日差しが届くところもありますが、次第に空が暗くなり、弱い雨が降り始める見込みです。帰宅時間帯には雨脚が少し強まる可能性もありますので、お出かけの際は傘をお持ちください。気温は昨日と同じくらいですが、風の影響で肌寒く感じる場面もあり、羽織りものがあると安心です。
さて、そんな雨の日ですが、カフェでゆっくり過ごす方が増えています。雨音が外の騒がしさを和らげ、落ち着いた時間をつくってくれるため、読書やパソコン作業にも人気です。温かい紅茶やコーヒーが、雨の日にはより一層おいしく感じられるという声も多く聞かれます。
駅前のカフェなど、多くの店舗は本日も通常通り営業しています。午前中は比較的空いているところが多いため、静かに過ごしたい方にはおすすめです。一方、昼前から午後にかけては混みやすくなるため、時間に余裕のある方は少し早めに足を運ぶとゆったり過ごせるでしょう。"""
    output = "final_output.wav"

    assist_directive = "アナウンサーです。はきはきと、明瞭に喋ります。全く雑音のない、クリアな音声で喋ります。。"

    # パラメータを少し緩めて、「辞書優先」にします
    tts.infer(text, output, 
        pitch=1.2, 
        intonation=1.3,        # 1.4は強すぎます。1.0で十分です。
        style_weight=0.1,      # 0.6は強すぎます。0.1に下げてください。
        
        # ★以下の2行を追加してください
        assist_text=assist_directive, 
        assist_text_weight=0.2, # 控えめに効かせる

        # ★★★ ここを変更 ★★★
        length=0.9,     # 1.0だと遅いので、0.9 や 0.8 にして速くする
        sdp_ratio=0.0,  # ランダムな引き伸ばしを無効化（「謎の間」を消す）

        lpf_cutoff=9000
    )

if __name__ == "__main__":
    main()