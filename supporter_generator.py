# /workspace/supporter_generator.py (修正版)

import google.generativeai as genai
import os

# Google (Gemini)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"[ERROR] Google genai の設定に失敗: {e}")


# --- システムプロンプト ---
SYSTEM_PROMPT = """
あなたは保険代理店の電話対応を行うAIアドバイザーです。
お客様の音声認識結果に対し、以下の優先順位とルールに従って応答してください。

### 【最重要】入力判定と応答ルール（優先度：高）
最初に入力テキストを分析し、**応答すべきか無視すべきか**を厳格に判断してください。

**1. 無視すべき入力（出力：[SILENCE]）**
以下に該当する場合、または判断に迷う場合は、一切応答せず **[SILENCE]** とだけ出力してください。
- **フィラー・言い淀み**: 「あー」「えーと」「んー」「うん」「まあ」などの短い発声。
- **独り言・感想**: 「まだダメだな」「違うな」「よし」「設定が...」など、AIへの明確な問いかけではない言葉。
- **第三者との会話**: マイクが拾った横の人との会話（「これどう思う？」「高いね」など）。
- **無関係な話題**: 天気、政治、単なる挨拶のみで用件がない場合など。

**2. 応答すべき入力**
- 保険に関する質問、相談、AIへの明確な呼びかけ（「すみません」「保険のことで」など）。

---

### 出力生成ガイドライン（応答する場合のみ）
上記の判定で「応答すべき」となった場合のみ、以下のルールで回答を生成してください。

1. **役割**: 保険のプロとして、親身かつ「です・ます」調で話す。
2. **リード**: 「医療保険ですか？それともガン保険ですか？」のように選択肢を提示し、会話を前に進める。
3. **形式**: Markdown、絵文字、URLは禁止。40〜80文字程度の短文推奨。

### 【重要】判定の具体例（Few-Shot）
User: "んー、えっと..."
AI: [SILENCE]
User: "まだダメだな、聞こえてないのかな"
AI: [SILENCE]
User: "あ、もしもし。保険の相談をしたいんですけど。"
AI: "はい、お電話ありがとうございます。どのような保険をご検討でしょうか。"
User: "（横の人に）ねえ、これ意外と高いよね"
AI: [SILENCE]
"""

# モデル名 (確実に動作するもの)
DEFAULT_MODEL = "gemini-2.5-flash-lite"

def generate_answer_stream(question: str, model=DEFAULT_MODEL, history: list = None):
    """
    回答をストリーミング(ジェネレータ)として返す
    """
    # 履歴がNoneなら空リストで初期化
    if history is None:
        history = []
        
    print(f"[DEBUG] ストリーミング生成開始... (モデル: {model}, 履歴数: {len(history)})")
    
    if not question:
        yield "質問を聞き取れませんでした。"
        return

    try:
        if model.startswith("gemini-"):
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY が設定されていません")
            
            model_instance = genai.GenerativeModel(
                model_name=model,
                system_instruction=SYSTEM_PROMPT,
                generation_config={"temperature": 0.2}
            )
            
            # 会話履歴を引き継ぐ
            chat_session = model_instance.start_chat(history=history)
            
            # ストリーミングリクエスト
            response = chat_session.send_message(question, stream=True)
            
            # ★★★ エラー対策の修正箇所 ★★★
            for chunk in response:
                try:
                    # chunk.text にアクセスしてみて、中身があれば yield する
                    text_part = chunk.text
                    if text_part:
                        yield text_part
                except ValueError:
                    # 「終了合図」などの空データが来た場合、ValueErrorが出るので無視して次へ
                    continue

        else:
            yield f"対応していないモデル名です: {model}"

    except Exception as e:
        print(f"[ERROR] ストリーミング生成エラー: {e}")
        yield "申し訳ありません、エラーが発生しました。"


# --- 単体テスト ---
if __name__ == "__main__":
    print("--- テスト実行 ---")
    if GOOGLE_API_KEY:
        # 空の履歴でテスト
        iterator = generate_answer_stream("こんにちは", history=[])
        for text in iterator:
            print(text, end="", flush=True)
        print("\n")
    else:
        print("APIキーがありません")