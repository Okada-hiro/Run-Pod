# /workspace/supporter_generator.py (マルチAPI対応版)

import google.generativeai as genai
import os
import time

# Google (Gemini)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"[ERROR] Google genai の設定に失敗: {e}")


# --- システムプロンプトの定義 (無視機能付き) ---
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

---

### 【重要】判定の具体例（Few-Shot）
AIは以下の例に従って判断を行ってください。

User: "んー、えっと..."
AI: [SILENCE]

User: "まだダメだな、聞こえてないのかな"
AI: [SILENCE]

User: "あ、もしもし。保険の相談をしたいんですけど。"
AI: "はい、お電話ありがとうございます。どのような保険をご検討でしょうか。"

User: "（横の人に）ねえ、これ意外と高いよね"
AI: [SILENCE]

User: "入院したときの保証が心配で。"
AI: "入院時の保証ですね、ご安心ください。日帰りの入院からカバーするプランもございますが、詳細をご説明しましょうか？"

User: "あー、はいはい。"
AI: [SILENCE]
"""

# --- 2. モデル名に応じて処理を分岐する ---
DEFAULT_MODEL = "gemini-2.5-flash-lite"
def generate_answer(question: str, model=DEFAULT_MODEL, history: list = None) -> str:
    """
    受け取った質問テキストに対し、指定されたモデルで回答を生成する。
    """
    print(f"[DEBUG] 回答生成中... (モデル: {model}) 質問: '{question}'")
    if history is None:
        history = []

    print(f"[DEBUG] ストリーミング生成開始... (履歴数: {len(history)})")
    if not question:
        return "質問を聞き取れませんでした。"

    answer = ""

    try:
        # --- Google (gemini-...) ---
        if model.startswith("gemini-"):
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY が設定されていません")
            
            # Geminiモデルを初期化
            model_instance = genai.GenerativeModel(
                model_name=model,
                system_instruction=SYSTEM_PROMPT
            )
            # ★ ここを変更: start_chat で履歴付きセッションを開始
            chat_session = model_instance.start_chat(history=history)
            
            # 履歴を踏まえてメッセージを送信
            response = chat_session.send_message(question, stream=True)
            answer = response.text.strip()

        else:
            raise ValueError(f"対応していないモデル名です: {model}")

        print(f"[DEBUG] {model} 回答生成完了: '{answer[:30]}...'")

    except Exception as e:
        print(f"[ERROR] {model} での回答生成に失敗しました: {e}")
        # エラー時も沈黙させるか、エラーを返すかは運用次第ですが、ここではエラーを伝えます
        answer = f"申し訳ありません、回答を生成中にエラーが発生しました。"

    return answer

# ★ モデル名を修正
def generate_answer_stream(question: str, model="gemini-2.5-flash-lite", history: list = None):
    """
    回答をストリーミング(ジェネレータ)として返す
    """
    print(f"[DEBUG] 回答ストリーミング生成開始... (モデル: {model})")
    
    if not question:
        yield "質問を聞き取れませんでした。"
        return

    try:
        if model.startswith("gemini-"):
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY が設定されていません")
            
            model_instance = genai.GenerativeModel(
                model_name=model,
                system_instruction=SYSTEM_PROMPT
            )
            
            # ★ ここを変更: start_chat で履歴付きセッションを開始
            chat_session = model_instance.start_chat(history=history)
            
            # 履歴を踏まえてメッセージを送信
            response = chat_session.send_message(question, stream=True)
            answer = response.text.strip()
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        else:
            yield f"対応していないモデル名です: {model}"

    except Exception as e:
        print(f"[ERROR] ストリーミング生成エラー: {e}")
        yield "申し訳ありません、エラーが発生しました。"

# --- 3. 単体テスト ---
if __name__ == "__main__":
    print("--- マルチAPI回答生成 単体テスト ---")
    test_q = "こんにちは、家賃の相場を教えてください。"

    if GOOGLE_API_KEY:
        # ★ モデル名を修正してテスト
        print("\n[Test] gemini-2.5-flash-lite")
        ans_gemini = generate_answer(test_q, model="gemini-2.5-flash-lite")
        print(f"Gemini 回答: {ans_gemini}")
        
        print("\n[Test] Ignore Case (独り言)")
        ans_ignore = generate_answer("あー、えっと、なんだっけ", model="gemini-2.5-flash-lite")
        print(f"Gemini 無視判定: {ans_ignore}")
    else:
        print("\n[Skipped] GOOGLE_API_KEY 未設定")