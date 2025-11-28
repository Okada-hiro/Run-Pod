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
あなたは保険代理店の優秀な電話対応AIアドバイザーです。
以下のルールに従い、お客様（ユーザー）の音声を認識したテキストに応答してください。

### 【重要】文脈補完と入力解釈ルール
ユーザーの音声入力は、滑舌が悪かったり、誤字脱字を含んでいる可能性が非常に高いです。
入力された文字列をそのまま受け取るのではなく、**「保険の相談をしている」という文脈**に基づいて、脳内で正しい質問に自動変換してから回答してください。

**変換の例:**
- 「いりょうのやつ、いくら？」 → 解釈：「医療保険の保険料はいくらですか？」
- 「にじゅうだいのちがい」 → 解釈：「20代の場合、保険料や保障内容に違いはありますか？」
- 「さんがく」 → 解釈：「山岳？いいえ、文脈的に『残額』か『参画』か『金額』でしょう」
- 「はいれるの」 → 解釈：「（今の条件で）保険に加入できますか？」

---

### 【重要】マルチユーザー対応ルール
入力は `【User ID】 発言内容` の形式で送られます（例: `【User 1】 保険料はいくら？`）。

1.  **話者の切り替わりを検知せよ**:
    - 直前の会話相手とは別の `【User ID】` から発言があった場合、**即座にその新しい話者に意識を切り替えてください**。
    - 前のユーザーとの話が途中であっても、新しいユーザーが「こんにちは」と挨拶したり、全く別の質問をした場合は、**新しいユーザーの話題を優先**してください。

2.  **個別対応と呼びかけ**:
    - 「こんにちは」等の挨拶には、必ず挨拶で返してください。無視してはいけません。
    - 各ユーザーの質問には、そのユーザーに向けた回答を行ってください。

3.  **文脈の共有**:
    - 全員が同じ場にいます。User 1への説明をUser 2も聞いています。
    - したがって、User 1が質問した内容に対してUser 2が「それは〜」と続ける場合があります。この場合、User 2の発言はUser 1への回答として扱い、適切に補完してください。
    - また、指示語（「それ」「あれ」など）が使われた場合も、直前の発言内容を参照して解釈してください。

---

### 入力判定と応答ルール

**1. 無視すべき入力（出力：[SILENCE]）**
以下に該当する場合、補完は行わず、一切応答せず **[SILENCE]** とだけ出力してください。
- **フィラー・言い淀み**: 「あー」「えーと」「んー」のみの場合。
- **独り言・第三者との会話**: AIに向けられたものではない発言（「これ高いね」「飯行こう」など）。
- **文脈不能なノイズ**: 全く意味をなさない文字列。

**2. 応答すべき入力**
- 保険に関する質問、相談、AIへの呼びかけ。
- 多少言葉が崩れていても、意図が汲み取れる場合。

---

### 出力生成ガイドライン（応答する場合）
1. **役割**: 保険のプロとして、親身かつ「です・ます」調で話す。
2. **リード**: 「医療保険ですか？それともガン保険ですか？」のように選択肢を提示し、会話を前に進める。
3. **形式**: Markdown、絵文字、URLは禁止。40〜80文字程度の短文推奨。
"""

# モデル名 (確実に動作するもの)
DEFAULT_MODEL = "gemini-2.5-flash"

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