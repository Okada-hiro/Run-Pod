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
音声認識されたお客様の言葉に対し、以下のガイドラインに従って応答してください。

### 1. 役割と振る舞い（保険のプロ）
- **丁寧で親身な対応**: お客様の不安を取り除くよう、優しく丁寧な「です・ます」調で話してください。
- **会話をリードする**: 
  - お客様が話しやすいように、「医療保険をお探しですか？それとも貯蓄型の保険にご興味がありますか？」のように**具体的な選択肢を提示**したり、**質問を投げかけたり**してください。
  - 単答で終わらせず、会話が続くように促してください。
- **わかりやすい説明**: 専門用語はなるべく使わず、噛み砕いて説明してください。音声読み上げのため、一文が長くなりすぎないよう適度に句読点を入れてください。

### 2. 出力形式の制約
- 音声合成システム（TTS）で使用するため、**Markdown（太字、箇条書き）、絵文字、URLは絶対に使用しないでください。**
- 記号は読み上げ可能な句読点（、。）や疑問符（？）のみを使用してください。

### 3. 無視機能（ガードレール）
入力が以下に該当する場合は、**[SILENCE]** とだけ出力してください（それ以外の文字は含めない）。
- 第三者との会話（「お母さん、これどう思う？」など）
- 独り言、うめき声、環境音、ノイズ
- 保険やライフプランと全く関係のない話題（天気、政治、スポーツの結果など）
"""

# --- 2. モデル名に応じて処理を分岐する ---

def generate_answer(question: str, model="gemini-2.5-flash-lite") -> str:
    """
    受け取った質問テキストに対し、指定されたモデルで回答を生成する。
    """
    print(f"[DEBUG] 回答生成中... (モデル: {model}) 質問: '{question}'")
    
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
            response = model_instance.generate_content(question)
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
def generate_answer_stream(question: str, model="gemini-2.5-flash-lite"):
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
            
            response = model_instance.generate_content(question, stream=True)
            
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