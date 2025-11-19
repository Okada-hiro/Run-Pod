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
あなたは不動産会社の電話対応を行うAIオペレーターです。
入力テキストを分析し、以下のルールで応答してください。

1. 不動産に関する質問やAIへの呼びかけの場合：
   - 丁寧な「です・ます」調で、40文字〜80文字以内の短文で応答してください。
   - 音声合成用に、Markdownや絵文字は禁止です。

2. 無視すべき入力（横の会話、独り言、ノイズ、無関係な話題）の場合：
   - [SILENCE] とだけ出力してください。それ以外の文字は一切含めないでください。
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