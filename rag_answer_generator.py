# /workspace/new_answer_generator.py
import google.generativeai as genai
import os
import chromadb # ★追加

# API設定
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"[ERROR] Google genai config error: {e}")

# --- ★ RAG用の準備 ---
class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input: list[str]) -> list[list[float]]:
        model = 'models/text-embedding-004'
        # クエリ用なので task_type="retrieval_query" にする
        return [
            genai.embed_content(model=model, content=text, task_type="retrieval_query")['embedding']
            for text in input
        ]

# DBクライアントの初期化 (PersistentClientで読み込み)
try:
    # setupで作成したパスと同じ場所を指定
    db_client = chromadb.PersistentClient(path="./chroma_db")
    collection = db_client.get_collection(
        name="real_estate_knowledge",
        embedding_function=GeminiEmbeddingFunction()
    )
    print("[INFO] ChromaDB loaded successfully.")
except Exception as e:
    print(f"[WARN] ChromaDB could not be loaded: {e}")
    collection = None


def generate_answer_stream(question: str, model="gemini-2.5-flash-lite"):
    print(f"[DEBUG] RAG回答生成開始: '{question}'")
    
    if not question:
        yield "質問を聞き取れませんでした。"
        return

    # --- ★ 1. 検索 (Retrieval) ---
    retrieved_context = ""
    if collection:
        try:
            # 質問に近い上位3件を取得
            results = collection.query(
                query_texts=[question],
                n_results=3
            )
            # 検索結果のテキストを結合
            retrieved_texts = results['documents'][0]
            retrieved_context = "\n".join([f"- {text}" for text in retrieved_texts])
            print(f"[DEBUG] 検索結果:\n{retrieved_context}")
        except Exception as e:
            print(f"[ERROR] 検索失敗: {e}")

    # --- ★ 2. プロンプト構築 (Augmentation) ---
    # 不動産屋として振る舞うためのシステムプロンプト
    system_prompt = f"""
あなたは不動産会社の優秀な電話オペレーターです。
以下の【社内マニュアル・物件情報】を元に、顧客の質問に答えてください。

【社内マニュアル・物件情報】
{retrieved_context}

【ルール】
1. 上記の情報に答えがある場合は、それを使って回答してください。
2. 情報に載っていないこと（例：他のエリアの物件、詳細な住所など）は、「申し訳ありません、手元の資料にはございませんので、担当者に確認いたします」と答えてください。勝手に捏造してはいけません。
3. 丁寧で安心感のある口調で話してください。
4. 回答は長くなりすぎないよう、簡潔にまとめてください。
"""

    try:
        if model.startswith("gemini-"):
            if not GOOGLE_API_KEY:
                raise ValueError("No API Key")
            
            model_instance = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_prompt # ここに検索結果入りプロンプトを渡す
            )
            
            response = model_instance.generate_content(question, stream=True)
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        else:
            yield f"Unknown model: {model}"

    except Exception as e:
        print(f"[ERROR] Generate error: {e}")
        yield "申し訳ありません、システムエラーが発生しました。"

# 単体テスト用
if __name__ == "__main__":
    # テスト実行時にDB検索が動くか確認
    q = "ペット可の物件はありますか？"
    print(f"Q: {q}")
    for txt in generate_answer_stream(q):
        print(txt, end="", flush=True)
    print()