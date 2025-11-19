import chromadb
import google.generativeai as genai
import os

# APIキーの設定
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY is not set.")
    exit(1)
genai.configure(api_key=GOOGLE_API_KEY)

# --- 1. 社内知識データ（本来はファイルから読み込む） ---
documents = [
    # 物件情報
    "物件A: 渋谷区神南1丁目、1LDK、家賃25万円、即入居可。ペット不可。",
    "物件B: 世田谷区三軒茶屋、2DK、家賃18万円、入居時期は来月から。ペット相談可。",
    "物件C: 港区六本木、Studioタイプ、家賃35万円、礼金なしキャンペーン中。",
    
    # コールセンター対応マニュアル
    "家賃の値下げ交渉について: 原則として電話ではお答えできません。「担当者に確認して折り返します」と伝えてください。",
    "内見の予約について: 希望日時を3つほど伺ってください。その後、営業担当のスケジュールを確認します。",
    "保証人について: 原則として保証会社を利用していただきますが、詳細は店舗での相談となります。",
    "退去の手続き: 退去希望日の1ヶ月前までにご連絡いただく必要があります。"
]

# IDとメタデータ
ids = [str(i) for i in range(len(documents))]
metadatas = [{"source": "internal_manual"} for _ in range(len(documents))]

# --- 2. 埋め込み関数の定義 (Gemini Embeddings) ---
class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input: list[str]) -> list[list[float]]:
        # Geminiの埋め込みモデルを使用
        model = 'models/text-embedding-004' 
        # バッチ処理でEmbeddingsを取得
        embeddings = [
            genai.embed_content(model=model, content=text, task_type="retrieval_document")['embedding']
            for text in input
        ]
        return embeddings

# --- 3. データベースの作成と保存 ---
print("データベースを作成中...")

# ディレクトリに保存して永続化する
client = chromadb.PersistentClient(path="./chroma_db") 

# コレクション（テーブルのようなもの）を作成
# 既存なら取得、なければ作成
collection = client.get_or_create_collection(
    name="real_estate_knowledge",
    embedding_function=GeminiEmbeddingFunction()
)

# データを追加（自動的にベクトル化される）
collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print(f"完了: {len(documents)} 件のデータを登録しました。path='./chroma_db'")