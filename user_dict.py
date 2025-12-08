import csv
import os

# --- 設定 ---
OUTPUT_DIR = "/workspace/Style_Bert_VITS2/dict_data"
OUTPUT_FILE = "user_dict.csv"

# 登録したい単語リスト
# フォーマット: ("単語", "読み(カタカナ)", "品詞", "品詞細分類1")
# 
# ★重要ポイント:
# 「こんにちは」は「感動詞」にすることで、初めて挨拶としてのアクセントが適用されます。
# 「名詞」のままだと「今日(名詞)+は」と解釈されて失敗します。
FIX_DATA = [
    # (単語, 読み, 品詞, 品詞細分類)
    ("こんにちは", "コンニチワ", "感動詞", "*"),
    ("Style-Bert-VITS2", "スタイルバートビッツツー", "名詞", "固有名詞"),
    ("雨", "アメ", "名詞", "一般"),
    ("飴", "アメ", "名詞", "一般"),
    ("おはようございます", "オハヨウゴザイマス", "感動詞", "*"),
    ("ありがとうございます", "アリガトウゴザイマス", "感動詞", "*"),
]

def create_user_dict():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    print(f"[INFO] {output_path} を生成中...")

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        # OpenJTalk(MeCab)用フォーマット:
        # 表層形,左文脈ID,右文脈ID,コスト,品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
        
        for surface, yomi, pos, sub_pos in FIX_DATA:
            # 左文脈ID, 右文脈ID: 0 (自動推定させるため0でOK。本来は品詞ごとに決まっているがユーザー辞書では0が一般的)
            # コスト: 1 (最小値にして、システム辞書よりも優先度を最大にする)
            
            line = f"{surface},0,0,1,{pos},{sub_pos},*,*,*,*,{surface},{yomi},{yomi}"
            f.write(line + "\n")
            
    print(f"[SUCCESS] {len(FIX_DATA)}語を登録しました。")
    print("--------------------------------------------------")
    print("【重要】反映させるために以下を実行してください:")
    print("cd /workspace/Style_Bert_VITS2")
    print("python initialize.py")
    print("--------------------------------------------------")

if __name__ == "__main__":
    create_user_dict()