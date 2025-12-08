import sys
import os
from pathlib import Path

# --- パス設定 ---
# new_text_to_speech.py と同様にリポジトリをパスに追加
WORKSPACE_DIR = os.getcwd()
REPO_PATH = os.path.join(WORKSPACE_DIR, "Style_Bert_VITS2")
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

try:
    # 辞書更新用の関数をインポート
    from style_bert_vits2.nlp.japanese.user_dict import update_dict
    from style_bert_vits2.constants import Languages
except ImportError as e:
    print(f"[ERROR] Style-Bert-VITS2のライブラリが見つかりません: {e}")
    sys.exit(1)

def apply_dictionary():
    # 作成したCSVのパス
    csv_path = Path("/workspace/Style_Bert_VITS2/dict_data/user_dict.csv")
    
    if not csv_path.exists():
        print(f"[ERROR] 辞書CSVが見つかりません: {csv_path}")
        print("先に create_user_dict.py を実行してください。")
        return

    print(f"[INFO] 辞書をコンパイルしています... (Source: {csv_path})")
    
    # update_dict関数を実行
    # これにより、ライブラリが使用するバイナリ辞書(user.dict)が適切な場所に生成・更新されます
    # ※バージョンによって引数が異なる場合がありますが、基本的にはパスを指定しなくても
    #   デフォルトの場所(dict_data)を見てくれることが多いです。
    #   念のため明示的に動くように実装します。
    
    try:
        # 方法A: 引数なしで実行（デフォルトの dict_data/user_dict.csv を見に行く設定の場合）
        # 多くの場合、これで user_dict.csv を読み込み user.dict を生成します
        update_dict() 
        print("[SUCCESS] 辞書のコンパイルと適用が完了しました！")
        
    except Exception as e:
        print(f"[WARNING] デフォルト設定での更新に失敗しました: {e}")
        print("引数を指定して再試行します...")
        try:
            # 方法B: パスを明示的に指定
            # 出力先などはライブラリのデフォルトに任せます
            env_dict_path = Path(REPO_PATH) / "dict_data" / "user_dict.csv"
            update_dict(env_dict_path)
            print("[SUCCESS] 辞書のコンパイルと適用が完了しました！(パス指定)")
        except Exception as e2:
             print(f"[ERROR] 辞書の更新に失敗しました: {e2}")

if __name__ == "__main__":
    apply_dictionary()