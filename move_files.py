# move_files.py (修正後)
import requests

def move_file(file_path):
    """
    指定したファイルをRunPodのPodに送信する関数
    成功時は True、失敗時は False を返す
    """
    #url = "https://lb7lmstivfh9eh-5000.proxy.runpod.net/upload"
    #url = "https://h4o274t83xeumy-8000.proxy.runpod.net/upload"
    url = "https://8vm9dxp402l5oh-5000.proxy.runpod.net/upload"  # 古いURL（使用しないでください）
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, timeout=30) # タイムアウトを設定
         
        print(f"ステータスコード: {response.status_code}")
        print(f"サーバ応答: {response.text}")
        
        # ステータスコードが 200番台なら成功
        if 200 <= response.status_code < 300:
            return True
        else:
            return False
    
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {file_path}")
        return False
    except requests.exceptions.RequestException as e: # ネットワークエラーをキャッチ
        print(f"送信中にネットワークエラーが発生しました: {e}")
        return False
    except Exception as e:
        print(f"送信中に予期せぬエラーが発生しました: {e}")
        return False

# ... (if __name__ == "__main__": の部分は変更なし) ...
# 確認用
if __name__ == "__main__":
    test_file = "/Users/okadahiroaki/Music/Music/Media.localized/Music/Unknown Artist/Unknown Album/日本語音源-2.wav"  # ここに送信したいファイル名を指定
    move_file(test_file)