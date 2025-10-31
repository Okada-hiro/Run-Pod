from flask import Flask, request
import os
import uuid

app = Flask(__name__)

# 保存先のディレクトリを incoming_audio に固定
SAVE_DIR = "incoming_audio"
os.makedirs(SAVE_DIR, exist_ok=True)  # ディレクトリがなければ作成

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    
    # ファイル名を安全に処理
    filename = file.filename
    if isinstance(filename, bytes):
        # bytesの場合はUTF-8デコード
        filename = filename.decode('utf-8', errors='ignore')
    
    # 保存ファイル名を "received_FILENAME" にする
    safe_filename = f"received_{filename}"

    # 文字化けが気になる場合は UUID に置き換えることも可能
    # ext = os.path.splitext(filename)[1]
    # safe_filename = f"received_{uuid.uuid4().hex}{ext}"

    save_path = os.path.join(SAVE_DIR, safe_filename)
    
    try:
        file.save(save_path)
        print(f"File saved to: {save_path}")  # ログ出力
        return f"File received and saved to {save_path}!", 200
    except Exception as e:
        print(f"File save error: {e}")
        return f"Error saving file: {e}", 500

if __name__ == "__main__":
    # ポート5000で起動
    app.run(host="0.0.0.0", port=5000, debug=True)
