import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re

# --- 設定 ---
# 複数の単語を一気に取得したい場合はここに追加してください
TARGET_WORDS = ["こんにちは", "こんばんは", "ありがとう"] 
OUTPUT_FILE = "accents.json"

def main():
    print("ブラウザを起動しています...")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    
    extracted_data = {}

    try:
        url = "https://accentjiten.com"
        print(f"アクセス中: {url}")
        driver.get(url)
        
        # 検索窓待機
        wait = WebDriverWait(driver, 10)
        search_input = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text']"))
        )

        for target_word in TARGET_WORDS:
            print(f"\n--- '{target_word}' を検索します ---")
            
            # 入力クリア & 入力
            search_input.clear()
            search_input.send_keys(target_word)
            
            # 検索結果が変わるのを少し待つ
            time.sleep(1.5) 

            # 現在のHTMLを解析
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            
            tone_divs = soup.find_all("div", class_="tonetext")
            
            found_for_this_word = False
            
            for tone_div in tone_divs:
                spans = tone_div.find_all("span", recursive=False)
                
                full_text = ""
                tones = []
                
                for span in spans:
                    char = span.get_text(strip=True)
                    full_text += char
                    
                    classes = span.get("class", [])
                    
                    # ★★★ ここを修正しました ★★★
                    # "high"を含む(in)ではなく、"high"で始まる(startswith)クラスだけを1にする
                    # これで 'lowtonenexthigh' (低) を誤検知しなくなります
                    is_high = False
                    for c in classes:
                        if c.startswith("high"): # hightone, hightonenextlow
                            is_high = True
                            break
                    
                    tones.append(1 if is_high else 0)
                
                # ターゲット一致チェック
                if target_word in full_text:
                    # 部分一致した場合、その場所を切り出す
                    try:
                        start_index = full_text.index(target_word)
                        end_index = start_index + len(target_word)
                        target_tones = tones[start_index:end_index]
                        
                        extracted_data[target_word] = target_tones
                        print(f"[GET] {target_word} -> {target_tones}")
                        found_for_this_word = True
                        break
                    except:
                        pass
            
            if not found_for_this_word:
                print(f"[警告] '{target_word}' のデータが見つかりませんでした。")

        # 保存
        if extracted_data:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(extracted_data, f, ensure_ascii=False, indent=2)
            print("\n" + "=" * 30)
            print(f"[成功] '{OUTPUT_FILE}' を作成しました！")
            print("RunPodにアップロードしてください。")
        else:
            print("\n[失敗] データが一つも取れませんでした。")

    except Exception as e:
        print(f"エラー: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()