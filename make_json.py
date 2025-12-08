import json
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pyopenjtalk # ★追加: 音素変換に必要

# --- 設定 ---
TARGET_WORDS = ["今日", "午後", "雨", "予報", "傘", "降る", "持って", "安心"] 
OUTPUT_FILE = "accents.json"

def convert_mora_to_phoneme_tones(word, mora_tones):
    """
    文字単位(モーラ)のトーンを、音素単位のトーンに変換・拡張する
    例: word="あめ", mora_tones=[1, 0] 
        -> 音素: a, m, e
        -> トーン: 1, 0, 0
    """
    # 1. 単語を音素に分解 (pyopenjtalkを使用)
    # g2pの結果例: "a m e" (スペース区切り)
    phoneme_str = pyopenjtalk.g2p(word, kana=False)
    phones = phoneme_str.split(" ")
    
    # "pau"や"sil"などの無音区間を除去
    phones = [p for p in phones if p not in ('pau', 'sil', 'cl')]
    
    # 2. 音素とモーラの対応付け（簡易ロジック）
    # 母音(a,i,u,e,o,N)や、拗音の母音部分で「1拍」とカウントし、
    # その拍に属する子音には、その拍のトーンを適用する。
    
    converted_tones = []
    mora_index = 0
    
    # 母音・撥音・長音の判定用セット
    vowels = {'a', 'i', 'u', 'e', 'o', 'N', 'I', 'U', 'E', 'O'}
    
    # ※厳密なアライメントは難しいですが、ルールベースで近似します
    # 子音は「次の母音」と同じトーン、母音は「現在のモーラ」のトーン
    
    # 処理しやすいように、現在の音素がどのモーラに属するか判定
    # 基本方針: 母音が来たらモーラカウントアップ
    
    # 一時的なバッファ
    temp_tones = []
    
    current_mora_tone = 0
    if mora_index < len(mora_tones):
        current_mora_tone = mora_tones[mora_index]
        
    for p in phones:
        # トーンを追加
        converted_tones.append(current_mora_tone)
        
        # もし母音や撥音(N)なら、そのモーラは終了 -> 次のモーラへ
        if p in vowels:
            mora_index += 1
            if mora_index < len(mora_tones):
                current_mora_tone = mora_tones[mora_index]
    
    # 検証
    if len(converted_tones) != len(phones):
        print(f"[警告] 音素変換で長さ不一致: {word} (音素:{len(phones)} vs トーン:{len(converted_tones)})")
    
    return converted_tones

def main():
    print("ブラウザを起動しています...")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    
    extracted_data = {}

    try:
        url = "https://accentjiten.com"
        driver.get(url)
        
        wait = WebDriverWait(driver, 10)
        search_input = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text']"))
        )

        for target_word in TARGET_WORDS:
            print(f"\n--- '{target_word}' を検索します ---")
            search_input.clear()
            search_input.send_keys(target_word)
            time.sleep(1.5) 

            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            tone_divs = soup.find_all("div", class_="tonetext")
            
            found = False
            
            for tone_div in tone_divs:
                spans = tone_div.find_all("span", recursive=False)
                full_text = ""
                mora_tones = [] # モーラ単位のトーン
                
                for span in spans:
                    char = span.get_text(strip=True)
                    full_text += char
                    
                    classes = span.get("class", [])
                    is_high = False
                    for c in classes:
                        if c.startswith("high"):
                            is_high = True
                            break
                    mora_tones.append(1 if is_high else 0)
                
                if target_word in full_text:
                    try:
                        # 1. サイト上の表記からターゲット部分の「モーラトーン」を切り出す
                        start_index = full_text.index(target_word)
                        end_index = start_index + len(target_word)
                        target_mora_tones = mora_tones[start_index:end_index]
                        
                        print(f"  [取得] {target_word} (モーラ): {target_mora_tones}")

                        # 2. ★音素単位に変換★
                        phoneme_tones = convert_mora_to_phoneme_tones(target_word, target_mora_tones)
                        
                        extracted_data[target_word] = phoneme_tones
                        print(f"  [変換] {target_word} (音素)  : {phoneme_tones}")
                        
                        found = True
                        break 
                    except Exception as e:
                        print(f"  -> エラー: {e}")
            
            if not found:
                print(f"[警告] '{target_word}' が見つかりませんでした。")

        # 保存
        if extracted_data:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(extracted_data, f, ensure_ascii=False, indent=2)
            print("\n" + "=" * 30)
            print(f"[成功] 音素対応版 '{OUTPUT_FILE}' を作成しました！")
        else:
            print("\n[失敗] データなし")

    except Exception as e:
        print(f"エラー: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()