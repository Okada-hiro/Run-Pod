import re
import MeCab
import unidic_lite
from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp.symbols import PUNCTUATIONS

TAGGER = MeCab.Tagger(f"-d {unidic_lite.DICDIR}")
PAUSE_SYMBOL = '_' 

# === 辞書定義 ===
DIGRAPHS = {
    "キャ": ["k", "y", "a"],
    "キュ": ["k", "y", "u"],
    "キョ": ["k", "y", "o"],
    "シャ": ["sh", "a"],
    "シュ": ["sh", "u"],
    "ショ": ["sh", "o"],
    "チャ": ["ch", "a"],
    "チュ": ["ch", "u"],
    "チョ": ["ch", "o"],
    "ニャ": ["n", "y", "a"],
    "ニュ": ["n", "y", "u"],
    "ニョ": ["n", "y", "o"],
    "ヒャ": ["h", "y", "a"],
    "ヒュ": ["h", "y", "u"],
    "ヒョ": ["h", "y", "o"],
    "ミャ": ["m", "y", "a"],
    "ミュ": ["m", "y", "u"],
    "ミョ": ["m", "y", "o"],
    "リャ": ["r", "y", "a"],
    "リュ": ["r", "y", "u"],
    "リョ": ["r", "y", "o"],
    "ギャ": ["g", "y", "a"],
    "ギュ": ["g", "y", "u"],
    "ギョ": ["g", "y", "o"],
    "ジャ": ["j", "a"],
    "ジュ": ["j", "u"],
    "ジョ": ["j", "o"],
    "ビャ": ["b", "y", "a"],
    "ビュ": ["b", "y", "u"],
    "ビョ": ["b", "y", "o"],
    "ピャ": ["p", "y", "a"],
    "ピュ": ["p", "y", "u"],
    "ピョ": ["p", "y", "o"],
    "ティ": ["t", "i"],
    "ディ": ["d", "i"],
    "シェ": ["sh", "e"],
    "ジェ": ["j", "e"],
    "チェ": ["ch", "e"],
    "トゥ": ["t", "u"],
    "ドゥ": ["d", "u"],
    "ウィ": ["w", "i"],
    "ウェ": ["w", "e"],
    "ウォ": ["w", "o"],
    "ファ": ["f", "a"],
    "フィ": ["f", "i"],
    "フェ": ["f", "e"],
    "フォ": ["f", "o"],
    "フュ": ["f", "y", "u"],
}

KANA_TO_PHONEME = {
    "ア": ["a"],
    "イ": ["i"],
    "ウ": ["u"],
    "エ": ["e"],
    "オ": ["o"],
    "カ": ["k", "a"],
    "キ": ["k", "i"],
    "ク": ["k", "u"],
    "ケ": ["k", "e"],
    "コ": ["k", "o"],
    "サ": ["s", "a"],
    "シ": ["sh", "i"],
    "ス": ["s", "u"],
    "セ": ["s", "e"],
    "ソ": ["s", "o"],
    "タ": ["t", "a"],
    "チ": ["ch", "i"],
    "ツ": ["ts", "u"],
    "テ": ["t", "e"],
    "ト": ["t", "o"],
    "ナ": ["n", "a"],
    "ニ": ["n", "i"],
    "ヌ": ["n", "u"],
    "ネ": ["n", "e"],
    "ノ": ["n", "o"],
    "ハ": ["h", "a"],
    "ヒ": ["h", "i"],
    "フ": ["f", "u"],
    "ヘ": ["h", "e"],
    "ホ": ["h", "o"],
    "マ": ["m", "a"],
    "ミ": ["m", "i"],
    "ム": ["m", "u"],
    "メ": ["m", "e"],
    "モ": ["m", "o"],
    "ヤ": ["y", "a"],
    "ユ": ["y", "u"],
    "ヨ": ["y", "o"],
    "ラ": ["r", "a"],
    "リ": ["r", "i"],
    "ル": ["r", "u"],
    "レ": ["r", "e"],
    "ロ": ["r", "o"],
    "ワ": ["w", "a"],
    "ヲ": ["o"],
    "ン": ["N"],
    "ガ": ["g", "a"],
    "ギ": ["g", "i"],
    "グ": ["g", "u"],
    "ゲ": ["g", "e"],
    "ゴ": ["g", "o"],
    "ザ": ["z", "a"],
    "ジ": ["j", "i"],
    "ズ": ["z", "u"],
    "ゼ": ["z", "e"],
    "ゾ": ["z", "o"],
    "ダ": ["d", "a"],
    "ヂ": ["j", "i"],
    "ヅ": ["z", "u"],
    "デ": ["d", "e"],
    "ド": ["d", "o"],
    "バ": ["b", "a"],
    "ビ": ["b", "i"],
    "ブ": ["b", "u"],
    "ベ": ["b", "e"],
    "ボ": ["b", "o"],
    "パ": ["p", "a"],
    "ピ": ["p", "i"],
    "プ": ["p", "u"],
    "ペ": ["p", "e"],
    "ポ": ["p", "o"],
    "ヴ": ["v", "u"],
}

# === bert_feature.py用関数 ===
def text_to_sep_kata(text, raise_yomi_error=False):
    node = TAGGER.parseToNode(text)
    res = []
    is_katakana_regex = re.compile(r'[\u30A1-\u30F4]+')
    
    while node:
        if node.feature.startswith("BOS/EOS"):
            node = node.next
            continue
        
        features = node.feature.split(",")
        
        # g2pと同じ読み取得ロジック
        reading = node.surface
        if len(features) > 9 and is_katakana_regex.fullmatch(features[9]):
            reading = features[9]
        elif len(features) > 6 and is_katakana_regex.fullmatch(features[6]):
            reading = features[6]
            
        res.append(reading)
        node = node.next
    
    return res, None


def g2p(
    norm_text: str, use_jp_extra: bool = True, raise_yomi_error: bool = False
) -> tuple[list[str], list[int], list[int]]:
    
    node = TAGGER.parseToNode(norm_text)
    full_phonemes = []
    full_tones = []
    word2ph = [1] 
    
    full_phonemes.append(PAUSE_SYMBOL)
    full_tones.append(0) 

    is_katakana_regex = re.compile(r'[\u30A1-\u30F4]+')
    
    while node:
        if node.feature.startswith("BOS/EOS"):
            node = node.next
            continue
        
        features = node.feature.split(",")
        surface = node.surface
        pos_major = features[0] # 品詞大分類
        
        # 読み取得
        reading = surface 
        if len(features) > 9 and is_katakana_regex.fullmatch(features[9]):
            reading = features[9]
        elif len(features) > 6 and is_katakana_regex.fullmatch(features[6]):
            reading = features[6]

        # アクセント型
        acc_type = 0
        try:
            for f in reversed(features):
                f_clean = f.strip().replace('"', '')
                if f_clean.isdigit():
                    acc_type = int(f_clean)
                    break
        except:
            acc_type = 0

        # 記号処理
        if surface in ['、', '。', '！', '？', '!', '?', ' ', '　'] or reading == '*':
            current_phs = [PAUSE_SYMBOL]
            current_tones = [0]
            full_phonemes.extend(current_phs)
            full_tones.extend(current_tones)
            # 記号の場合でも reading (またはsurface) の長さに合わせる
            word2ph.extend(_distribute_evenly(len(current_phs), len(reading)))
            node = node.next
            continue

        # モーラ分解
        moras, hl_list = _accent_to_HL(reading, acc_type)
        
        node_phonemes = []
        node_tones = []
        
        # 名詞のみトーン適用
        use_mecab_tone = (pos_major == "名詞")

        for m_idx, mora in enumerate(moras):
            m_phs = []
            if mora in DIGRAPHS: m_phs = DIGRAPHS[mora]
            elif mora in KANA_TO_PHONEME: m_phs = KANA_TO_PHONEME[mora]
            else: 
                sub_phs = []
                for char in mora:
                    if char in KANA_TO_PHONEME: sub_phs.extend(KANA_TO_PHONEME[char])
                    else: sub_phs.append(PAUSE_SYMBOL)
                m_phs = sub_phs if sub_phs else [PAUSE_SYMBOL]

            final_phs = []
            for p in m_phs:
                if p == '-' and len(node_phonemes) > 0:
                    prev = node_phonemes[-1]
                    if prev in ['a','i','u','e','o']: final_phs.append(prev)
                    else: final_phs.append(p)
                elif p == '-' and len(node_phonemes) == 0:
                     final_phs.append(PAUSE_SYMBOL)
                else:
                    final_phs.append(p)
            
            m_phs = final_phs

            if use_mecab_tone:
                val = 1 if (m_idx < len(hl_list) and hl_list[m_idx] == 'H') else 0
            else:
                val = 0
            
            node_phonemes.extend(m_phs)
            node_tones.extend([val] * len(m_phs))

        full_phonemes.extend(node_phonemes)
        full_tones.extend(node_tones)

        # --- word2ph の修正 ---
        # BERT特徴量抽出(bert_feature.py)はテキストを全て「読み(カタカナ)」に変換して処理します。
        # そのため、word2phも「元の表記(漢字)」ではなく「読み(カタカナ)」の文字数に合わせる必要があります。
        # 以前のロジック(surfaceに合わせる処理)を廃止し、常に reading の長さに均等配分します。
        word2ph.extend(_distribute_evenly(len(node_phonemes), len(reading)))

        node = node.next

    full_phonemes.append(PAUSE_SYMBOL)
    full_tones.append(0)
    word2ph.append(1)

    # 整合性チェック (このチェックは phonemes と word2ph の合計が合うかを見るものなので、このままでOK)
    if len(phones := full_phonemes) != sum(word2ph):
        # Fallback (ここに入ると漢字ベースで再計算されるため、BERTとズレる可能性があるが、緊急回避として残す)
        pure_char_len = len(norm_text)
        pure_ph_len = len(phones) - 2
        new_w2p = _distribute_evenly(pure_ph_len, pure_char_len)
        word2ph = [1] + new_w2p + [1]

    if not use_jp_extra:
        phones = [phone if phone != "N" else "n" for phone in phones]

    return phones, full_tones, word2ph

# Helper
def _katakana_to_moras(text):
    small = set("ャュョァィゥェォ")
    moras = []
    i = 0
    while i < len(text):
        if i + 1 < len(text) and text[i + 1] in small:
            moras.append(text[i:i+2])
            i += 2
        else:
            moras.append(text[i])
            i += 1
    return moras

def _accent_to_HL(pronunciation, accent_type):
    moras = _katakana_to_moras(pronunciation)
    n = len(moras)
    HL = []
    if n == 0: return moras, []
    if accent_type == 0:
        if n == 1: HL = ["L"]
        else: HL = ["L"] + ["H"] * (n - 1)
    elif accent_type == 1:
        HL = ["H"] + ["L"] * (n - 1)
    else:
        for i in range(n):
            if i == 0: HL.append("L")
            elif i < accent_type: HL.append("H")
            else: HL.append("L")
    return moras, HL

def _distribute_evenly(n_ph, n_char):
    if n_char == 0: return []
    base = n_ph // n_char
    remainder = n_ph % n_char
    counts = [base] * n_char
    for i in range(remainder): counts[i] += 1
    return counts

def _is_kana_only(text):
    return re.fullmatch(r'[ァ-ヴーぁ-ん・]+', text) is not None