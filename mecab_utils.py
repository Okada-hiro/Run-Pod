import torch
import MeCab
import unidic_lite

class MeCabTokenizer:
    def __init__(self):
        # unidic-liteを使用
        self.tagger = MeCab.Tagger(f"-d {unidic_lite.DICDIR}")
        
        # 品詞の語彙定義
        # 0はPadding/Unknown用
        self.pos_list = [
            "UNK", "名詞", "動詞", "助詞", "形容詞", "副詞", 
            "助動詞", "連体詞", "感動詞", "接続詞", "記号", "フィラー",
            "接頭辞", "接尾辞"
        ]
        self.pos2idx = {p: i for i, p in enumerate(self.pos_list)}
        self.vocab_size = len(self.pos_list)

    def text_to_ids(self, text):
        """テキストをMeCab品詞ID列に変換"""
        node = self.tagger.parseToNode(text)
        ids = []
        while node:
            if node.surface:
                features = node.feature.split(",")
                pos = features[0]
                idx = self.pos2idx.get(pos, 0)
                ids.append(idx)
            node = node.next
        return torch.tensor(ids, dtype=torch.long)

    # ★前回不足していたメソッドを追加★
    def batch_text_to_ids_from_ids(self, ids_list):
        """IDリストのリストを受け取り、パディングしてTensor化"""
        # バッチ内の最大長を取得
        max_len = max([len(x) for x in ids_list])
        
        padded_batch = []
        for ids in ids_list:
            if len(ids) < max_len:
                # 0埋めパディング
                pad = torch.zeros(max_len - len(ids), dtype=torch.long)
                ids = torch.cat([ids, pad])
            else:
                ids = ids[:max_len]
            padded_batch.append(ids)
            
        return torch.stack(padded_batch)