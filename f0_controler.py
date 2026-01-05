import torch
import numpy as np

# あなたのコードが出力する「モーラの核となる音素」
# これらが来たら「1モーラ終わった」とみなしてアクセントを次に進めます
MORA_CONSUMING_PHONEMES = set(['a', 'i', 'u', 'e', 'o', 'N', 'cl'])

# アクセント付与対象外の音素
PAUSES = set(['pau', 'sil', 'sp'])

def align_phonemes_to_accent(phonemes, accent_hl_list):
    """
    音素リスト(phonemes) と モーラ単位H/Lリスト(accent_hl_list) を同期させる。
    
    Args:
        phonemes: list of str (例: ['k', 'o', 'N', 'n', 'i', 'c', 'h', 'i', 'w', 'a'])
        accent_hl_list: list of str (例: ['L', 'H', 'H', 'H', 'L']) from MeCab
    Returns:
        phoneme_hl_map: 音素ごとのH/Lリスト (例: ['L', 'L', 'H', 'H', ...])
    """
    phoneme_hl_map = []
    mora_index = 0
    
    # 安全策：アクセント情報が空の場合は全部Lにする
    if not accent_hl_list:
        return ["L"] * len(phonemes)

    for ph in phonemes:
        # 1. ポーズの場合: 強制的にLow (または無音扱い) にして、モーラは進めない
        if ph in PAUSES:
            phoneme_hl_map.append("L")
            continue

        # 2. 現在のアクセントを取得
        # (MeCabの解析ズレなどでインデックスを超えた場合は、最後の値を維持する安全策)
        if mora_index < len(accent_hl_list):
            current_acc = accent_hl_list[mora_index]
        else:
            current_acc = accent_hl_list[-1]

        phoneme_hl_map.append(current_acc)

        # 3. モーラを消化する音素（母音・ん・っ）なら、インデックスを進める
        if ph in MORA_CONSUMING_PHONEMES:
            mora_index += 1
            
    return phoneme_hl_map


def create_f0_tensor(phoneme_hl_map, predicted_durations, base_freq=220.0, pitch_range=1.25):
    """
    音素ごとのH/Lと、モデルが予測したDurationを使ってF0カーブを作る
    
    Args:
        phoneme_hl_map: align_phonemes_to_accent の出力
        predicted_durations: モデルが出力した w_ceil (Tensor or List)
        base_freq: ベースとなる周波数 (Low)
        pitch_range: Highは base_freq の何倍か (例: 1.25倍)
    """
    low_hz = base_freq
    high_hz = base_freq * pitch_range
    
    f0_values = []
    
    # Tensorの場合はCPUへリスト化
    if isinstance(predicted_durations, torch.Tensor):
        durations_list = predicted_durations.cpu().numpy()
    else:
        durations_list = predicted_durations

    # 長さが合わない場合の安全策 (短い方に合わせる)
    min_len = min(len(phoneme_hl_map), len(durations_list))
    
    for i in range(min_len):
        hl = phoneme_hl_map[i]
        dur = int(durations_list[i])
        
        target_hz = high_hz if hl == 'H' else low_hz
        f0_values.extend([target_hz] * dur)
        
    # 平滑化処理 (移動平均)
    f0_array = np.array(f0_values)
    window_size = 16 # 少し強めに滑らかにする
    
    if len(f0_array) > window_size:
        # padding mode='edge' で端っこの落ち込みを防ぐ
        f0_smooth = np.convolve(f0_array, np.ones(window_size)/window_size, mode='same')
    else:
        f0_smooth = f0_array

    # 形状を [1, 1, T] にしてTensor化
    return torch.from_numpy(f0_smooth).float().unsqueeze(0).unsqueeze(0)