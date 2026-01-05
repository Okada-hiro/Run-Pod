import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l


# --- Added Functions for Accent/End-of-sentence improvement ---

def mel_loss_with_sentence_end_weight(
    mel_pred: torch.Tensor,
    mel_gt: torch.Tensor,
    lengths: torch.Tensor,
    start_indices: torch.Tensor = None,
    end_weight: float = 2.0,
    end_ratio: float = 0.2,
):
    """
    文末フレームに重みを付けた mel L1 損失 (VITS Slice対応版)

    Parameters
    ----------
    mel_pred : (B, n_mel, T_slice)
    mel_gt   : (B, n_mel, T_slice)
    lengths  : (B,) 各サンプルの元の全有効フレーム長 (spec_lengths)
    start_indices : (B,) 各スライスの開始フレームインデックス (ids_slice)。
                    Noneの場合は全区間とみなす。
    end_weight : 文末の最大重み
    end_ratio  : 文末として扱う割合（例: 0.2 = 最後の20%）

    Returns
    -------
    loss : scalar
    """
    B, _, T = mel_pred.shape
    device = mel_pred.device

    # 基本の L1 誤差
    diff = torch.abs(mel_pred - mel_gt)  # (B, n_mel, T)

    # 重み行列の初期化 (全て1)
    weights = torch.ones((B, T), device=device)

    for b in range(B):
        L = lengths[b].item()
        end_len = int(L * end_ratio)

        if end_len > 0:
            # 文末領域の開始絶対位置
            ramp_start_global = L - end_len
            
            # 現在のスライスの絶対位置範囲
            if start_indices is not None:
                slice_start = start_indices[b].item()
            else:
                slice_start = 0
            
            slice_end = slice_start + T
            
            # スライス内の各時点の絶対位置
            # shape: (T,)
            current_indices = torch.arange(slice_start, slice_end, device=device)

            # 文末領域に入っているインデックスのマスク
            # ramp_start_global <= idx < L
            mask = (current_indices >= ramp_start_global) & (current_indices < L)

            if mask.any():
                # 重みを計算: 1.0 -> end_weight へ線形補間
                # 位置の正規化 (0.0 ~ 1.0)
                # global_idx = ramp_start_global のとき 0, L のとき 1 (に近い)
                
                # 対象となるインデックス
                valid_indices = current_indices[mask]
                
                # 重み計算: 1.0 + (end_weight - 1.0) * (pos / end_len)
                rel_pos = (valid_indices - ramp_start_global).float() / end_len
                # 念のためクリップ
                rel_pos = torch.clamp(rel_pos, 0.0, 1.0)
                
                ramp_weights = 1.0 + (end_weight - 1.0) * rel_pos
                
                # weightsに適用 (maskの位置に)
                weights[b, mask] = ramp_weights

    # (B, 1, T) に拡張して mel 次元にブロードキャスト
    weights = weights.unsqueeze(1)

    loss = torch.sum(diff * weights) / torch.sum(weights)
    return loss


def delta_loss(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    lengths: torch.Tensor = None,
    order: int = 1,
):
    """
    変化量（Δ）損失
    mel / f0 / energy など時間系列なら何でも使える

    Parameters
    ----------
    x_pred : (B, C, T) or (B, T)
    x_gt   : (B, C, T) or (B, T)
    lengths: (B,) マスク用。VITSのスライス済みの場合はNoneで良い（全区間有効）
    order  : 差分階数（1=Δ, 2=ΔΔ）

    Returns
    -------
    loss : scalar
    """

    if x_pred.dim() == 2:
        x_pred = x_pred.unsqueeze(1)
        x_gt = x_gt.unsqueeze(1)

    # 差分計算
    def compute_delta(x):
        for _ in range(order):
            x = x[:, :, 1:] - x[:, :, :-1]
        return x

    delta_pred = compute_delta(x_pred)
    delta_gt = compute_delta(x_gt)

    # 長さ調整 (VITSのスライス学習時はlengths=Noneで全要素計算してOK)
    if lengths is not None:
        max_delta_len = delta_pred.size(2)
        # lengths は元の長さだが、ここではdelta後の長さに合わせる必要がある
        # 簡易的に、スライス長より長ければマスク不要、短ければマスク
        mask = torch.arange(max_delta_len, device=x_pred.device)[None, :] < (lengths[:, None] - order)
        mask = mask.unsqueeze(1).float()
        loss = torch.sum(torch.abs(delta_pred - delta_gt) * mask) / (torch.sum(mask) + 1e-6)
    else:
        loss = torch.mean(torch.abs(delta_pred - delta_gt))

    return loss

# -------------------------------------------------------------


class WavLMLoss(torch.nn.Module):
    def __init__(self, model, wd, model_sr, slm_sr=16000):
        super(WavLMLoss, self).__init__()
        self.wavlm = AutoModel.from_pretrained(model)
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def forward(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16, output_hidden_states=True
        ).hidden_states

        floss = 0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))

        return floss.mean()

    def generator(self, y_rec):
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16, output_hidden_states=True
        ).hidden_states
        y_rec_embeddings = (
            torch.stack(y_rec_embeddings, dim=1)
            .transpose(-1, -2)
            .flatten(start_dim=1, end_dim=2)
        )
        y_df_hat_g = self.wd(y_rec_embeddings)
        loss_gen = torch.mean((1 - y_df_hat_g) ** 2)

        return loss_gen

    def discriminator(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self.wavlm(
                input_values=y_rec_16, output_hidden_states=True
            ).hidden_states

            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )
            y_rec_embeddings = (
                torch.stack(y_rec_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)

        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs

        r_loss = torch.mean((1 - y_df_hat_r) ** 2)
        g_loss = torch.mean((y_df_hat_g) ** 2)

        loss_disc_f = r_loss + g_loss

        return loss_disc_f.mean()

    def discriminator_forward(self, wav):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings)

        return y_d_rs