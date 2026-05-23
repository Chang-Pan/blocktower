import torch
import torch.nn as nn

# 特征布局 (与 neural_simulator.py 对齐):
# z0/输出: [B, obj_num, 17]
#   0:3   位置 pos (x, y, z)            —— 动态
#   3:7   四元数 quat (qx, qy, qz, qw)  —— 动态
#   7:10  尺寸 size (lx, ly, lz)        —— 静态
#   10:11 dynamic_mask                 —— 静态
#   11:14 速度 vel (vx, vy, vz)         —— 动态
#   14:17 角速度 angvel (wx, wy, wz)    —— 动态
FEATURE_DIM = 11  # pos3 + quat4 + size3 + mask1
STATE_DIM = 17    # FEATURE_DIM + vel3 + angvel3

# 喂给 Transformer 的输入维度: pos3 + quat4 + size3 + vel3 + angvel3 = 16 (去掉 dynamic_mask)
IN_PROJ_DIM = 16


def get_sin_pos_enc(seq_len, d_model):
    """Sinusoid absolute positional encoding. Returns [1, L, d_model]."""
    inv_freq = 1. / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
    pos_seq = torch.arange(seq_len - 1, -1, -1).type_as(inv_freq)
    sinusoid_inp = torch.outer(pos_seq, inv_freq)
    pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
    return pos_emb.unsqueeze(0)  # [1, L, C]


def build_pos_enc(pos_enc, input_len, d_model):
    """Positional Encoding of shape [1, L, D]."""
    if not pos_enc:
        return None
    if pos_enc == 'learnable':
        pos_embedding = nn.Parameter(torch.zeros(1, input_len, d_model))
    elif 'sin' in pos_enc:
        pos_embedding = nn.Parameter(
            get_sin_pos_enc(input_len, d_model), requires_grad=False)
    else:
        raise NotImplementedError(f'unsupported pos enc {pos_enc}')
    return pos_embedding


class SlotRollouter(nn.Module):
    """Transformer encoder that autoregressively rolls out object states.

    动态 num_slots: slots_pe='' (不给 slot 加位置编码, Transformer 对 slot 维度
    permutation-equivariant), forward 时从输入 shape 读取实际积木数。
    """

    def __init__(
        self,
        slot_size,           # 输出维度 = STATE_DIM = 17
        history_len,         # burn-in 帧数
        t_pe='sin',          # temporal P.E.
        slots_pe='',         # slot P.E.; 留空以支持动态 num_slots
        d_model=128,
        num_layers=4,
        num_heads=8,
        ffn_dim=512,
        norm_first=True,
        slotres_scale=1e2,
    ):
        super().__init__()
        self.history_len = history_len
        self.slot_size = slot_size
        self.slotres_scale = slotres_scale

        self.in_proj = nn.Linear(IN_PROJ_DIM, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            norm_first=norm_first,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer, num_layers=num_layers)

        self.enc_t_pe = build_pos_enc(t_pe, history_len, d_model)
        # slots_pe 留空 (动态 num_slots); 若指定则按固定 num_slots 构建
        self.slots_pe_type = slots_pe
        self.out_proj = nn.Linear(d_model, slot_size)

        # 动态维度掩码: 只有这些维度做残差更新, 其余保持初始值不变
        # 0:3 pos, 3:7 quat, 11:14 vel, 14:17 angvel
        feature_mask = torch.zeros(1, 1, slot_size)
        feature_mask[..., 0:7] = 1.0    # pos + quat
        feature_mask[..., 11:17] = 1.0  # vel + angvel
        self.register_buffer('feature_mask', feature_mask)

    def _proj_input(self, in_x):
        """从 17 维状态中取出 16 维喂给 Transformer (去掉 dynamic_mask, idx=10)."""
        return torch.cat([in_x[..., 0:10], in_x[..., 11:17]], dim=-1)  # [.., 16]

    def forward(self, x, pred_len):
        """
        Args:
            x: [B, history_len, num_slots, slot_size] burn-in 状态
            pred_len: int, 自回归预测步数

        Returns:
            [B, pred_len, num_slots, slot_size]
        """
        assert x.shape[1] == self.history_len, \
            f'expected burn-in {self.history_len}, got {x.shape[1]}'
        B = x.shape[0]
        num_slots = x.shape[2]

        # temporal P.E.: [1, T, D] -> [B, T, N, D] -> [B, T*N, D]
        enc_pe = self.enc_t_pe.unsqueeze(2).repeat(B, 1, num_slots, 1).flatten(1, 2)

        x = x.flatten(1, 2)  # [B, T*N, slot_size]
        in_x = x

        pred_out = []
        for _ in range(pred_len):
            proj = self._proj_input(in_x)               # [B, T*N, 16]
            h = self.in_proj(proj) + enc_pe             # [B, T*N, d_model]
            h = self.transformer_encoder(h)             # [B, T*N, d_model]
            # 取最后 num_slots 个 token 预测残差
            res = self.out_proj(h[:, -num_slots:]) / self.slotres_scale  # [B, N, slot_size]
            res = res * self.feature_mask               # 只更新动态维度

            pred_slots = res + in_x[:, -num_slots:]      # 残差加到上一帧

            # 四元数归一化 (与 NFF ODE 输出后处理一致)
            quat = pred_slots[..., 3:7]
            quat_norm = torch.clamp(torch.norm(quat, dim=-1, keepdim=True), min=1e-8)
            pred_slots = pred_slots.clone()
            pred_slots[..., 3:7] = quat / quat_norm

            pred_out.append(pred_slots)
            # 自回归: 丢弃最早一帧, 追加新预测帧
            in_x = torch.cat([in_x[:, num_slots:], pred_slots], dim=1)

        return torch.stack(pred_out, dim=1)  # [B, pred_len, N, slot_size]

    @property
    def device(self):
        return self.in_proj.weight.device


class DynamicsSlotFormer(nn.Module):
    """Transformer-based dynamics model, 接口对齐 NeuralODEModel.

    forward(z0, t, scene_scale) -> [B, timesteps, num_slots, STATE_DIM]
    """

    def __init__(
        self,
        slot_size=STATE_DIM,
        history_len=1,
        t_pe='sin',
        slots_pe='',
        d_model=128,
        num_layers=4,
        num_heads=8,
        ffn_dim=512,
        norm_first=True,
        slotres_scale=1e2,
    ):
        super().__init__()
        self.history_len = history_len
        self.rollouter = SlotRollouter(
            slot_size=slot_size,
            history_len=history_len,
            t_pe=t_pe,
            slots_pe=slots_pe,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            norm_first=norm_first,
            slotres_scale=slotres_scale,
        )

    def forward(self, z0, t, scene_scale=None):
        """
        Args:
            z0: [B, num_slots, STATE_DIM] 初始状态 (第0帧)
            t:  [B, timesteps] 时间索引 (与 NFF 接口一致, 仅用 shape 取 timesteps)
            scene_scale: 收下以对齐接口, SlotFormer 内部不使用 (输入已归一化)

        Returns:
            [B, timesteps, num_slots, STATE_DIM]
        """
        timesteps = t.shape[1]

        # burn-in: 用第0帧重复 history_len 次
        burn_in = z0.unsqueeze(1).repeat(1, self.history_len, 1, 1)  # [B, H, N, D]

        pred_len = timesteps - 1
        if pred_len > 0:
            pred_slots = self.rollouter(burn_in, pred_len)  # [B, pred_len, N, D]
            # 第0帧 (burn-in 最后一帧) + 预测帧
            traj = torch.cat([burn_in[:, -1:], pred_slots], dim=1)
        else:
            traj = burn_in[:, -1:]

        return traj  # [B, timesteps, N, STATE_DIM]


# 自测
if __name__ == "__main__":
    B, N, T = 4, 6, 150
    z0 = torch.randn(B, N, STATE_DIM)
    # 归一化初始四元数
    z0[..., 3:7] = z0[..., 3:7] / z0[..., 3:7].norm(dim=-1, keepdim=True)
    t = torch.linspace(0, (T - 1) / 25.0, T).unsqueeze(0).repeat(B, 1)

    model = DynamicsSlotFormer(
        slot_size=STATE_DIM, history_len=1,
        d_model=128, num_layers=4, num_heads=8, ffn_dim=512,
        slotres_scale=1e2,
    )
    out = model(z0, t)
    print('output shape:', out.shape)  # [4, 150, 6, 17]
    q = out[..., 3:7]
    print('quat norm range:', q.norm(dim=-1).min().item(), q.norm(dim=-1).max().item())
    # 测试动态 num_slots
    z0b = torch.randn(2, 10, STATE_DIM)
    z0b[..., 3:7] = z0b[..., 3:7] / z0b[..., 3:7].norm(dim=-1, keepdim=True)
    tb = torch.linspace(0, 5, 100).unsqueeze(0).repeat(2, 1)
    outb = model(z0b, tb)
    print('dynamic num_slots output:', outb.shape)  # [2, 100, 10, 17]
