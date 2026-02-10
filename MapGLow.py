# -*- coding: utf-8 -*-
"""
Glow-based trajectory model with cross-flow shared *exogenous* context + 2D RoPE spatial attention.

Exogenous context ONLY depends on {map, map_mask, history, labels, agent_types, agent_shape} — never on in_a/out.
No detach hacks needed; per-sample logdet; lane Top-K uses map_mask-safe distances.

Channel convention (C=5):
0:x, 1:y, 2:vx, 3:vy, 4:yaw   (units: meters, m/s, radians)
"""

from math import log, pi
import torch
from torch import nn
from torch.nn import functional as F

# ------------------------- Performance Settings -------------------------
# 启用 TF32 加速（Ampere+ GPU）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# 检测是否支持 Flash Attention
HAS_FLASH_ATTN = hasattr(F, 'scaled_dot_product_attention')

# ------------------------- helpers -------------------------
logabs = lambda x: torch.log(torch.abs(x) + 1e-12)


@torch.jit.script
def _fix_all_pad_mask(mask: torch.Tensor, all_pad_rows: torch.Tensor) -> torch.Tensor:
    """JIT-compiled helper to fix all-padding rows."""
    cleaned_mask = mask.clone()
    cleaned_mask[all_pad_rows] = False
    return cleaned_mask


def ensure_not_all_pad(mask):
    """
    确保 key_padding_mask 没有整行全为 True 的情况。
    如果某行全为 True（全 padding），则将该行全置为 False，并返回标记。
    
    Args:
        mask: [B, S] bool tensor, True = padding (to be ignored)
    
    Returns:
        cleaned_mask: [B, S] 清理后的 mask
        all_pad_rows: [B] bool tensor, True = 该行原本全是 padding
    """
    if mask is None:
        return None, None
    
    # 检测哪些行全是 True（全 padding）
    all_pad_rows = mask.all(dim=-1)  # [B]
    
    # 如果有全 padding 的行，将这些行的 mask 全部置为 False
    # 这样 softmax 不会产生 NaN，但我们需要在后续将这些行的输出置零
    if all_pad_rows.any():
        cleaned_mask = _fix_all_pad_mask(mask, all_pad_rows)
        return cleaned_mask, all_pad_rows
    
    return mask, all_pad_rows


def build_padding_attn_bias(key_padding_mask, num_heads, tgt_len, dtype):
    """
    Convert key_padding_mask to additive attention bias.
    Args:
        key_padding_mask: [N, S] bool, True = padding
        num_heads: int
        tgt_len: int (query length L)
        dtype: float dtype for attention bias
    Returns:
        attn_bias: [N*num_heads, L, S] float, padded keys are -inf
    """
    if key_padding_mask is None:
        return None
    if key_padding_mask.dtype != torch.bool:
        key_padding_mask = key_padding_mask.bool()
    N, S = key_padding_mask.shape
    bias = torch.zeros(N, num_heads, tgt_len, S, device=key_padding_mask.device, dtype=dtype)
    bias = bias.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
    return bias.view(N * num_heads, tgt_len, S)


def gaussian_log_p(x, mean, log_sd, mask=None):
    """
    Compute Gaussian log probability.
    
    Args:
        x: input tensor
        mean: mean of Gaussian
        log_sd: log standard deviation
        mask: optional boolean mask [B, C, T, V] or [B, ...], True = valid (compute log_p)
              If None, all positions are valid.
    
    Returns:
        log_p per sample [B] (summed over valid positions only)
    """
    # Clamp log_sd to prevent numerical instability
    log_sd = torch.clamp(log_sd, min=-10.0, max=10.0)
    
    # Clamp the squared term to prevent overflow
    diff_sq = torch.clamp((x - mean) ** 2, max=1e6)
    var = torch.exp(2 * log_sd).clamp(min=1e-10)
    
    log_p_elem = -0.5 * log(2 * pi) - log_sd - 0.5 * diff_sq / var
    
    if mask is not None:
        # mask: True = valid, False = padding (ignore)
        # Ensure mask has same shape or is broadcastable
        log_p_elem = log_p_elem * mask.float()
    
    return log_p_elem


def gaussian_sample(eps, mean, log_sd):
    # Clamp log_sd to prevent extreme values
    log_sd = torch.clamp(log_sd, min=-10.0, max=10.0)
    return mean + torch.exp(log_sd) * eps


def create_vehicle_mask(data, threshold=1e-6):
    """检测哪些车辆是真实的（非全零padding） data: [B,C,T,V] -> [B,V]"""
    vehicle_sum = data.abs().sum(dim=(1, 2))  # [B, V]
    return vehicle_sum > threshold


def create_timestep_mask(data, padding_value=0.0, threshold=0.5):
    """
    创建精细的时间步 mask。
    
    Args:
        data: [B, C, T, V] 轨迹数据
        padding_value: padding 使用的值（默认 -1.0）
        threshold: 判断为 padding 的通道比例阈值
    
    Returns:
        mask: [B, T, V] bool tensor, True = valid (real data), False = padding
    """
    # Fallback heuristic only.
    # Prefer passing explicit timestep mask from dataloader / wrapper.
    # Here we use xy non-zero criterion, which is usually the most stable for trajectory data.
    if data.size(1) >= 2:
        mask = data[:, :2].abs().sum(dim=1) > 1e-6  # [B, T, V]
    else:
        mask = data.abs().sum(dim=1) > 1e-6
    return mask.bool()


def create_squeezed_mask(timestep_mask, squeeze_factor=2):
    """
    将 timestep mask 转换为 squeeze 后的形状。
    
    Args:
        timestep_mask: [B, T, V] bool, True = valid
        squeeze_factor: squeeze 因子（默认 2）
    
    Returns:
        squeezed_mask: [B, C_squeezed, T_squeezed, V] bool
            其中 C_squeezed = squeeze_factor, T_squeezed = T // squeeze_factor
    """
    B, T, V = timestep_mask.shape
    T_squeezed = T // squeeze_factor
    
    # Reshape: [B, T, V] -> [B, T_squeezed, squeeze_factor, V]
    mask_reshaped = timestep_mask.view(B, T_squeezed, squeeze_factor, V)
    
    # 对于 squeeze 后的每个时间步，只要原始的任一子时间步有效，就认为有效
    # 或者更严格：所有子时间步都有效才算有效
    # 这里使用 any（宽松）：至少一个有效
    squeezed_mask_t = mask_reshaped.any(dim=2)  # [B, T_squeezed, V]
    
    # 扩展到通道维度：[B, C_squeezed, T_squeezed, V]
    # squeeze 后通道变为原来的 squeeze_factor 倍
    squeezed_mask = squeezed_mask_t.unsqueeze(1).expand(B, squeeze_factor, T_squeezed, V)
    
    return squeezed_mask.contiguous()


# ------------------------- ActNorm -------------------------
class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()
        self.loc   = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.logdet = logdet

    @torch.no_grad()
    def initialize(self, x, mask=None):
        # x: [B, C, H, W]
        # mask: [B, C, H, W] or [B, 1, H, W] or [B, H, W], True = valid
        B, C, H, W = x.shape
        
        if mask is not None:
            # Expand mask to [B, C, H, W] if needed
            if mask.dim() == 3:  # [B, H, W]
                mask = mask.unsqueeze(1).expand(B, C, H, W)
            elif mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.expand(B, C, H, W)
            
            # Compute masked mean and var per channel
            # flat: [C, BHW], mask_flat: [C, BHW]
            flat = x.permute(1, 0, 2, 3).contiguous().view(C, -1)
            mask_flat = mask.permute(1, 0, 2, 3).contiguous().view(C, -1).float()
            
            # Masked mean: sum(x * mask) / sum(mask)
            valid_count = mask_flat.sum(dim=1, keepdim=True).clamp(min=1.0)  # [C, 1]
            mean = (flat * mask_flat).sum(dim=1, keepdim=True) / valid_count  # [C, 1]
            
            # Masked var: sum((x - mean)^2 * mask) / sum(mask)
            var = ((flat - mean) ** 2 * mask_flat).sum(dim=1, keepdim=True) / valid_count  # [C, 1]
            
            mean = mean.view(1, C, 1, 1)
            var = var.view(1, C, 1, 1)
        else:
            flat = x.permute(1, 0, 2, 3).contiguous().view(C, -1)           # [C, BHW]
            mean = flat.mean(dim=1).view(1, C, 1, 1)
            var  = flat.var(dim=1, unbiased=False).view(1, C, 1, 1)
        
        std = torch.sqrt(var + 1e-6)
        self.loc.data.copy_(-mean)
        self.scale.data.copy_(1.0 / std)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, C, H, W]
            mask: optional [B, C, H, W] or [B, 1, H, W] bool, True = valid
        """
        B, C, H, W = x.shape
        if not self.initialized:
            self.initialize(x, mask=mask)
            self.initialized.fill_(True)
        y = self.scale * (x + self.loc)
        if not self.logdet:
            return y
        log_abs = logabs(self.scale)  # [1, C, 1, 1]
        
        if mask is not None:
            # 计算每个样本的有效元素数
            # mask: [B, C, H, W] or broadcastable
            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.expand(B, C, H, W)
            elif mask.dim() == 3:  # [B, H, W]
                mask = mask.unsqueeze(1).expand(B, C, H, W)
            # logdet = sum over valid positions
            log_abs_expanded = log_abs.expand(B, C, H, W)
            logdet = (log_abs_expanded * mask.float()).view(B, -1).sum(dim=1)
        else:
            logdet = (H * W * log_abs.view(-1)).sum().unsqueeze(0).repeat(B)
        
        return y, logdet

    def reverse(self, y):
        return y / (self.scale + 1e-12) - self.loc


# ------------------------- Invertible 1x1 conv -------------------------
class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.linalg.qr(weight, mode='reduced')
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, C, H, W]
            mask: optional [B, C, H, W] or [B, 1, H, W] bool, True = valid
        """
        B, C, H, W = x.shape
        out = F.conv2d(x, self.weight)
        det_w = torch.slogdet(self.weight.squeeze().double())[1].float()
        
        if mask is not None:
            # 计算每个样本的有效空间位置数
            if mask.dim() == 4:
                # 取任一通道的 mask（因为 1x1 conv 的 logdet 只与空间位置数有关）
                spatial_mask = mask[:, 0, :, :]  # [B, H, W]
            else:
                spatial_mask = mask  # [B, H, W]
            valid_count = spatial_mask.float().view(B, -1).sum(dim=1)  # [B]
            logdet = det_w * valid_count
        else:
            logdet = (H * W * det_w).unsqueeze(0).repeat(B)
        
        return out, logdet

    def reverse(self, y):
        w_inv = self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        return F.conv2d(y, w_inv)


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.linalg.qr(weight, mode='reduced')
        p, l, u = torch.linalg.lu(q)

        s = torch.diag(u)
        sign_s = torch.sign(s)
        log_s = torch.log(torch.abs(s) + 1e-12)
        u = torch.triu(u, diagonal=1)
        l = torch.tril(l, diagonal=-1)

        self.register_buffer("p", p)
        self.register_buffer("sign_s", sign_s)
        self.register_buffer("l_mask", torch.tril(torch.ones_like(l), diagonal=-1))
        self.register_buffer("u_mask", torch.triu(torch.ones_like(u), diagonal=1))
        self.register_buffer("eye", torch.eye(in_channel))

        self.l = nn.Parameter(l)
        self.u = nn.Parameter(u)
        self.log_s = nn.Parameter(log_s)

    def _get_weight(self):
        l = self.l * self.l_mask + self.eye
        s = torch.diag(self.sign_s * torch.exp(self.log_s))
        u = self.u * self.u_mask + s
        w = self.p @ l @ u
        return w

    def forward(self, x, mask=None):
        B, C, H, W = x.shape
        weight = self._get_weight().unsqueeze(2).unsqueeze(3)
        out = F.conv2d(x, weight)

        det_w = self.log_s.sum()
        if mask is not None:
            if mask.dim() == 4:
                spatial_mask = mask[:, 0, :, :]
            else:
                spatial_mask = mask
            valid_count = spatial_mask.float().view(B, -1).sum(dim=1)
            logdet = det_w * valid_count
        else:
            logdet = (H * W * det_w).unsqueeze(0).repeat(B)

        return out, logdet

    def reverse(self, y):
        weight = self._get_weight()
        w_inv = torch.inverse(weight).unsqueeze(2).unsqueeze(3)
        return F.conv2d(y, w_inv)


# ------------------------- ZeroConv2d -------------------------
class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, (3, 1), padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, x):
        out = F.pad(x, [0, 0, 1, 1], value=0.0)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)
        return out


# ------------------------- Agent Interaction Module -------------------------
class AgentInteractionModule(nn.Module):
    """
    增强agent间交互的模块，包括：
    1. 跨时间步的agent交互（temporal-aware spatial attention）
    2. 相对运动关系建模（relative motion encoding）
    3. 多尺度时空交互（multi-scale spatio-temporal interaction）
    """
    def __init__(self, filter_size=256, num_heads=8, num_layers=2, max_agents=64):
        super().__init__()
        self.filter_size = filter_size
        self.num_heads = num_heads
        
        # 相对位置/速度编码
        self.relative_encoder = nn.Sequential(
            nn.Linear(6, filter_size // 2),  # [dx, dy, dvx, dvy, d_yaw, dist]
            nn.ReLU(inplace=True),
            nn.Linear(filter_size // 2, filter_size)
        )
        
        # 跨agent注意力（考虑相对关系）
        self.cross_agent_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=filter_size, num_heads=num_heads, batch_first=True, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.cross_agent_ln = nn.ModuleList([nn.LayerNorm(filter_size) for _ in range(num_layers)])
        self.cross_agent_ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(filter_size, filter_size*4), nn.GELU(), nn.Linear(filter_size*4, filter_size))
            for _ in range(num_layers)
        ])
        self.cross_agent_ln2 = nn.ModuleList([nn.LayerNorm(filter_size) for _ in range(num_layers)])
        self.rel_bias_proj = nn.Linear(filter_size, num_heads)
        
        # 时空联合交互
        self.spatiotemporal_attn = nn.MultiheadAttention(embed_dim=filter_size, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.st_ln1 = nn.LayerNorm(filter_size)
        self.st_ffn = nn.Sequential(nn.Linear(filter_size, filter_size*4), nn.GELU(), nn.Linear(filter_size*4, filter_size))
        self.st_ln2 = nn.LayerNorm(filter_size)
        
        # 可学习的交互重要性权重
        self.interaction_gate = nn.Sequential(
            nn.Linear(filter_size * 2, filter_size),
            nn.Sigmoid()
        )
    
    def compute_relative_features(self, traj_data, vehicle_mask):
        """
        计算agent之间的相对运动特征
        traj_data: [B, C, T, V] (C=5: x, y, vx, vy, yaw)
        vehicle_mask: [B, V] bool
        Returns: [B, T, V, V, F] 相对特征
        """
        B, C, T, V = traj_data.shape
        device = traj_data.device
        
        # 提取位置、速度、航向
        pos = traj_data[:, :2, :, :]  # [B, 2, T, V]
        vel = traj_data[:, 2:4, :, :] if C > 2 else torch.zeros(B, 2, T, V, device=device)
        yaw = traj_data[:, 4:5, :, :] if C > 4 else torch.zeros(B, 1, T, V, device=device)
        
        # 计算相对位置 [B, T, V, V, 2]
        pos_t = pos.permute(0, 2, 3, 1)  # [B, T, V, 2]
        rel_pos = pos_t.unsqueeze(3) - pos_t.unsqueeze(2)  # [B, T, V, V, 2]
        
        # 计算相对速度 [B, T, V, V, 2]
        vel_t = vel.permute(0, 2, 3, 1)  # [B, T, V, 2]
        rel_vel = vel_t.unsqueeze(3) - vel_t.unsqueeze(2)  # [B, T, V, V, 2]
        
        # 计算相对航向 [B, T, V, V, 1]
        yaw_t = yaw.permute(0, 2, 3, 1)  # [B, T, V, 1]
        rel_yaw = yaw_t.unsqueeze(3) - yaw_t.unsqueeze(2)  # [B, T, V, V, 1]
        rel_yaw = torch.atan2(torch.sin(rel_yaw), torch.cos(rel_yaw))  # 归一化到[-pi, pi]
        
        # 计算距离 [B, T, V, V, 1]
        dist = torch.sqrt((rel_pos ** 2).sum(dim=-1, keepdim=True) + 1e-6)
        
        # 拼接相对特征 [B, T, V, V, 6]
        rel_feat = torch.cat([rel_pos, rel_vel, rel_yaw, dist], dim=-1)
        
        return rel_feat
    
    def forward(self, agent_feat, traj_data, vehicle_mask, timestep_mask=None):
        """
        agent_feat: [B, V, T, F] agent特征
        traj_data: [B, C, T, V] 原始轨迹数据
        vehicle_mask: [B, V] bool
        timestep_mask: [B, T, V] bool
        Returns: [B, V, T, F] 增强后的agent特征
        """
        B, V, T, F = agent_feat.shape
        device = agent_feat.device
        
        # 计算相对运动特征
        rel_feat = self.compute_relative_features(traj_data, vehicle_mask)  # [B, T, V, V, 6]
        rel_encoded = self.relative_encoder(rel_feat)  # [B, T, V, V, F]
        
        # 跨agent注意力（每个时间步）
        agent_out = agent_feat.permute(0, 2, 1, 3).reshape(B * T, V, F)  # [B*T, V, F]
        
        # 创建注意力mask
        if vehicle_mask is not None:
            attn_mask = ~vehicle_mask.unsqueeze(1).expand(B, T, V).reshape(B * T, V)
        else:
            attn_mask = None
        
        attn_mask_cleaned, all_pad = ensure_not_all_pad(attn_mask)
        
        for i in range(len(self.cross_agent_attn)):
            # 将相对特征映射为多头 attention bias
            rel_bias = self.rel_bias_proj(rel_encoded)  # [B, T, V, V, H]
            rel_bias = rel_bias.permute(0, 1, 4, 2, 3).contiguous()  # [B, T, H, V, V]
            rel_bias = 0.1 * torch.tanh(rel_bias)
            rel_bias = rel_bias.view(B * T * self.num_heads, V, V)  # [B*T*H, V, V]

            # Merge key padding into additive attn_mask to avoid dtype mismatch warnings.
            pad_bias = build_padding_attn_bias(
                attn_mask_cleaned, num_heads=self.num_heads, tgt_len=V, dtype=rel_bias.dtype
            )
            combined_attn_mask = rel_bias if pad_bias is None else (rel_bias + pad_bias)
            
            attn_out, _ = self.cross_agent_attn[i](
                query=agent_out, key=agent_out, value=agent_out,
                key_padding_mask=None,
                attn_mask=combined_attn_mask
            )
            if all_pad is not None and all_pad.any():
                attn_out = attn_out.clone()
                attn_out[all_pad] = 0.0
            
            agent_out = self.cross_agent_ln[i](agent_out + attn_out)
            agent_out = self.cross_agent_ln2[i](agent_out + self.cross_agent_ffn[i](agent_out))
        
        agent_out = agent_out.view(B, T, V, F).permute(0, 2, 1, 3)  # [B, V, T, F]
        
        # 时空联合交互（将所有V*T位置一起attention）
        st_input = agent_out.reshape(B, V * T, F)
        
        # 创建时空mask
        if timestep_mask is not None and vehicle_mask is not None:
            st_mask = ~(timestep_mask.permute(0, 2, 1) & vehicle_mask.unsqueeze(-1))  # [B, V, T]
            st_mask = st_mask.reshape(B, V * T)
        elif vehicle_mask is not None:
            st_mask = ~vehicle_mask.unsqueeze(-1).expand(B, V, T).reshape(B, V * T)
        else:
            st_mask = None
        
        st_mask_cleaned, st_all_pad = ensure_not_all_pad(st_mask)
        
        st_out, _ = self.spatiotemporal_attn(
            query=st_input, key=st_input, value=st_input,
            key_padding_mask=st_mask_cleaned
        )
        if st_all_pad is not None and st_all_pad.any():
            st_out = st_out.clone()
            st_out[st_all_pad] = 0.0
        
        st_out = self.st_ln1(st_input + st_out)
        st_out = self.st_ln2(st_out + self.st_ffn(st_out))
        st_out = st_out.view(B, V, T, F)
        
        # 门控融合
        gate = self.interaction_gate(torch.cat([agent_feat, st_out], dim=-1))
        output = agent_feat + gate * (st_out - agent_feat)
        
        return output


# ------------------------- Lane-Aware Attention -------------------------
class LaneAwareAttention(nn.Module):
    """
    增强地图约束的模块，考虑：
    1. Lane方向与agent运动方向的一致性
    2. 多尺度lane特征（点级、段级、全局）
    3. 可微分的lane遵循约束
    """
    def __init__(self, filter_size=256, num_heads=8):
        super().__init__()
        self.filter_size = filter_size
        self.num_heads = num_heads
        
        # 方向一致性编码
        self.direction_encoder = nn.Sequential(
            nn.Linear(4, filter_size // 2),  # [cos_diff, sin_diff, lateral_dist, longitudinal_dist]
            nn.ReLU(inplace=True),
            nn.Linear(filter_size // 2, filter_size)
        )
        
        # Agent-to-lane注意力（带方向bias）
        self.agent_to_lane_attn = nn.MultiheadAttention(
            embed_dim=filter_size, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.direction_bias_proj = nn.Linear(filter_size, num_heads)
        self.a2l_ln1 = nn.LayerNorm(filter_size)
        self.a2l_ffn = nn.Sequential(
            nn.Linear(filter_size, filter_size * 4), nn.GELU(), nn.Linear(filter_size * 4, filter_size)
        )
        self.a2l_ln2 = nn.LayerNorm(filter_size)
    
    def compute_direction_features(self, agent_pos, agent_vel, lane_data, lane_mask):
        """
        计算agent与lane之间的方向关系特征
        agent_pos: [B, V, 2] agent位置
        agent_vel: [B, V, 2] agent速度
        lane_data: [B, L, P, C] lane点数据
        lane_mask: [B, L, P] True=valid
        Returns: [B, V, L, F] 方向关系特征
        """
        B, V, _ = agent_pos.shape
        _, L, P, C = lane_data.shape
        device = agent_pos.device
        
        # 计算lane方向（使用相邻点）
        lane_pts = lane_data[..., :2]  # [B, L, P, 2]
        lane_dir = torch.zeros_like(lane_pts)
        if P > 1:
            lane_dir[:, :, :-1, :] = lane_pts[:, :, 1:, :] - lane_pts[:, :, :-1, :]
        lane_dir_norm = lane_dir / (torch.norm(lane_dir, dim=-1, keepdim=True) + 1e-6)
        
        # 平均lane方向
        valid_mask = lane_mask.unsqueeze(-1).float()  # [B, L, P, 1]
        lane_dir_avg = (lane_dir_norm * valid_mask).sum(dim=2) / (valid_mask.sum(dim=2) + 1e-6)  # [B, L, 2]
        
        # Agent速度方向
        agent_dir = agent_vel / (torch.norm(agent_vel, dim=-1, keepdim=True) + 1e-6)  # [B, V, 2]
        
        # 方向差异（cos和sin）
        # agent_dir: [B, V, 2], lane_dir_avg: [B, L, 2]
        cos_diff = (agent_dir.unsqueeze(2) * lane_dir_avg.unsqueeze(1)).sum(dim=-1)  # [B, V, L]
        
        # 计算垂直分量（cross product in 2D）
        sin_diff = agent_dir[:, :, 0:1] * lane_dir_avg[:, :, 1:2].transpose(1, 2) - \
                   agent_dir[:, :, 1:2] * lane_dir_avg[:, :, 0:1].transpose(1, 2)
        sin_diff = sin_diff.squeeze(-1)  # [B, V, L]
        
        # 计算到lane的横向和纵向距离
        lane_center = (lane_pts * valid_mask).sum(dim=2) / (valid_mask.sum(dim=2) + 1e-6)  # [B, L, 2]
        rel_pos = agent_pos.unsqueeze(2) - lane_center.unsqueeze(1)  # [B, V, L, 2]
        
        # 纵向距离（沿lane方向）
        long_dist = (rel_pos * lane_dir_avg.unsqueeze(1)).sum(dim=-1)  # [B, V, L]
        
        # 横向距离（垂直于lane方向）
        lat_dist = (rel_pos[:, :, :, 0] * lane_dir_avg[:, :, 1].unsqueeze(1) - 
                    rel_pos[:, :, :, 1] * lane_dir_avg[:, :, 0].unsqueeze(1))  # [B, V, L]
        
        # 组合特征
        dir_feat = torch.stack([cos_diff, sin_diff, lat_dist / 10.0, long_dist / 10.0], dim=-1)  # [B, V, L, 4]
        
        return self.direction_encoder(dir_feat)  # [B, V, L, F]
    
    def forward(self, agent_feat, lane_feat, agent_pos, agent_vel, lane_data, lane_mask,
                lane_indices=None, lane_point_feat=None, agent_valid_mask=None):
        """
        agent_feat: [B*V, T, F] agent特征
        lane_feat: [B*V, K, F] lane特征（已选择topK）
        agent_pos: [B, V, 2] agent位置
        agent_vel: [B, V, 2] agent速度
        lane_data: [B, L, P, C] 原始lane数据
        lane_mask: [B, L, P] lane mask
        lane_point_feat: [B, L, P, F] 可选的点级lane特征
        agent_valid_mask: [B, V] agent有效mask
        Returns: enhanced agent_feat [B*V, T, F], compliance_score [B, V]
        """
        BV, T, F = agent_feat.shape
        _, K, _ = lane_feat.shape
        
        # 推断B和V
        B = agent_pos.shape[0]
        V = agent_pos.shape[1]
        
        # Agent-to-lane attention
        lane_pad = (lane_feat.abs().sum(dim=-1) < 1e-6)  # [B*V, K]
        lane_pad_cleaned, all_pad = ensure_not_all_pad(lane_pad)

        lane_feat_for_attn = lane_feat
        attn_bias = None
        if lane_indices is not None and lane_data is not None and lane_mask is not None:
            # Compute direction features over full lanes, then gather selected lane slots.
            dir_feat_full = self.compute_direction_features(agent_pos, agent_vel, lane_data, lane_mask)  # [B,V,L,F]
            idx = lane_indices.clamp(min=0)
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, self.filter_size)  # [B,V,K,F]
            dir_feat_sel = torch.gather(dir_feat_full, 2, gather_idx)  # [B,V,K,F]
            idx_valid = (lane_indices >= 0).unsqueeze(-1).float()
            dir_feat_sel = dir_feat_sel * idx_valid
            dir_feat_sel = dir_feat_sel.reshape(B * V, K, self.filter_size)

            lane_feat_for_attn = lane_feat_for_attn + dir_feat_sel

            # Additive attention bias derived from direction consistency.
            dir_bias = self.direction_bias_proj(dir_feat_sel)  # [B*V,K,H]
            dir_bias = dir_bias.permute(0, 2, 1).contiguous()  # [B*V,H,K]
            dir_bias = 0.1 * torch.tanh(dir_bias)
            attn_bias = dir_bias.unsqueeze(2).expand(B * V, self.num_heads, T, K).reshape(B * V * self.num_heads, T, K)
        
        if all_pad is not None and all_pad.any():
            lane_feat_for_attn = lane_feat_for_attn.clone()
            lane_feat_for_attn[all_pad] = 0.0

        combined_attn_mask = attn_bias
        key_padding_for_attn = lane_pad_cleaned
        if attn_bias is not None:
            pad_bias = build_padding_attn_bias(
                lane_pad_cleaned, num_heads=self.num_heads, tgt_len=T, dtype=attn_bias.dtype
            )
            if pad_bias is not None:
                combined_attn_mask = attn_bias + pad_bias
            key_padding_for_attn = None
        
        attn_out, attn_weights = self.agent_to_lane_attn(
            query=agent_feat, key=lane_feat_for_attn, value=lane_feat_for_attn,
            key_padding_mask=key_padding_for_attn,
            attn_mask=combined_attn_mask
        )
        
        if all_pad is not None and all_pad.any():
            attn_out = attn_out.clone()
            attn_out[all_pad] = 0.0
        
        out = self.a2l_ln1(agent_feat + attn_out)
        out = self.a2l_ln2(out + self.a2l_ffn(out))
        
        return out


# ------------------------- MapFeatureExtractor -------------------------
class MapFeatureExtractor(nn.Module):
    def __init__(
        self,
        filter_size=128,
        num_heads=8,
        max_lanes=64,
        max_points=30,
        map_input_dim=5,
        num_lane_types=21,
        num_traffic_light_states=8,
        lane_type_encoding="embedding",
        lane_type_embed_dim=16,
        lane_tl_embed_dim=8,
    ):
        super().__init__()
        self.filter_size = filter_size
        self.max_lanes = max_lanes
        self.max_points = max_points
        self.map_input_dim = int(map_input_dim)
        self.num_lane_types = int(num_lane_types)
        self.num_traffic_light_states = int(num_traffic_light_states)
        self.lane_type_encoding = str(lane_type_encoding).lower()
        if self.lane_type_encoding not in ("embedding", "onehot"):
            self.lane_type_encoding = "embedding"
        self.lane_type_embed = None
        if self.lane_type_encoding == "embedding":
            self.lane_type_embed = nn.Embedding(self.num_lane_types, int(lane_type_embed_dim))
        self.lane_tl_embed = nn.Embedding(self.num_traffic_light_states, int(lane_tl_embed_dim))

        # point_feat = [x, y, sin(yaw), cos(yaw), dx, dy, lane_type_feat, lane_tl_emb]
        lane_type_feat_dim = int(lane_type_embed_dim) if self.lane_type_encoding == "embedding" else self.num_lane_types
        point_input_dim = 6 + lane_type_feat_dim + int(lane_tl_embed_dim)
        self.point_mlp = nn.Sequential(
            nn.Linear(point_input_dim, filter_size // 2), nn.ReLU(inplace=True), nn.Linear(filter_size // 2, filter_size)
        )
        self.point_pos_emb = nn.Parameter(torch.randn(max_points, filter_size))
        # 使用 fast_path 优化的 attention
        self.lane_self_attn = nn.MultiheadAttention(embed_dim=filter_size, num_heads=num_heads, batch_first=True, dropout=0.0)
        self.pool_q = nn.Parameter(torch.randn(1, filter_size))
        self.lane_pool_attn = nn.MultiheadAttention(embed_dim=filter_size, num_heads=num_heads, batch_first=True, dropout=0.0)
        self.global_lane_attn = nn.MultiheadAttention(embed_dim=filter_size, num_heads=num_heads, batch_first=True, dropout=0.0)
        self.point_ln  = nn.LayerNorm(filter_size)
        self.lane_attn_ln1 = nn.LayerNorm(filter_size)
        self.lane_ffn = nn.Sequential(nn.Linear(filter_size, filter_size*4), nn.ReLU(inplace=True), nn.Linear(filter_size*4, filter_size))
        self.lane_attn_ln2 = nn.LayerNorm(filter_size)
        self.global_attn_ln1 = nn.LayerNorm(filter_size)
        self.global_ffn = nn.Sequential(nn.Linear(filter_size, filter_size*4), nn.ReLU(inplace=True), nn.Linear(filter_size*4, filter_size))
        self.global_attn_ln2 = nn.LayerNorm(filter_size)
        
        # 新增：多层lane交互
        self.lane_inter_attn = nn.MultiheadAttention(embed_dim=filter_size, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.lane_inter_ln1 = nn.LayerNorm(filter_size)
        self.lane_inter_ffn = nn.Sequential(
            nn.Linear(filter_size, filter_size*4), nn.GELU(), nn.Linear(filter_size*4, filter_size)
        )
        self.lane_inter_ln2 = nn.LayerNorm(filter_size)

    def forward(self, map_data, map_mask):
        """
        map_data:
            - embedding mode: [B, L, P, 5] (x, y, yaw, lane_type_idx, lane_tl_state_idx)
            - onehot mode: [B, L, P, 4 + num_lane_types]
              (x, y, yaw, lane_type_onehot..., lane_tl_state_idx)
        map_mask: [B, L, P]  True=real point
        return: [B, L, F]
        """
        B, L, P, C = map_data.shape
        if self.lane_type_encoding == "embedding":
            if C < 5:
                raise ValueError(f"MapFeatureExtractor(embedding) expects map_data last dim >=5, got {C}")
        else:
            expected = 4 + self.num_lane_types
            if C < expected:
                raise ValueError(
                    f"MapFeatureExtractor(onehot) expects map_data last dim >= {expected}, got {C}"
                )
        D = self.filter_size
        device = map_data.device
        
        # map_data: x, y, yaw, discrete semantic ids
        flat_data = map_data.view(B * L, P, C)
        flat_mask = map_mask.view(B * L, P)
        valid = flat_mask.any(dim=1)
        N_valid = int(valid.sum())
        pooled = torch.zeros(B * L, D, device=device, dtype=map_data.dtype)
        
        if N_valid > 0:
            v_data = flat_data[valid]
            v_mask = flat_mask[valid]
            v_xy = v_data[..., :2]
            yaw_rad = v_data[..., 2]
            yaw_sin = torch.sin(yaw_rad).unsqueeze(-1)
            yaw_cos = torch.cos(yaw_rad).unsqueeze(-1)
            if self.lane_type_encoding == "embedding":
                lane_type_idx = v_data[..., 3].round().long().clamp(0, self.num_lane_types - 1)
                lane_tl_idx = v_data[..., 4].round().long().clamp(0, self.num_traffic_light_states - 1)
                lane_type_feat = self.lane_type_embed(lane_type_idx)
            else:
                lane_type_feat = v_data[..., 3:3 + self.num_lane_types]
                lane_tl_idx = v_data[..., 3 + self.num_lane_types].round().long().clamp(
                    0, self.num_traffic_light_states - 1
                )
            lane_tl_emb = self.lane_tl_embed(lane_tl_idx)
            valid_scale = v_mask.unsqueeze(-1).float()
            lane_type_feat = lane_type_feat * valid_scale
            lane_tl_emb = lane_tl_emb * valid_scale

            # 计算相邻点之间的向量，若任一点 padding 则置零
            deltas = torch.zeros_like(v_xy)
            if v_xy.size(1) > 1:
                delta_vals = v_xy[:, 1:, :] - v_xy[:, :-1, :]
                valid_pairs = (v_mask[:, 1:] & v_mask[:, :-1]).unsqueeze(-1).float()
                delta_vals = delta_vals * valid_pairs
                deltas[:, 1:, :] = delta_vals
            point_feat = torch.cat([v_xy, yaw_sin, yaw_cos, deltas, lane_type_feat, lane_tl_emb], dim=-1)
            inp = point_feat.view(N_valid * P, -1)
            v_feat = self.point_mlp(inp).view(N_valid, P, D)
            pos_emb = self.point_pos_emb[:P].unsqueeze(0).to(v_feat.dtype)
            v_feat = self.point_ln(v_feat + pos_emb)
            lane_point_mask = ~v_mask  # [N_valid, P], True = padding
            
            # 确保没有全 padding 的 lane（虽然 valid 已经过滤了，但以防万一）
            lane_point_mask_cleaned, all_pad_lanes = ensure_not_all_pad(lane_point_mask)
            if all_pad_lanes is not None and all_pad_lanes.any():
                v_feat_for_attn = v_feat.clone()
                v_feat_for_attn[all_pad_lanes] = 0.0
            else:
                v_feat_for_attn = v_feat
            
            attn_out, _ = self.lane_self_attn(query=v_feat_for_attn, key=v_feat_for_attn, value=v_feat_for_attn, key_padding_mask=lane_point_mask_cleaned)
            
            # 对于全 padding 的 lane，将输出置零
            if all_pad_lanes is not None and all_pad_lanes.any():
                attn_out = attn_out.clone()
                attn_out[all_pad_lanes] = 0.0
            
            x = self.lane_attn_ln1(v_feat + attn_out)
            y = self.lane_ffn(x)
            attn_out = self.lane_attn_ln2(x + y)
            q = self.pool_q.unsqueeze(0).expand(N_valid, 1, -1)
            
            pooled_v, _ = self.lane_pool_attn(query=q, key=attn_out, value=attn_out, key_padding_mask=lane_point_mask_cleaned)
            
            # 对于全 padding 的 lane，将 pooled 输出置零
            if all_pad_lanes is not None and all_pad_lanes.any():
                pooled_v = pooled_v.clone()
                pooled_v[all_pad_lanes] = 0.0
            
            pooled[valid] = pooled_v.squeeze(1).to(pooled.dtype)
        lane_feats = pooled.view(B, L, D)
        lane_pad = ~map_mask.any(dim=2)  # [B, L], True = padding
        
        # 确保没有全 padding 的行
        lane_pad_cleaned, all_pad_batches = ensure_not_all_pad(lane_pad)
        # 对于全 padding 的 batch，将 lane 特征置零
        if all_pad_batches is not None and all_pad_batches.any():
            lane_feats = lane_feats.clone()
            lane_feats[all_pad_batches] = 0.0
        
        global_out, _ = self.global_lane_attn(query=lane_feats, key=lane_feats, value=lane_feats, key_padding_mask=lane_pad_cleaned)
        
        # 对于全 padding 的 batch，将输出置零
        if all_pad_batches is not None and all_pad_batches.any():
            global_out = global_out.clone()
            global_out[all_pad_batches] = 0.0
        
        xg = self.global_attn_ln1(lane_feats + global_out)
        yg = self.global_ffn(xg)
        global_out = self.global_attn_ln2(xg + yg)
        
        # 新增：额外一层lane间交互，增强lane连接性建模
        inter_out, _ = self.lane_inter_attn(query=global_out, key=global_out, value=global_out, key_padding_mask=lane_pad_cleaned)
        if all_pad_batches is not None and all_pad_batches.any():
            inter_out = inter_out.clone()
            inter_out[all_pad_batches] = 0.0
        xi = self.lane_inter_ln1(global_out + inter_out)
        yi = self.lane_inter_ffn(xi)
        global_out = self.lane_inter_ln2(xi + yi)
        
        return global_out


class TrafficLightFeatureExtractor(nn.Module):
    def __init__(
        self,
        filter_size=128,
        num_heads=8,
        input_dim=3,
        max_lights=64,
        num_traffic_light_states=8,
        tl_state_embed_dim=8,
    ):
        super().__init__()
        self.filter_size = filter_size
        self.max_lights = max_lights
        self.input_dim = int(input_dim)
        self.num_traffic_light_states = int(num_traffic_light_states)
        self.tl_state_embed = nn.Embedding(self.num_traffic_light_states, int(tl_state_embed_dim))

        self.input_mlp = nn.Sequential(
            nn.Linear(2 + int(tl_state_embed_dim), filter_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(filter_size // 2, filter_size),
        )
        self.pos_emb = nn.Parameter(torch.randn(max_lights, filter_size))
        self.input_ln = nn.LayerNorm(filter_size)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=filter_size, num_heads=num_heads, batch_first=True, dropout=0.0
        )
        self.ln1 = nn.LayerNorm(filter_size)
        self.ffn = nn.Sequential(
            nn.Linear(filter_size, filter_size * 4),
            nn.ReLU(inplace=True),
            nn.Linear(filter_size * 4, filter_size),
        )
        self.ln2 = nn.LayerNorm(filter_size)

    def forward(self, traffic_light_data, traffic_light_mask):
        """
        traffic_light_data: [B, TL, 3] (x, y, state_idx)
        traffic_light_mask: [B, TL] bool, True=valid
        return: [B, TL, F]
        """
        B, TL, C = traffic_light_data.shape
        D = self.filter_size
        if TL == 0:
            return torch.zeros(B, 0, D, device=traffic_light_data.device, dtype=traffic_light_data.dtype)

        tl_xy = traffic_light_data[..., :2]
        tl_state_idx = traffic_light_data[..., 2].round().long().clamp(0, self.num_traffic_light_states - 1)
        tl_state_emb = self.tl_state_embed(tl_state_idx)
        tl_state_emb = tl_state_emb * traffic_light_mask.unsqueeze(-1).float()
        tl_feat = torch.cat([tl_xy, tl_state_emb], dim=-1)
        x = self.input_mlp(tl_feat.view(B * TL, -1)).view(B, TL, D)

        if TL <= self.max_lights:
            pos = self.pos_emb[:TL]
        else:
            extra = torch.zeros(TL - self.max_lights, D, device=x.device, dtype=x.dtype)
            pos = torch.cat([self.pos_emb, extra], dim=0)
        x = self.input_ln(x + pos.unsqueeze(0).to(x.dtype))

        tl_pad = ~traffic_light_mask.bool()
        tl_pad_cleaned, all_pad_rows = ensure_not_all_pad(tl_pad)

        if all_pad_rows is not None and all_pad_rows.any():
            x_for_attn = x.clone()
            x_for_attn[all_pad_rows] = 0.0
        else:
            x_for_attn = x

        attn_out, _ = self.self_attn(
            query=x_for_attn, key=x_for_attn, value=x_for_attn, key_padding_mask=tl_pad_cleaned
        )
        if all_pad_rows is not None and all_pad_rows.any():
            attn_out = attn_out.clone()
            attn_out[all_pad_rows] = 0.0

        h1 = self.ln1(x + attn_out)
        h2 = self.ffn(h1)
        out = self.ln2(h1 + h2)

        if all_pad_rows is not None and all_pad_rows.any():
            out = out.clone()
            out[all_pad_rows] = 0.0
        return out


# ------------------------- History-Map Cross Attention -------------------------
class HistoryMapCrossAttention(nn.Module):
    def __init__(self, filter_size, num_heads=8):
        super().__init__()
        self.history_to_map_attn = nn.MultiheadAttention(embed_dim=filter_size, num_heads=num_heads, batch_first=True, dropout=0.0)
        self.map_to_history_attn = nn.MultiheadAttention(embed_dim=filter_size, num_heads=num_heads, batch_first=True, dropout=0.0)
        self.history_ln1 = nn.LayerNorm(filter_size)
        self.history_ffn = nn.Sequential(nn.Linear(filter_size, filter_size*4), nn.ReLU(inplace=True), nn.Linear(filter_size*4, filter_size))
        self.history_ln2 = nn.LayerNorm(filter_size)
        self.map_ln1 = nn.LayerNorm(filter_size)
        self.map_ffn = nn.Sequential(nn.Linear(filter_size, filter_size*4), nn.ReLU(inplace=True), nn.Linear(filter_size*4, filter_size))
        self.map_ln2 = nn.LayerNorm(filter_size)

    def forward(self, history_feat, map_feat, map_mask=None, history_mask=None):
        """
        Args:
            history_feat: [B*V, T_hist, F] history features
            map_feat: [B*V, K, F] map/lane features
            map_mask: [B*V, K] bool, True = padding (to be ignored)
            history_mask: [B*V, T_hist] bool, True = padding (to be ignored)
        """
        # 清洗 map_mask，确保没有全 padding 的行
        map_mask_cleaned, all_pad_map = ensure_not_all_pad(map_mask)
        # 对于 map 全 padding 的行，将 map 特征置零
        if all_pad_map is not None and all_pad_map.any():
            map_feat = map_feat.clone()
            map_feat[all_pad_map] = 0.0
        
        # 清洗 history_mask，确保没有全 padding 的行
        history_mask_cleaned, all_pad_hist = ensure_not_all_pad(history_mask)
        # 对于 history 全 padding 的行，将 history 特征置零
        if all_pad_hist is not None and all_pad_hist.any():
            history_feat = history_feat.clone()
            history_feat[all_pad_hist] = 0.0
        
        # history -> map (query=history, key/value=map)
        hist_to_map, _ = self.history_to_map_attn(query=history_feat, key=map_feat, value=map_feat, key_padding_mask=map_mask_cleaned)
        # 对于 map 全 padding 的行，将注意力输出置零
        if all_pad_map is not None and all_pad_map.any():
            hist_to_map = hist_to_map.clone()
            hist_to_map[all_pad_map] = 0.0
        h1 = self.history_ln1(history_feat + hist_to_map)
        h2 = self.history_ffn(h1)
        enhanced_history = self.history_ln2(h1 + h2)
        
        # map -> history (query=map, key/value=history)
        map_to_hist, _ = self.map_to_history_attn(query=map_feat, key=history_feat, value=history_feat, key_padding_mask=history_mask_cleaned)
        # 对于 history 全 padding 的行，将注意力输出置零
        if all_pad_hist is not None and all_pad_hist.any():
            map_to_hist = map_to_hist.clone()
            map_to_hist[all_pad_hist] = 0.0
        m1 = self.map_ln1(map_feat + map_to_hist)
        m2 = self.map_ffn(m1)
        enhanced_map = self.map_ln2(m1 + m2)
        return enhanced_history, enhanced_map


# ------------------------- 2D RoPE spatial attention -------------------------
class RoPE2DSpatialAttention(nn.Module):
    """
    Scene-centric 2D RoPE for spatial attention:
    - Inputs: x_feats [BT, V, F], coords [BT, V, 2], key_padding_mask [BT, V]
    - Applies continuous 2D rotary embedding to Q/K using centered (x,y).
    - Learnable meters->index scale for robustness to scene size.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 rope_base: float = 10000.0, init_scene_diam_m: float = 3.0,
                 learnable_scale: bool = True, dropout_p: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE pairing"
        self.xy_chunk = self.head_dim // 2
        assert self.xy_chunk % 2 == 0, "half head_dim must be even"

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out = nn.Linear(embed_dim, embed_dim, bias=True)

        inv_idx = torch.arange(0, self.xy_chunk, 2).float() / max(1, self.xy_chunk)
        inv_freq = 1.0 / (rope_base ** inv_idx)
        self.register_buffer("inv_freq_x", inv_freq, persistent=False)
        self.register_buffer("inv_freq_y", inv_freq.clone(), persistent=False)

        scale0 = 2 * torch.pi / max(1.0, init_scene_diam_m)
        if learnable_scale:
            self.m2idx_x = nn.Parameter(torch.tensor(float(scale0)))
            self.m2idx_y = nn.Parameter(torch.tensor(float(scale0)))
        else:
            self.register_buffer("m2idx_x", torch.tensor(float(scale0)), persistent=False)
            self.register_buffer("m2idx_y", torch.tensor(float(scale0)), persistent=False)

        self.dropout_p = dropout_p

    @staticmethod
    def _rope_rotate(x, cos, sin):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        xr = x1 * cos - x2 * sin
        xi = x1 * sin + x2 * cos
        return torch.stack([xr, xi], dim=-1).flatten(-2)

    def _apply_2d_rope(self, q, k, coords):
        """
        q,k:    [BT, V, H, Dh]
        coords: [BT, V, 2]
        """
        BT, V, H, Dh = q.shape
        xy = coords - coords.mean(dim=1, keepdim=True)
        x = xy[..., 0]
        y = xy[..., 1]

        qx, qy = torch.split(q, [self.xy_chunk, self.xy_chunk], dim=-1)
        kx, ky = torch.split(k, [self.xy_chunk, self.xy_chunk], dim=-1)

        theta_x = (x * self.m2idx_x).unsqueeze(-1).unsqueeze(-1) * self.inv_freq_x.view(1,1,1,-1)
        theta_y = (y * self.m2idx_y).unsqueeze(-1).unsqueeze(-1) * self.inv_freq_y.view(1,1,1,-1)
        cos_x, sin_x = torch.cos(theta_x), torch.sin(theta_x)
        cos_y, sin_y = torch.cos(theta_y), torch.sin(theta_y)

        cos_x = cos_x.expand(BT, V, H, -1); sin_x = sin_x.expand_as(cos_x)
        cos_y = cos_y.expand(BT, V, H, -1); sin_y = sin_y.expand_as(cos_y)

        qx = self._rope_rotate(qx, cos_x, sin_x)
        kx = self._rope_rotate(kx, cos_x, sin_x)
        qy = self._rope_rotate(qy, cos_y, sin_y)
        ky = self._rope_rotate(ky, cos_y, sin_y)

        q = torch.cat([qx, qy], dim=-1)
        k = torch.cat([kx, ky], dim=-1)
        return q, k

    def _sdpa(self, q, k, v, attn_mask):
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=False
            )
        Dh = q.size(-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = torch.softmax(attn, dim=-1)
        if self.dropout_p and self.training:
            attn = torch.dropout(attn, p=self.dropout_p, train=True)
        out = torch.matmul(attn, v)
        return out

    def forward(self, x_feats, coords, key_padding_mask=None):
        """
        x_feats: [BT, V, D]
        coords:  [BT, V, 2]
        key_padding_mask: [BT, V] True=padding
        """
        BT, V, D = x_feats.shape
        H, Dh = self.num_heads, self.head_dim
        
        # 确保没有全 padding 的行
        all_pad_rows = None
        if key_padding_mask is not None:
            key_padding_mask, all_pad_rows = ensure_not_all_pad(key_padding_mask)
            # 对于全 padding 的行，将特征和坐标置零
            if all_pad_rows is not None and all_pad_rows.any():
                x_feats = x_feats.clone()
                coords = coords.clone()
                x_feats[all_pad_rows] = 0.0
                coords[all_pad_rows] = 0.0
        
        qkv = self.qkv(x_feats)      # [BT, V, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(BT, V, H, Dh)
        k = k.view(BT, V, H, Dh)
        v = v.view(BT, V, H, Dh)
        q, k = self._apply_2d_rope(q, k, coords)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask.view(BT, 1, 1, V)  # [BT,1,1,V]
            attn_mask = torch.zeros(BT, H, V, V, device=x_feats.device, dtype=torch.float32)
            attn_mask = attn_mask.masked_fill(mask, float("-inf"))

        out = self._sdpa(q, k, v, attn_mask)  # [BT, H, V, Dh]
        out = out.permute(0, 2, 1, 3).contiguous().view(BT, V, D)
        out = self.out(out)
        
        # 对于全 padding 的行，将输出置零
        if all_pad_rows is not None and all_pad_rows.any():
            out = out.clone()
            out[all_pad_rows] = 0.0
        
        return out


# ------------------------- Temporal Transformer -------------------------
class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, filter_size, num_heads=8, num_layers=2, max_seq_len=80, causal=True):
        super().__init__()
        self.filter_size = filter_size
        self.max_seq_len = max_seq_len
        self.causal = causal
        self.num_layers = num_layers
        self.input_proj = nn.Linear(input_dim, filter_size)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, filter_size))
        self.temporal_attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=filter_size, num_heads=num_heads, batch_first=True, dropout=0.0)
            for _ in range(num_layers)
        ])
        self.temporal_ln1_list = nn.ModuleList([nn.LayerNorm(filter_size) for _ in range(num_layers)])
        self.temporal_ffn_list = nn.ModuleList([
            nn.Sequential(nn.Linear(filter_size, filter_size*4), nn.ReLU(inplace=True), nn.Linear(filter_size*4, filter_size))
            for _ in range(num_layers)
        ])
        self.temporal_ln2_list = nn.ModuleList([nn.LayerNorm(filter_size) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(filter_size)
        if self.causal:
            self.register_buffer("causal_mask", torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool())

    def forward(self, x, timestep_mask=None):
        """
        Args:
            x: [B, C, T, V] input tensor
            timestep_mask: [B, T, V] bool, True = valid, False = padding
                          If provided, padding timesteps will be ignored in attention.
        """
        B, C, T, V = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * V, T, C)  # [B*V, T, C]
        x = self.input_proj(x)  # [B*V, T, F]
        pos_emb = self.pos_encoding[:T].unsqueeze(0).to(x.dtype)
        x = x + pos_emb
        
        # Prepare attention mask (causal)
        attn_mask = self.causal_mask[:T, :T] if self.causal else None
        
        # Prepare key_padding_mask for timesteps
        # key_padding_mask: [B*V, T], True = padding (to be ignored)
        key_padding_mask = None
        all_pad_seqs = None
        valid_timestep_mask = None
        if timestep_mask is not None:
            # timestep_mask: [B, T, V] (True = valid) -> key_padding_mask: [B*V, T] (True = padding)
            # Transpose to [B, V, T] then reshape to [B*V, T]
            valid_timestep_mask = timestep_mask.permute(0, 2, 1).reshape(B * V, T).bool()
            key_padding_mask = ~valid_timestep_mask

            # For causal attention, leading padded queries can have no valid keys.
            # Unmask leading padded keys so each query always has at least one valid key.
            if self.causal:
                has_valid_prefix = valid_timestep_mask.long().cumsum(dim=-1) > 0
                key_padding_mask = key_padding_mask & has_valid_prefix

            # 确保没有全 padding 的行，避免 softmax 产生 NaN
            key_padding_mask, all_pad_seqs = ensure_not_all_pad(key_padding_mask)

            # Zero invalid timesteps so unmasked padded keys stay numerically safe.
            x = x.masked_fill((~valid_timestep_mask).unsqueeze(-1), 0.0)
            # 对于全 padding 的序列，将输入置零
            if all_pad_seqs is not None and all_pad_seqs.any():
                x = x.clone()
                x[all_pad_seqs] = 0.0
        
        for i in range(self.num_layers):
            temporal_out, _ = self.temporal_attentions[i](
                query=x, key=x, value=x, 
                attn_mask=attn_mask, 
                key_padding_mask=key_padding_mask
            )
            temporal_out = torch.nan_to_num(temporal_out, nan=0.0, posinf=0.0, neginf=0.0)
            # 对于全 padding 的序列，将注意力输出置零
            if all_pad_seqs is not None and all_pad_seqs.any():
                temporal_out = temporal_out.clone()
                temporal_out[all_pad_seqs] = 0.0
            res_t = x + temporal_out
            t1 = self.temporal_ln1_list[i](res_t)
            t2 = self.temporal_ffn_list[i](t1)
            x = self.temporal_ln2_list[i](t1 + t2)
            if valid_timestep_mask is not None:
                x = x.masked_fill((~valid_timestep_mask).unsqueeze(-1), 0.0)
        x = self.layer_norm(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x.reshape(B, V, T, self.filter_size)
        return x


# ------------------------- Unified Spatio-Temporal Extractor (2D RoPE spatial) -------------------------
class UnifiedSpatioTemporalExtractor(nn.Module):
    def __init__(self, input_dim, filter_size, num_heads=8, num_layers=2, max_seq_len=80, causal=False,
                 max_agents=64, pos_xy_channels=(0, 1), yaw_channel=4, rope_init_scene_diam_m=80.0):
        super().__init__()
        self.filter_size = filter_size
        self.max_agents = max_agents
        self.pos_xy_channels = pos_xy_channels
        self.yaw_channel = yaw_channel
        self.temporal_transformer = TemporalTransformer(
            input_dim=input_dim, filter_size=filter_size, num_heads=num_heads, num_layers=num_layers,
            max_seq_len=max_seq_len, causal=causal
        )
        self.input_proj = nn.Linear(input_dim, filter_size)
        self.spatial_attn_rope = RoPE2DSpatialAttention(
            embed_dim=filter_size, num_heads=num_heads, rope_base=10000.0,
            init_scene_diam_m=rope_init_scene_diam_m, learnable_scale=True, dropout_p=0.0
        )
        self.spatial_ln1 = nn.LayerNorm(filter_size)
        self.spatial_ffn = nn.Sequential(nn.Linear(filter_size, filter_size*4), nn.ReLU(inplace=True), nn.Linear(filter_size*4, filter_size))
        self.spatial_ln2 = nn.LayerNorm(filter_size)
        self.fusion_layer = nn.Linear(filter_size * 2, filter_size)

    def forward(self, x, vehicle_mask=None, timestep_mask=None):
        """
        Args:
            x: [B, C, T, V] input tensor
            vehicle_mask: [B, V] bool, True = valid vehicle
            timestep_mask: [B, T, V] bool, True = valid timestep
        """
        B, C, T, V = x.shape
        if vehicle_mask is None:
            vehicle_mask = (x.abs().sum(dim=(1, 2)) > 1e-6)
        # temporal (with timestep mask)
        temporal_feat = self.temporal_transformer(x, timestep_mask=timestep_mask)  # [B,V,T,F]
        # spatial per-time
        spatial_input = x.permute(0, 2, 3, 1)
        spatial_input = self.input_proj(spatial_input).reshape(B * T, V, -1)  # [BT,V,F]
        
        # Combine vehicle_mask with timestep_mask for spatial attention
        # vehicle_mask: [B, V] -> expand to [B, T, V] -> [B*T, V]
        # timestep_mask: [B, T, V] (True = valid)
        # Final spatial_mask: True = padding (to be ignored)
        if timestep_mask is not None:
            # Combine: position is padding if vehicle is padding OR timestep is padding
            combined_mask = vehicle_mask.unsqueeze(1).expand(B, T, V) & timestep_mask  # True = valid
            spatial_mask = ~combined_mask.reshape(B * T, V)  # True = padding
        else:
            spatial_mask = ~vehicle_mask.unsqueeze(1).expand(B, T, V).reshape(B * T, V)
        
        # coords from x (pos channels 0,1)
        coords_bt_v2 = x[:, (0, 1), :, :].permute(0, 2, 3, 1).reshape(B * T, V, 2)
        spatial_feat = self.spatial_attn_rope(spatial_input, coords_bt_v2, key_padding_mask=spatial_mask)
        res_s = spatial_input + spatial_feat
        s1 = self.spatial_ln1(res_s)
        s2 = self.spatial_ffn(s1)
        spatial_feat = self.spatial_ln2(s1 + s2)
        spatial_feat = spatial_feat.reshape(B, T, V, -1).permute(0, 2, 1, 3)  # [B,V,T,F]
        fused = torch.cat([temporal_feat, spatial_feat], dim=-1)
        output = self.fusion_layer(fused)  # [B,V,T,F]
        vehicle_mask_expanded = vehicle_mask.unsqueeze(-1).unsqueeze(-1)
        output = output * vehicle_mask_expanded.float()
        if timestep_mask is not None:
            timestep_mask_expanded = timestep_mask.permute(0, 2, 1).unsqueeze(-1)
            output = output * timestep_mask_expanded.float()
        output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        return output


# ------------------------- ZeroLinear -------------------------
class ZeroLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.0)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


# ------------------------- Context encoder (exogenous, shared across flows) -------------------------
class ContextEncoder(nn.Module):
    """
    Builds context strictly from exogenous inputs:
      - map lane features (shared across all agents)
      - history features
      - anchor-state token from last valid history timestep (NEW)
      - label / agent-type embeddings
      - agent shape embeddings (length/width/height)
      - agent interaction features (NEW)
      - lane-aware attention (NEW)
    Returns kv and kv_mask for global attention.
    """
    def __init__(
        self,
        filter_size=256,
        num_heads=8,
        history_input_dim=5,
        topk_lanes=16,
        lane_selection_mode="topk",
        hybrid_global_lanes=16,
        max_points=30,
        map_input_dim=5,
        traffic_light_input_dim=3,
        num_lane_types=21,
        num_traffic_light_states=8,
        lane_type_encoding="embedding",
        lane_type_embed_dim=16,
        lane_tl_embed_dim=8,
        tl_state_embed_dim=8,
    ):
        super().__init__()
        self.filter_size = filter_size
        self.topk_lanes = int(topk_lanes)
        self.lane_selection_mode = str(lane_selection_mode).lower()
        if self.lane_selection_mode not in ("topk", "all", "hybrid"):
            self.lane_selection_mode = "topk"
        self.hybrid_global_lanes = int(hybrid_global_lanes)
        self.lane_type_encoding = str(lane_type_encoding).lower()
        self.num_lane_types = int(num_lane_types)
        self.history_extractor = UnifiedSpatioTemporalExtractor(
            input_dim=history_input_dim, filter_size=filter_size,
            num_heads=num_heads, num_layers=1, max_seq_len=30, causal=True
        )
        self.history_fusion = nn.Sequential(nn.Linear(filter_size, filter_size), nn.ReLU(inplace=True))
        self.map_feature_extractor = MapFeatureExtractor(
            filter_size=filter_size,
            num_heads=num_heads,
            max_points=max_points,
            map_input_dim=map_input_dim,
            num_lane_types=num_lane_types,
            num_traffic_light_states=num_traffic_light_states,
            lane_type_encoding=lane_type_encoding,
            lane_type_embed_dim=lane_type_embed_dim,
            lane_tl_embed_dim=lane_tl_embed_dim,
        )
        self.traffic_light_feature_extractor = TrafficLightFeatureExtractor(
            filter_size=filter_size,
            num_heads=num_heads,
            input_dim=traffic_light_input_dim,
            num_traffic_light_states=num_traffic_light_states,
            tl_state_embed_dim=tl_state_embed_dim,
        )
        self.label_embed = nn.Embedding(10, filter_size, padding_idx=0)
        self.agent_types_embed = nn.Embedding(10, filter_size, padding_idx=0)
        self.agent_shape_proj = nn.Sequential(
            nn.Linear(3, filter_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(filter_size // 2, filter_size),
        )
        # Rich anchor token from last valid history state:
        # [x, y, vx, vy, speed, cos(yaw), sin(yaw), dist, vel_dir_x, vel_dir_y, valid]
        self.agent_anchor_proj = nn.Sequential(
            nn.Linear(11, filter_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(filter_size // 2, filter_size),
        )
        self.history_map_cross_attn = HistoryMapCrossAttention(filter_size=filter_size, num_heads=num_heads)
        
        # 新增：Agent交互模块
        self.agent_interaction = AgentInteractionModule(filter_size=filter_size, num_heads=num_heads, num_layers=2)
        
        # 新增：Lane-aware attention
        self.lane_aware_attn = LaneAwareAttention(filter_size=filter_size, num_heads=num_heads)
        

    def _extract_agent_anchor(self, history_data, history_timestep_mask, history_vehicle_mask, agent_valid_mask):
        """Compute last valid history position/velocity per agent for lane-related conditioning."""
        B, V = agent_valid_mask.shape
        if history_data is None:
            device = agent_valid_mask.device
            anchor = torch.zeros(B, V, 2, device=device, dtype=torch.float32)
            anchor_vel = torch.zeros(B, V, 2, device=device, dtype=torch.float32)
            anchor_yaw = torch.zeros(B, V, 1, device=device, dtype=torch.float32)
            # Default to all valid if history is missing, assuming agents are at (0,0) or relevant
            anchor_valid = agent_valid_mask.clone()
            return anchor, anchor_vel, anchor_yaw, anchor_valid

        device = history_data.device
        dtype = history_data.dtype
        anchor = torch.zeros(B, V, 2, device=device, dtype=dtype)
        anchor_vel = torch.zeros(B, V, 2, device=device, dtype=dtype)
        anchor_yaw = torch.zeros(B, V, 1, device=device, dtype=dtype)
        anchor_valid = torch.zeros(B, V, dtype=torch.bool, device=device)

        if history_timestep_mask is None:
            history_timestep_mask = create_timestep_mask(history_data, padding_value=0.0)  # [B, T_hist, V]
        agent_time_mask = history_timestep_mask.permute(0, 2, 1).bool()  # [B, V, T_hist]
        has_valid = agent_time_mask.any(dim=2)
        if history_vehicle_mask is not None:
            has_valid = has_valid & history_vehicle_mask.to(device)

        time_range = torch.arange(history_data.shape[2], device=device).view(1, 1, -1)
        weighted = torch.where(agent_time_mask, time_range, torch.full_like(time_range, -1))
        last_idx = weighted.max(dim=2).values.clamp(min=0).long()  # [B, V]

        pos = history_data[:, :2, :, :].permute(0, 3, 2, 1)  # [B, V, T_hist, 2]
        pos_flat = pos.reshape(B * V, history_data.shape[2], 2)
        gather_idx = last_idx.reshape(B * V, 1, 1).expand(-1, 1, 2)
        gathered = torch.gather(pos_flat, 1, gather_idx).squeeze(1).view(B, V, 2)
        anchor = torch.where(has_valid.unsqueeze(-1), gathered, anchor)
        if history_data.shape[1] >= 4:
            vel = history_data[:, 2:4, :, :].permute(0, 3, 2, 1)  # [B, V, T_hist, 2]
            vel_flat = vel.reshape(B * V, history_data.shape[2], 2)
            gathered_vel = torch.gather(vel_flat, 1, gather_idx).squeeze(1).view(B, V, 2)
            anchor_vel = torch.where(has_valid.unsqueeze(-1), gathered_vel, anchor_vel)
        if history_data.shape[1] >= 5:
            yaw = history_data[:, 4:5, :, :].permute(0, 3, 2, 1)  # [B, V, T_hist, 1]
            yaw_flat = yaw.reshape(B * V, history_data.shape[2], 1)
            gathered_yaw = torch.gather(yaw_flat, 1, gather_idx[..., :1]).squeeze(1).view(B, V, 1)
            anchor_yaw = torch.where(has_valid.unsqueeze(-1), gathered_yaw, anchor_yaw)
        anchor_valid = has_valid & agent_valid_mask
        return anchor, anchor_vel, anchor_yaw, anchor_valid

    def _lane_type_per_lane(self, map_data, map_mask):
        """
        Extract one lane type id per lane from map_data.
        Returns:
            lane_type_idx: [B, L] long
            lane_valid: [B, L] bool
        """
        lane_valid = map_mask.any(dim=-1)  # [B, L]
        B, L, P, _ = map_data.shape
        first_valid_idx = map_mask.long().argmax(dim=-1)  # [B, L]

        if self.lane_type_encoding == "embedding":
            lane_type_pts = map_data[..., 3].round().long().clamp(0, max(self.num_lane_types - 1, 0))
            lane_type_idx = torch.gather(lane_type_pts, 2, first_valid_idx.unsqueeze(-1)).squeeze(-1)  # [B, L]
        else:
            c0 = 3
            c1 = c0 + self.num_lane_types
            lane_type_onehot = map_data[..., c0:c1]
            gather_idx = first_valid_idx.unsqueeze(-1).unsqueeze(-1).expand(B, L, 1, self.num_lane_types)
            lane_type_first = torch.gather(lane_type_onehot, 2, gather_idx).squeeze(2)  # [B, L, C]
            lane_type_idx = lane_type_first.argmax(dim=-1)

        lane_type_idx = lane_type_idx.masked_fill(~lane_valid, 0)
        return lane_type_idx, lane_valid

    def _build_lane_center_mask(self, map_data, map_mask):
        """
        Keep only drivable lane-center types for lane candidate selection.
        Waymo roadgraph lane-center types: 1, 2, 3.
        """
        lane_type_idx, lane_valid = self._lane_type_per_lane(map_data, map_mask)
        lane_center = (lane_type_idx == 1) | (lane_type_idx == 2) | (lane_type_idx == 3)
        keep = lane_valid & lane_center
        # Fallback: if a scene has no lane-center lane, keep all valid lanes.
        no_keep = ~keep.any(dim=-1)
        if no_keep.any():
            keep = keep.clone()
            keep[no_keep] = lane_valid[no_keep]
        return keep

    def _select_topk_lanes(self, lane_feat_shared, map_data, map_mask,
                           agent_xy, agent_valid_mask, anchor_valid_mask, lane_keep_mask=None, return_indices=False):
        """Select top-K nearest lanes per agent based on point-wise distance."""
        if lane_feat_shared is None or map_data is None or map_mask is None:
            if return_indices:
                return None, None, None, None
            return None, None
        B, L, _ = lane_feat_shared.shape
        if L == 0 or self.topk_lanes <= 0:
            if return_indices:
                return None, None, None, None
            return None, None

        coords = map_data[..., :2]  # [B, L, P, 2]
        lane_valid_pts = map_mask
        V = agent_xy.shape[1]

        lane_pts = coords.unsqueeze(1)  # [B, 1, L, P, 2]
        agents = agent_xy.unsqueeze(2).unsqueeze(3)  # [B, V, 1, 1, 2]
        diff = lane_pts - agents
        dist2 = diff.pow(2).sum(dim=-1)  # [B, V, L, P]
        lane_valid = lane_valid_pts.unsqueeze(1).expand(B, V, L, lane_valid_pts.size(-1))
        dist2 = dist2.masked_fill(~lane_valid, 1e9)
        lane_dist = dist2.min(dim=-1).values  # [B, V, L]

        if lane_keep_mask is not None:
            lane_dist = lane_dist.masked_fill(~lane_keep_mask.unsqueeze(1), 1e9)

        effective_agent_mask = agent_valid_mask & anchor_valid_mask
        lane_dist = lane_dist.masked_fill(~effective_agent_mask.unsqueeze(-1), 1e9)

        finite_mask = torch.isfinite(lane_dist) & (lane_dist < 1e8)
        valid_agents = finite_mask.any(dim=-1)  # [B, V]
        if not valid_agents.any():
            if return_indices:
                return None, None, None, None
            return None, None
        lane_dist = torch.where(finite_mask, lane_dist, torch.full_like(lane_dist, 1e8))

        K = min(self.topk_lanes, L)
        topk_vals, topk_idx = torch.topk(lane_dist, k=K, dim=-1, largest=False)
        lane_valid_mask = valid_agents.unsqueeze(-1) & (topk_vals < 1e8)
        lane_pad = ~lane_valid_mask  # True => padding lane slot

        lane_feat_exp = lane_feat_shared.unsqueeze(1).expand(B, V, L, self.filter_size)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.filter_size)
        selected = torch.gather(lane_feat_exp, 2, gather_idx)
        selected = selected.masked_fill(lane_pad.unsqueeze(-1), 0.0)

        lane_features = selected.reshape(B * V, K, self.filter_size)
        lane_pad = lane_pad.reshape(B * V, K)
        if return_indices:
            lane_indices = topk_idx.masked_fill(~lane_valid_mask, -1)
            return lane_features, lane_pad, lane_indices, lane_valid_mask
        return lane_features, lane_pad

    def _expand_all_lanes(self, lane_feat_shared, map_mask, B, V):
        """Expand all valid lanes for each agent."""
        L = lane_feat_shared.shape[1]
        lane_features = lane_feat_shared.unsqueeze(1).expand(B, V, L, -1).reshape(B * V, L, self.filter_size)
        lane_valid_L = map_mask.any(dim=-1)
        lane_pad_L = ~lane_valid_L
        lane_pad = lane_pad_L.unsqueeze(1).expand(B, V, L).reshape(B * V, L)
        lane_indices = torch.arange(L, device=lane_feat_shared.device, dtype=torch.long).view(1, 1, L).expand(B, V, L)
        lane_indices = lane_indices.masked_fill(lane_pad.view(B, V, L), -1)
        return lane_features, lane_pad, lane_indices

    def _select_agent_global_lane_indices(
        self,
        map_data,
        map_mask,
        agent_xy,
        agent_valid_mask,
        anchor_valid_mask,
        global_k,
        lane_keep_mask=None,
    ):
        """
        Select per-agent global lanes by distance quantiles to each agent anchor.
        This avoids using a single ego-origin global list for all agents.
        """
        if map_data is None or map_mask is None or global_k <= 0:
            return None, None
        B, L, P, _ = map_data.shape
        if L == 0:
            return None, None

        coords = map_data[..., :2]  # [B, L, P, 2]
        lane_pts = coords.unsqueeze(1)  # [B,1,L,P,2]
        agents = agent_xy.unsqueeze(2).unsqueeze(3)  # [B,V,1,1,2]
        dist2 = (lane_pts - agents).pow(2).sum(dim=-1)  # [B,V,L,P]
        lane_valid_pts = map_mask.unsqueeze(1).expand(B, agent_xy.shape[1], L, P)
        dist2 = dist2.masked_fill(~lane_valid_pts, 1e9)
        lane_dist = dist2.min(dim=-1).values  # [B,V,L]

        if lane_keep_mask is not None:
            lane_dist = lane_dist.masked_fill(~lane_keep_mask.unsqueeze(1), 1e9)

        effective_agent_mask = agent_valid_mask & anchor_valid_mask
        lane_dist = lane_dist.masked_fill(~effective_agent_mask.unsqueeze(-1), 1e9)

        valid_lane = torch.isfinite(lane_dist) & (lane_dist < 1e8)
        if not valid_lane.any():
            return None, None

        G = min(int(global_k), L)
        V = agent_xy.shape[1]
        global_idx = torch.zeros(B, V, G, device=map_data.device, dtype=torch.long)
        global_valid = torch.zeros(B, V, G, device=map_data.device, dtype=torch.bool)

        for b in range(B):
            for v in range(V):
                valid_idx = torch.where(valid_lane[b, v])[0]
                if valid_idx.numel() == 0:
                    continue
                sorted_local = valid_idx[torch.argsort(lane_dist[b, v, valid_idx])]
                n = sorted_local.numel()
                if n <= G:
                    pick = sorted_local
                else:
                    pos = torch.linspace(0, n - 1, steps=G, device=map_data.device).round().long()
                    pick = sorted_local[pos]
                g = pick.numel()
                global_idx[b, v, :g] = pick
                global_valid[b, v, :g] = True

        if not global_valid.any():
            return None, None
        return global_idx, global_valid

    def _select_hybrid_lanes(
        self,
        lane_feat_shared,
        map_data,
        map_mask,
        agent_xy,
        agent_valid_mask,
        anchor_valid_mask,
        lane_keep_mask=None,
    ):
        """
        Hybrid lane selection:
        1) per-agent local top-k nearest lanes
        2) plus scene-level global lanes for long-horizon coverage
        """
        topk_feat, topk_pad, topk_indices, topk_valid = self._select_topk_lanes(
            lane_feat_shared,
            map_data,
            map_mask,
            agent_xy,
            agent_valid_mask,
            anchor_valid_mask,
            lane_keep_mask=lane_keep_mask,
            return_indices=True,
        )

        B, L, _ = lane_feat_shared.shape
        V = agent_xy.shape[1]
        global_idx, global_valid = self._select_agent_global_lane_indices(
            map_data,
            map_mask,
            agent_xy,
            agent_valid_mask,
            anchor_valid_mask,
            self.hybrid_global_lanes,
            lane_keep_mask=lane_keep_mask,
        )

        K = topk_indices.shape[2] if topk_indices is not None else 0
        if global_idx is None:
            if topk_feat is None:
                return None, None, None
            return topk_feat, topk_pad, topk_indices

        G = global_idx.shape[2]
        global_idx_agent = global_idx  # [B,V,G]
        global_valid_agent = global_valid & agent_valid_mask.unsqueeze(-1)

        if topk_indices is not None and K > 0:
            dup = (global_idx_agent.unsqueeze(-1) == topk_indices.unsqueeze(2)).any(dim=-1)  # [B,V,G]
            global_valid_agent = global_valid_agent & (~dup)

        combined_idx = torch.cat(
            [topk_indices if topk_indices is not None else global_idx_agent[:, :, :0], global_idx_agent], dim=2
        )  # [B,V,K+G]
        combined_valid = torch.cat(
            [topk_valid if topk_valid is not None else global_valid_agent[:, :, :0], global_valid_agent], dim=2
        )  # [B,V,K+G]

        S = combined_idx.shape[2]
        gather_idx = combined_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, self.filter_size)
        lane_feat_exp = lane_feat_shared.unsqueeze(1).expand(B, V, L, self.filter_size)
        merged_feat = torch.gather(lane_feat_exp, 2, gather_idx)
        merged_feat = merged_feat.masked_fill(~combined_valid.unsqueeze(-1), 0.0)
        merged_pad = ~combined_valid
        lane_indices = combined_idx.masked_fill(~combined_valid, -1)

        return merged_feat.reshape(B * V, S, self.filter_size), merged_pad.reshape(B * V, S), lane_indices

    def forward(
        self, *,
        condition=None,            # [B, V] or None
        agent_types=None,          # [B, V] or None
        agent_shape=None,          # [B, V, 3] or None, normalized [length, width, height]
        map_data=None,             # [B, L, P, C]
        map_mask=None,             # [B, L, P] (True=real point)
        traffic_light_data=None,   # [B, TL, C_tl]
        traffic_light_mask=None,   # [B, TL] (True=real point)
        history_data=None,         # [B, C, t, V] or None
        history_timestep_mask=None,# [B, t, V] or None
        target_vehicle_mask=None,  # [B, V] or None
        history_vehicle_mask=None, # [B, V] or None
        B_hint=None,
        V_hint=None
    ):
        # ---------- 1) 解析 B,V 与 device/dtype ----------
        B = V = None
        device = dtype = None

        if history_data is not None:
            B, _, _, V = history_data.shape
            device, dtype = history_data.device, history_data.dtype
        elif history_timestep_mask is not None:
            B, _, V = history_timestep_mask.shape
            device, dtype = history_timestep_mask.device, torch.float32
        elif agent_types is not None:
            B, V = agent_types.shape
            device, dtype = agent_types.device, torch.float32
        elif condition is not None:
            B, V = condition.shape
            device, dtype = condition.device, torch.float32
        elif target_vehicle_mask is not None:
            B, V = target_vehicle_mask.shape
            device, dtype = target_vehicle_mask.device, torch.float32
        elif (B_hint is not None) and (V_hint is not None):
            B, V = int(B_hint), int(V_hint)

        if B is None and map_data is not None:
            B = map_data.shape[0]
            device = map_data.device
            dtype = map_data.dtype
        if B is None and traffic_light_data is not None:
            B = traffic_light_data.shape[0]
            device = traffic_light_data.device
            dtype = traffic_light_data.dtype
        if device is None and map_data is not None:
            device = map_data.device
            dtype = map_data.dtype
        if device is None and traffic_light_data is not None:
            device = traffic_light_data.device
            dtype = traffic_light_data.dtype
        if device is None:
            device = torch.device("cpu")
            dtype = torch.float32
        if B is None or V is None:
            raise RuntimeError("ContextEncoder cannot infer (B,V). Provide history/labels/types/mask or pass B_hint,V_hint.")

        # ---------- agent 有效性 (用于嵌入 mask) ----------
        if target_vehicle_mask is not None:
            agent_valid_mask = target_vehicle_mask.to(device).bool()
        elif history_vehicle_mask is not None:
            agent_valid_mask = history_vehicle_mask.to(device).bool()
        elif history_timestep_mask is not None:
            agent_valid_mask = history_timestep_mask.any(dim=1).to(device).bool()
        elif history_data is not None:
            agent_valid_mask = (history_data.abs().sum(dim=(1, 2)) > 1e-6)
        else:
            agent_valid_mask = torch.ones(B, V, dtype=torch.bool, device=device)

        agent_anchor_xy, agent_anchor_vel, agent_anchor_yaw, agent_anchor_valid = self._extract_agent_anchor(
            history_data, history_timestep_mask, history_vehicle_mask, agent_valid_mask
        )

        # ---------- 2) 历史特征 (per-agent: [B*V, T_hist, F]) ----------
        history_tokens = None
        history_token_mask = None
        T_hist = 0
        hist_feat_bvtf = None  # [B, V, T, F] 用于agent交互
        if history_data is not None:
            hist_vehicle_mask = history_vehicle_mask if history_vehicle_mask is not None else (history_data.abs().sum(dim=(1, 2)) > 1e-6)
            hist_vehicle_mask = hist_vehicle_mask.to(history_data.device).bool()
            hist_timestep_mask = history_timestep_mask
            if hist_timestep_mask is None:
                hist_timestep_mask = create_timestep_mask(history_data, padding_value=0.0)
            hist_timestep_mask = hist_timestep_mask.bool()
            hist_vehicle_mask = hist_vehicle_mask & hist_timestep_mask.any(dim=1)

            hist_feat = self.history_extractor(history_data, hist_vehicle_mask, timestep_mask=hist_timestep_mask)  # [B,V,t,F]
            hist_feat = self.history_fusion(hist_feat)
            
            # 新增：Agent交互增强
            hist_feat = self.agent_interaction(
                hist_feat, history_data, hist_vehicle_mask, 
                timestep_mask=hist_timestep_mask
            )
            hist_feat_bvtf = hist_feat  # 保存用于后续
            
            B_h, V_h, T_hist, _ = hist_feat.shape
            # Per-agent 结构: [B*V, T_hist, F]
            history_tokens = hist_feat.reshape(B_h * V_h, T_hist, self.filter_size)
            # history_token_mask: [B*V, T_hist], True = padding
            agent_invalid = ~hist_vehicle_mask  # [B, V], True = invalid agent
            history_token_mask = ~hist_timestep_mask.permute(0, 2, 1).reshape(B_h * V_h, T_hist)
            history_token_mask = history_token_mask | agent_invalid.unsqueeze(-1).expand(B_h, V_h, T_hist).reshape(B_h * V_h, T_hist)

        # ---------- 3) 车道特征（所有 agent 共享，扩展为 per-agent） ----------
        lane_features = None
        lane_pad = None
        lane_indices = None
        L = 0
        if (map_data is not None) and (map_mask is not None):
            lane_feat_shared = self.map_feature_extractor(map_data, map_mask)  # [B, L, F]
            L = lane_feat_shared.shape[1]
            lane_keep_mask = self._build_lane_center_mask(map_data, map_mask)

            selection_mode = self.lane_selection_mode
            if history_data is None and selection_mode != "all":
                # Cold start / unconditional sampling should not be constrained by fixed Top-K.
                selection_mode = "all"

            if selection_mode == "topk":
                lane_features, lane_pad, lane_indices, _ = self._select_topk_lanes(
                    lane_feat_shared,
                    map_data,
                    map_mask,
                    agent_anchor_xy,
                    agent_valid_mask,
                    agent_anchor_valid,
                    lane_keep_mask=lane_keep_mask,
                    return_indices=True,
                )
            elif selection_mode == "hybrid":
                lane_features, lane_pad, lane_indices = self._select_hybrid_lanes(
                    lane_feat_shared,
                    map_data,
                    map_mask,
                    agent_anchor_xy,
                    agent_valid_mask,
                    agent_anchor_valid,
                    lane_keep_mask=lane_keep_mask,
                )
            else:
                lane_features, lane_pad, lane_indices = self._expand_all_lanes(lane_feat_shared, map_mask, B, V)

            if lane_features is None or lane_pad is None:
                lane_features, lane_pad, lane_indices = self._expand_all_lanes(lane_feat_shared, map_mask, B, V)

        # ---------- 3.5) 红绿灯特征（所有 agent 共享，扩展为 per-agent） ----------
        traffic_light_features = None
        traffic_light_pad = None
        if (traffic_light_data is not None) and (traffic_light_mask is not None):
            tl_feat_shared = self.traffic_light_feature_extractor(traffic_light_data, traffic_light_mask)  # [B, TL, F]
            TL = tl_feat_shared.shape[1]
            if TL > 0:
                traffic_light_features = tl_feat_shared.unsqueeze(1).expand(B, V, TL, -1).reshape(
                    B * V, TL, self.filter_size
                )
                traffic_light_pad = (~traffic_light_mask.bool()).unsqueeze(1).expand(B, V, TL).reshape(B * V, TL)

        # ---------- 4) 互相增强 (history ↔ map)，per-agent 形式 ----------
        if (history_tokens is not None) and (lane_features is not None):
            # history_tokens: [B*V, T_hist, F], lane_features: [B*V, L, F]
            enhanced_history, enhanced_map = self.history_map_cross_attn(
                history_tokens, lane_features,
                map_mask=lane_pad,
                history_mask=history_token_mask
            )
            
            # 新增：Lane-aware attention增强
            if history_data is not None and map_data is not None:
                # 获取agent位置和速度（使用最后一个有效时间步）
                agent_pos = agent_anchor_xy  # [B, V, 2]
                # 使用每个agent最后一个有效history时间步的速度，避免直接取[-1]导致mask错位
                agent_vel = agent_anchor_vel  # [B, V, 2]
                
                enhanced_history = self.lane_aware_attn(
                    enhanced_history, enhanced_map, 
                    agent_pos, agent_vel, 
                    map_data, map_mask if map_mask is not None else None,
                    lane_indices=lane_indices,
                    agent_valid_mask=agent_valid_mask
                )
            
            history_tokens = enhanced_history
            lane_features = enhanced_map

        # ---------- 5) 拼 KV 与 KV_MASK (per-agent: [B*V, S, F]) ----------
        kv_list, mask_list = [], []
        if lane_features is not None:
            kv_list.append(lane_features)  # [B*V, L, F]
            mask_list.append(lane_pad)      # [B*V, L]
        if traffic_light_features is not None:
            kv_list.append(traffic_light_features)  # [B*V, TL, F]
            mask_list.append(traffic_light_pad)     # [B*V, TL]
        if condition is not None:
            label_emb = self.label_embed(condition)  # [B, V, F]
            # 扩展为 per-agent: [B*V, 1, F]
            label_emb = label_emb.reshape(B * V, 1, self.filter_size)
            agent_invalid_mask = (~agent_valid_mask).reshape(B * V, 1)  # [B*V, 1]
            kv_list.append(label_emb)
            mask_list.append(agent_invalid_mask)
        if agent_types is not None:
            type_emb = self.agent_types_embed(agent_types)  # [B, V, F]
            type_emb = type_emb.reshape(B * V, 1, self.filter_size)
            agent_invalid_mask = (~agent_valid_mask).reshape(B * V, 1)
            kv_list.append(type_emb)
            mask_list.append(agent_invalid_mask)
        if agent_shape is not None:
            shape_feat = torch.nan_to_num(agent_shape.to(device=device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            shape_emb = self.agent_shape_proj(shape_feat).to(dtype).reshape(B * V, 1, self.filter_size)
            agent_invalid_mask = (~agent_valid_mask).reshape(B * V, 1)
            kv_list.append(shape_emb)
            mask_list.append(agent_invalid_mask)

        # Add rich anchor-state token so target branch can recover absolute geometry more directly.
        # This is especially helpful when target trajectory is represented as deltas.
        speed = torch.norm(agent_anchor_vel, dim=-1, keepdim=True)  # [B,V,1]
        dist = torch.norm(agent_anchor_xy, dim=-1, keepdim=True)    # [B,V,1]
        cos_yaw = torch.cos(agent_anchor_yaw)
        sin_yaw = torch.sin(agent_anchor_yaw)
        vel_dir = agent_anchor_vel / (speed + 1e-6)  # [B,V,2]
        stationary = (speed < 0.05).to(dtype=agent_anchor_xy.dtype)  # normalized-speed threshold
        anchor_valid_f = agent_anchor_valid.unsqueeze(-1).to(dtype=agent_anchor_xy.dtype)
        anchor_feat = torch.cat(
            [
                agent_anchor_xy,      # x, y
                agent_anchor_vel,     # vx, vy
                speed,                # speed
                cos_yaw, sin_yaw,     # heading unit vector
                dist,                 # radial distance to local origin
                vel_dir,              # velocity direction unit vector
                anchor_valid_f,       # validity bit
            ],
            dim=-1,
        )  # [B,V,11]
        anchor_feat = torch.nan_to_num(anchor_feat, nan=0.0, posinf=0.0, neginf=0.0)
        anchor_emb = self.agent_anchor_proj(anchor_feat.to(device=device, dtype=torch.float32)).to(dtype)
        anchor_emb = anchor_emb.reshape(B * V, 1, self.filter_size)
        anchor_invalid_mask = (~agent_anchor_valid).reshape(B * V, 1)
        kv_list.append(anchor_emb)
        mask_list.append(anchor_invalid_mask)

        if history_tokens is not None:
            kv_list.append(history_tokens)  # [B*V, T_hist, F]
            mask_list.append(history_token_mask)  # [B*V, T_hist]

        kv = torch.cat(kv_list, dim=1) if len(kv_list) > 0 else None            # [B*V, S, F]
        kv_mask = torch.cat(mask_list, dim=1) if len(mask_list) > 0 else None   # [B*V, S]

        context = {
            "kv": kv,
            "kv_mask": kv_mask,
            "B": B,
            "V": V,
        }
        return context


class BlockContextAdapter(nn.Module):
    """
    Lightweight per-block adapter for shared context.
    Starts as identity (zero-init scale) and learns block-specific context shifts.
    """
    def __init__(self, filter_size):
        super().__init__()
        self.ln = nn.LayerNorm(filter_size)
        self.mlp = nn.Sequential(
            nn.Linear(filter_size, filter_size * 4),
            nn.ReLU(inplace=True),
            nn.Linear(filter_size * 4, filter_size),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, context):
        if context is None:
            return None
        kv = context.get("kv", None)
        if kv is None:
            return context

        delta = self.mlp(self.ln(kv))
        kv_adapted = kv + self.scale.to(dtype=delta.dtype) * delta

        out = dict(context)
        out["kv"] = kv_adapted
        return out

# ------------------------- Affine Coupling -------------------------
class AffineCoupling(nn.Module):
    """
    Affine Coupling layer for normalizing flow.
    Uses shared context from ContextEncoder (computed once per Glow pass),
    optionally adapted per block before entering the flow.
    """
    def __init__(self, in_channel, condition_dim, filter_size=256, affine=True):
        super().__init__()
        self.affine = affine
        self.filter_size = filter_size
        self.out_channels = in_channel // 2
        
        # Main feature extractor for in_a
        self.spatiotemporal_extractor = UnifiedSpatioTemporalExtractor(
            input_dim=in_channel // 2, filter_size=filter_size, num_heads=8, num_layers=1, causal=True
        )
        self.scene_linear = nn.Linear(filter_size, filter_size)
        
        # Cross-attention with external context (kv from ContextEncoder)
        self.global_attention = nn.MultiheadAttention(embed_dim=filter_size, num_heads=8, batch_first=True, dropout=0.0)
        self.global_ln1 = nn.LayerNorm(filter_size)
        self.global_ffn = nn.Sequential(
            nn.Linear(filter_size, filter_size * 4), 
            nn.ReLU(inplace=True), 
            nn.Linear(filter_size * 4, filter_size)
        )
        self.global_ln2 = nn.LayerNorm(filter_size)
        
        # Output head: per-time-step affine parameters => [B,V,T,2*C/2]
        self.head = nn.Sequential(
            nn.Linear(filter_size, filter_size), 
            nn.ReLU(inplace=True), 
            ZeroLinear(filter_size, 2 * self.out_channels)
        )

    def _compute_affine_params(self, in_a, vehicle_mask, context, timestep_mask=None):
        """Compute affine parameters from in_a and context.
        
        Args:
            in_a: [B, C/2, T, V] input tensor
            vehicle_mask: [B, V] bool, True = valid agent
            context: dict with 'kv' [B*V, S, F], 'kv_mask' [B*V, S], 'B', 'V'
            timestep_mask: [B, C, T, V] bool mask (squeezed form), True = valid
        """
        B, C_half, T, V = in_a.shape
        
        # Convert squeezed mask to [B, T, V] for spatiotemporal extractor
        st_timestep_mask = None
        if timestep_mask is not None:
            # timestep_mask: [B, C, T, V] -> take first channel or any channel
            st_timestep_mask = timestep_mask[:, 0, :, :]  # [B, T, V]
        
        # Extract features from in_a
        main_feat = self.spatiotemporal_extractor(in_a, vehicle_mask, timestep_mask=st_timestep_mask)  # [B,V,T,F]
        scene_emb = self.scene_linear(main_feat)  # [B,V,T,F]
        # Per-agent 结构: [B*V, T, F]
        scene_emb_flat = scene_emb.reshape(B * V, T, self.filter_size)
        
        # Cross-attention with context (per-agent: [B*V, S, F])
        kv = context.get("kv", None) if context is not None else None
        kv_mask = context.get("kv_mask", None) if context is not None else None
        
        if kv is not None:
            # 确保没有全 padding 的行
            kv_mask_cleaned, all_pad_kv = ensure_not_all_pad(kv_mask)
            # 对于 kv 全 padding 的行，将 kv 置零
            if all_pad_kv is not None and all_pad_kv.any():
                kv = kv.clone()
                kv[all_pad_kv] = 0.0
            
            # Query: [B*V, T, F], Key/Value: [B*V, S, F]
            fused, _ = self.global_attention(query=scene_emb_flat, key=kv, value=kv, key_padding_mask=kv_mask_cleaned)
            
            # 对于 kv 全 padding 的行，将注意力输出置零
            if all_pad_kv is not None and all_pad_kv.any():
                fused = fused.clone()
                fused[all_pad_kv] = 0.0
        else:
            fused = torch.zeros_like(scene_emb_flat)
        
        # Residual + FFN
        res_g = scene_emb_flat + fused
        g1 = self.global_ln1(res_g)
        g2 = self.global_ffn(g1)
        fused_feat = self.global_ln2(g1 + g2).view(B, V, T, self.filter_size)
        
        # Compute affine parameters
        affine_params = self.head(fused_feat)  # [B,V,T,2*C/2]
        t_offset, log_s = torch.chunk(affine_params, 2, dim=-1)  # [B,V,T,C/2]
        
        # --- Fix: Clamp t_offset to prevent numerical explosion ---
        t_offset = torch.clamp(t_offset, -50.0, 50.0)
        # ----------------------------------------------------------

        t_offset = t_offset.permute(0, 3, 2, 1).contiguous()  # [B,C/2,T,V]
        log_s = log_s.permute(0, 3, 2, 1).contiguous()
        t_offset = torch.nan_to_num(t_offset, nan=0.0, posinf=0.0, neginf=0.0)
        log_s = torch.nan_to_num(log_s, nan=0.0, posinf=0.0, neginf=0.0)
        
        return t_offset, log_s

    def forward(self, input, condition=None, map_data=None, map_mask=None, agent_types=None, 
                history_data=None, target_vehicle_mask=None, history_vehicle_mask=None, 
                context=None, timestep_mask=None):
        """
        Args:
            timestep_mask: [B, C, T, V] bool, True = valid position
        """
        B, C, T, V = input.shape
        in_a, in_b = input.chunk(2, dim=1)
        
        # Get vehicle mask
        if target_vehicle_mask is not None:
            vehicle_mask = target_vehicle_mask
        elif history_data is not None:
            vehicle_mask = (history_data.abs().sum(dim=(1, 2)) > 1e-6)
        else:
            vehicle_mask = (input.abs().sum(dim=(1, 2)) > 1e-6)
        
        # Compute affine parameters (pass timestep_mask for attention masking)
        t_offset, log_s = self._compute_affine_params(in_a, vehicle_mask, context, timestep_mask=timestep_mask)
        
        # Apply affine transformation
        if self.affine:
            log_s = torch.clamp(log_s, -5.0, 5.0)
            s = torch.exp(log_s)
            out_b = (in_b + t_offset) * s
            
            # Compute logdet with mask
            if timestep_mask is not None:
                # timestep_mask: [B, C, T, V], 只取后半通道对应 in_b
                mask_b = timestep_mask[:, C//2:, :, :]  # [B, C/2, T, V]
                mask_b_f = mask_b.float()
                out_b = out_b * mask_b_f + in_b * (1.0 - mask_b_f)
                logdet = (log_s * mask_b.float()).view(B, -1).sum(dim=1)
            else:
                logdet = log_s.view(B, -1).sum(dim=1)
        else:
            out_b = in_b + t_offset
            if timestep_mask is not None:
                mask_b = timestep_mask[:, C//2:, :, :].float()
                out_b = out_b * mask_b + in_b * (1.0 - mask_b)
            logdet = torch.zeros(B, device=input.device)
        
        return torch.cat([in_a, out_b], dim=1), logdet

    def reverse(self, output, condition=None, map_data=None, map_mask=None, agent_types=None, 
                 history_data=None, target_vehicle_mask=None, history_vehicle_mask=None, context=None, timestep_mask=None):
        B, C, T, V = output.shape
        out_a, out_b = output.chunk(2, dim=1)
        
        # Get vehicle mask
        if target_vehicle_mask is not None:
            vehicle_mask = target_vehicle_mask
        elif history_data is not None:
            vehicle_mask = (history_data.abs().sum(dim=(1, 2)) > 1e-6)
        else:
            vehicle_mask = (output.abs().sum(dim=(1, 2)) > 1e-6)
        
        # Compute affine parameters (same as forward, with timestep_mask)
        t_offset, log_s = self._compute_affine_params(out_a, vehicle_mask, context, timestep_mask=timestep_mask)
        
        # Reverse affine transformation
        if self.affine:
            log_s = torch.clamp(log_s, -5.0, 5.0)
            s = torch.exp(log_s)
            in_b = out_b / s - t_offset
            if timestep_mask is not None:
                mask_b = timestep_mask[:, C//2:, :, :].float()
                in_b = in_b * mask_b + out_b * (1.0 - mask_b)
        else:
            in_b = out_b - t_offset
            if timestep_mask is not None:
                mask_b = timestep_mask[:, C//2:, :, :].float()
                in_b = in_b * mask_b + out_b * (1.0 - mask_b)
        
        return torch.cat([out_a, in_b], dim=1)


# ------------------------- Flow / Block / Glow -------------------------
class Flow(nn.Module):
    def __init__(self, in_channel, condition_dim, affine=True, conv_lu=True):
        super().__init__()
        self.actnorm = ActNorm(in_channel)
        self.invconv = InvConv2dLU(in_channel) if conv_lu else InvConv2d(in_channel)
        self.coupling = AffineCoupling(in_channel, condition_dim, affine=affine)

    def forward(self, input, condition=None, map_data=None, map_mask=None, agent_types=None, 
                history_data=None, target_vehicle_mask=None, history_vehicle_mask=None, 
                context=None, timestep_mask=None):
        """
        Args:
            timestep_mask: [B, C, T, V] bool, True = valid
        """
        out, logdet = self.actnorm(input, mask=timestep_mask)
        out, det1 = self.invconv(out, mask=timestep_mask)
        out, det2 = self.coupling(
            out, condition, map_data, map_mask, agent_types, history_data, 
            target_vehicle_mask, history_vehicle_mask, context=context, timestep_mask=timestep_mask
        )
        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        return out, logdet

    def reverse(self, output, condition=None, map_data=None, map_mask=None, agent_types=None, history_data=None, target_vehicle_mask=None, history_vehicle_mask=None, context=None, timestep_mask=None):
        input = self.coupling.reverse(output, condition, map_data, map_mask, agent_types, history_data, target_vehicle_mask, history_vehicle_mask, context=context, timestep_mask=timestep_mask)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input


class Block(nn.Module):
    def __init__(
        self,
        in_channel,
        condition_dim,
        n_flow,
        split=True,
        affine=True,
        conv_lu=True,
        max_points=30,
        map_input_dim=5,
        traffic_light_input_dim=3,
        num_lane_types=21,
        num_traffic_light_states=8,
        lane_type_encoding="embedding",
        lane_type_embed_dim=16,
        lane_tl_embed_dim=8,
        tl_state_embed_dim=8,
        lane_selection_mode="topk",
        topk_lanes=16,
        hybrid_global_lanes=16,
    ):
        super().__init__()
        squeeze_dim = in_channel * 2  # squeeze factor = 2
        self.flows = nn.ModuleList([Flow(squeeze_dim, condition_dim, affine=affine, conv_lu=conv_lu) for _ in range(n_flow)])
        self.split = split
        if split:
            self.prior = ZeroConv2d(in_channel, in_channel * 2)  # half channels after split
        else:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)  # no split, full channels

    def _squeeze_time2(self, input):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width)
        squeezed = squeezed.permute(0, 1, 3, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 2, height // 2, width)
        return out

    def _unsqueeze_time2(self, input):
        b_size, n_channel, height, width = input.shape
        unsqueezed = input.view(b_size, n_channel // 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 3, 2, 4)
        unsqueezed = unsqueezed.contiguous().view(b_size, n_channel // 2, height * 2, width)
        return unsqueezed

    def forward(
        self,
        input,
        condition=None,
        map_data=None,
        map_mask=None,
        traffic_light_data=None,
        traffic_light_mask=None,
        agent_types=None,
        agent_shape=None,
        history_data=None,
        history_timestep_mask=None,
        target_vehicle_mask=None,
        history_vehicle_mask=None,
        timestep_mask=None,
        context=None,
    ):
        """
        Args:
            timestep_mask: [B, T, V] bool, True = valid. Will be converted to squeezed form.
        """
        b_size, n_channel, height, width = input.shape

        out = self._squeeze_time2(input)  # [B, 2C, T/2, V]
        b_size, squeezed_c, squeezed_t, width = out.shape
        
        # Convert timestep_mask to squeezed form
        squeezed_mask = None
        if timestep_mask is not None:
            # timestep_mask: [B, T, V] -> squeezed_mask: [B, 2C, T/2, V]
            squeezed_mask = create_squeezed_mask(timestep_mask, squeeze_factor=2)
            # 扩展通道维度以匹配 squeezed 后的通道数
            # squeezed_mask 目前是 [B, 2, T/2, V]，需要扩展到 [B, 2*n_channel, T/2, V]
            squeezed_mask = squeezed_mask.repeat(1, n_channel, 1, 1)  # [B, 2C, T/2, V]

        if context is None:
            raise RuntimeError("Block.forward requires precomputed context from Glow.")

        logdet = torch.zeros(b_size, device=out.device)
        for flow in self.flows:
            out, det = flow(
                out, condition, map_data, map_mask, agent_types, history_data,
                target_vehicle_mask, history_vehicle_mask, context=context, timestep_mask=squeezed_mask
            )
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            # 计算 log_p 时使用 mask
            log_p_elem = gaussian_log_p(z_new, mean, log_sd, mask=squeezed_mask[:, :squeezed_c//2] if squeezed_mask is not None else None)
            log_p = log_p_elem.view(b_size, -1).sum(1)
        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p_elem = gaussian_log_p(out, mean, log_sd, mask=squeezed_mask if squeezed_mask is not None else None)
            log_p = log_p_elem.view(b_size, -1).sum(1)
            z_new = out
        
        return out, logdet, log_p, z_new

    def reverse(
        self,
        output,
        condition=None,
        map_data=None,
        map_mask=None,
        traffic_light_data=None,
        traffic_light_mask=None,
        agent_types=None,
        agent_shape=None,
        eps=None,
        history_data=None,
        history_timestep_mask=None,
        reconstruct=False,
        target_vehicle_mask=None,
        history_vehicle_mask=None,
        timestep_mask=None,
        context=None,
    ):
        """
        Reverse pass for sampling.
        
        Args:
            timestep_mask: [B, T, V] bool (original resolution), True = valid.
                          Will be downsampled for squeezed domain.
        """
        input = output
        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)
            else:
                input = eps
        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)
            else:
                zero = torch.zeros_like(input)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        if context is None:
            raise RuntimeError("Block.reverse requires precomputed context from Glow.")

        # Prepare squeezed timestep_mask for flow.reverse if needed
        # Note: Flow.reverse currently doesn't use timestep_mask for computation,
        # but we pass it for potential future use or consistency
        squeezed_mask = None
        if timestep_mask is not None:
            B_m, T_m, V_m = timestep_mask.shape
            T_squeezed = T_m // 2
            if T_squeezed > 0:
                squeezed_mask = create_squeezed_mask(timestep_mask, squeeze_factor=2)
                # squeezed_mask: [B, 2, T/2, V] -> repeat to match channel count
                n_channel = input.shape[1] // 2  # original channel count before squeeze
                squeezed_mask = squeezed_mask.repeat(1, n_channel, 1, 1)  # [B, 2C, T/2, V]

        for flow in self.flows[::-1]:
            input = flow.reverse(input, condition, map_data, map_mask, agent_types, history_data, target_vehicle_mask, history_vehicle_mask, context=context, timestep_mask=squeezed_mask)

        input = self._unsqueeze_time2(input)
        return input


class Glow(nn.Module):
    def __init__(
        self,
        in_channel,
        condition_dim,
        n_flow,
        n_block,
        affine=True,
        conv_lu=True,
        max_points=30,
        map_input_dim=5,
        traffic_light_input_dim=3,
        num_lane_types=21,
        num_traffic_light_states=8,
        lane_type_encoding="embedding",
        lane_type_embed_dim=16,
        lane_tl_embed_dim=8,
        tl_state_embed_dim=8,
        lane_selection_mode="topk",
        topk_lanes=16,
        hybrid_global_lanes=16,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        shared_filter = None
        for i in range(n_block - 1):
            block = Block(
                n_channel,
                condition_dim,
                n_flow,
                affine=affine,
                conv_lu=conv_lu,
                max_points=max_points,
                map_input_dim=map_input_dim,
                traffic_light_input_dim=traffic_light_input_dim,
                num_lane_types=num_lane_types,
                num_traffic_light_states=num_traffic_light_states,
                lane_type_encoding=lane_type_encoding,
                lane_type_embed_dim=lane_type_embed_dim,
                lane_tl_embed_dim=lane_tl_embed_dim,
                tl_state_embed_dim=tl_state_embed_dim,
                lane_selection_mode=lane_selection_mode,
                topk_lanes=topk_lanes,
                hybrid_global_lanes=hybrid_global_lanes,
            )
            self.blocks.append(block)
            if shared_filter is None:
                shared_filter = block.flows[0].coupling.filter_size
            # n_channel *= 2
        final_block = Block(
            n_channel,
            condition_dim,
            n_flow,
            split=False,
            affine=affine,
            max_points=max_points,
            map_input_dim=map_input_dim,
            traffic_light_input_dim=traffic_light_input_dim,
            num_lane_types=num_lane_types,
            num_traffic_light_states=num_traffic_light_states,
            lane_type_encoding=lane_type_encoding,
            lane_type_embed_dim=lane_type_embed_dim,
            lane_tl_embed_dim=lane_tl_embed_dim,
            tl_state_embed_dim=tl_state_embed_dim,
            lane_selection_mode=lane_selection_mode,
            topk_lanes=topk_lanes,
            hybrid_global_lanes=hybrid_global_lanes,
        )
        self.blocks.append(final_block)
        if shared_filter is None:
            shared_filter = final_block.flows[0].coupling.filter_size

        # Shared context encoder for all blocks: compute once per Glow forward/reverse.
        self.context_encoder = ContextEncoder(
            filter_size=shared_filter,
            num_heads=8,
            history_input_dim=5,
            topk_lanes=topk_lanes,
            lane_selection_mode=lane_selection_mode,
            hybrid_global_lanes=hybrid_global_lanes,
            max_points=max_points,
            map_input_dim=map_input_dim,
            traffic_light_input_dim=traffic_light_input_dim,
            num_lane_types=num_lane_types,
            num_traffic_light_states=num_traffic_light_states,
            lane_type_encoding=lane_type_encoding,
            lane_type_embed_dim=lane_type_embed_dim,
            lane_tl_embed_dim=lane_tl_embed_dim,
            tl_state_embed_dim=tl_state_embed_dim,
        )
        # Per-block lightweight adapters (zero-init => identity at start).
        self.block_context_adapters = nn.ModuleList(
            [BlockContextAdapter(shared_filter) for _ in range(len(self.blocks))]
        )

    def forward(
        self,
        input,
        condition=None,
        map_data=None,
        map_mask=None,
        traffic_light_data=None,
        traffic_light_mask=None,
        agent_types=None,
        agent_shape=None,
        history_data=None,
        history_timestep_mask=None,
        target_vehicle_mask=None,
        history_vehicle_mask=None,
        timestep_mask=None,
    ):
        """
        Args:
            input: [B, C, T, V] trajectory data
            timestep_mask: [B, T, V] bool, True = valid position, False = padding
                          If None, will be auto-generated from input data.
        """
        B = input.size(0)
        log_p_sum = torch.zeros(B, device=input.device)
        logdet = torch.zeros(B, device=input.device)
        out = input
        z_outs = []
        
        # Auto-generate timestep_mask if not provided
        if timestep_mask is None:
            timestep_mask = create_timestep_mask(input, padding_value=-1.0)

        context = self.context_encoder(
            condition=condition,
            agent_types=agent_types,
            agent_shape=agent_shape,
            map_data=map_data,
            map_mask=map_mask,
            traffic_light_data=traffic_light_data,
            traffic_light_mask=traffic_light_mask,
            history_data=history_data,
            history_timestep_mask=history_timestep_mask,
            target_vehicle_mask=target_vehicle_mask,
            history_vehicle_mask=history_vehicle_mask,
            B_hint=input.shape[0],
            V_hint=input.shape[3],
        )
        
        current_mask = timestep_mask  # [B, T, V]
        
        for block_idx, block in enumerate(self.blocks):
            block_context = self.block_context_adapters[block_idx](context)
            out, det, log_p, z_new = block(
                out, condition, map_data, map_mask, traffic_light_data, traffic_light_mask,
                agent_types, agent_shape, history_data, history_timestep_mask,
                target_vehicle_mask, history_vehicle_mask, timestep_mask=current_mask, context=block_context
            )
            z_outs.append(z_new)
            logdet = logdet + det
            if log_p is not None:
                log_p_sum = log_p_sum + log_p
            
            # Update mask for next block (after squeeze, T becomes T/2)
            if current_mask is not None:
                B_m, T_m, V_m = current_mask.shape
                if T_m >= 2:
                    # Downsample mask: [B, T, V] -> [B, T/2, V]
                    current_mask = current_mask.view(B_m, T_m // 2, 2, V_m).any(dim=2)
        
        return log_p_sum, logdet, z_outs

    def reverse(
        self,
        z_list,
        condition=None,
        map_data=None,
        map_mask=None,
        traffic_light_data=None,
        traffic_light_mask=None,
        agent_types=None,
        agent_shape=None,
        history_data=None,
        history_timestep_mask=None,
        reconstruct=False,
        target_vehicle_mask=None,
        history_vehicle_mask=None,
        timestep_mask=None,
    ):
        """
        Reverse pass for sampling.
        
        Args:
            timestep_mask: [B, T, V] bool, True = valid. Used for context encoding.
                          If None, mask information is not used in reverse.
        """
        n_block = len(self.blocks)
        
        # 预计算每个 block 对应的 downsampled timestep_mask
        # block 0 对应原始分辨率，block 1 对应 T/2，block 2 对应 T/4，...
        # reverse 时从最后一个 block 开始，所以需要反向的 mask 列表
        mask_list = []
        if timestep_mask is not None:
            current_mask = timestep_mask
            for _ in range(n_block):
                mask_list.append(current_mask)
                B_m, T_m, V_m = current_mask.shape
                if T_m >= 2:
                    # Downsample: [B, T, V] -> [B, T/2, V]
                    current_mask = current_mask.view(B_m, T_m // 2, 2, V_m).any(dim=2)
                # 如果 T < 2，保持原样

        context = self.context_encoder(
            condition=condition,
            agent_types=agent_types,
            agent_shape=agent_shape,
            map_data=map_data,
            map_mask=map_mask,
            traffic_light_data=traffic_light_data,
            traffic_light_mask=traffic_light_mask,
            history_data=history_data,
            history_timestep_mask=history_timestep_mask,
            target_vehicle_mask=target_vehicle_mask,
            history_vehicle_mask=history_vehicle_mask,
            B_hint=z_list[-1].shape[0],
            V_hint=z_list[-1].shape[3],
        )
        
        for i, block in enumerate(self.blocks[::-1]):
            block_idx = n_block - 1 - i
            block_context = self.block_context_adapters[block_idx](context)
            # 获取当前 block 对应的 mask（reverse 顺序：最后一个 block 用最小分辨率的 mask）
            block_mask = None
            if timestep_mask is not None and len(mask_list) > 0:
                # blocks[::-1] 的第 i 个对应原始的第 (n_block - 1 - i) 个
                # 但 mask_list 是按 forward 顺序存的，所以取 mask_list[n_block - 1 - i]
                block_mask = mask_list[block_idx]
            
            if i == 0:
                input = block.reverse(
                    z_list[-1],
                    condition,
                    map_data,
                    map_mask,
                    traffic_light_data,
                    traffic_light_mask,
                    agent_types,
                    agent_shape,
                    z_list[-1],
                    history_data,
                    history_timestep_mask,
                    reconstruct=reconstruct,
                    target_vehicle_mask=target_vehicle_mask,
                    history_vehicle_mask=history_vehicle_mask,
                    timestep_mask=block_mask,
                    context=block_context,
                )
            else:
                input = block.reverse(
                    input,
                    condition,
                    map_data,
                    map_mask,
                    traffic_light_data,
                    traffic_light_mask,
                    agent_types,
                    agent_shape,
                    z_list[-(i + 1)],
                    history_data,
                    history_timestep_mask,
                    reconstruct=reconstruct,
                    target_vehicle_mask=target_vehicle_mask,
                    history_vehicle_mask=history_vehicle_mask,
                    timestep_mask=block_mask,
                    context=block_context,
                )
        return input
