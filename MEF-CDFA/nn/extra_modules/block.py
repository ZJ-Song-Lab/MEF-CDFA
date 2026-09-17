import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import Conv

__all__ = ['SobelConv', 'MutilScaleEdgeInfoGenetator', 'ConvEdgeFusion', 'GetIndexOutput',
           'HaarWaveletConv', 'ContrastDrivenFeatureAggregation']


class GetIndexOutput(nn.Module):
    """Select one tensor out of a list/tuple produced by an upstream multi-output module.

    Used to route the individual MSEG scales (r = 1, 2, 4) to the matching semantic level.
    """

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        assert isinstance(x, (list, tuple)), 'GetIndexOutput expects a multi-output input'
        return x[self.index]


# --------------------------------------------------------------------------------------
# MEF: Multi-Scale Edge-Semantic Fusion
# --------------------------------------------------------------------------------------
class SobelConv(nn.Module):
    """Fixed Sobel gradient-magnitude operator (parameter-free).

    E = sqrt((Kx * F)^2 + (Ky * F)^2), computed per channel with depthwise convolution.
    The kernels are strictly fixed and isolated from backpropagation.
    """

    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        sobel = torch.tensor([[1., 2., 1.],
                              [0., 0., 0.],
                              [-1., -2., -1.]])
        # Kx = [1, 2, 1]^T [1, 0, -1]  ->  transpose of the mask above
        kx = sobel.t().contiguous().view(1, 1, 3, 3).repeat(channel, 1, 1, 1)
        ky = sobel.view(1, 1, 3, 3).repeat(channel, 1, 1, 1)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, x):
        gx = F.conv2d(x, self.kx, bias=None, stride=1, padding=1, groups=self.channel)
        gy = F.conv2d(x, self.ky, bias=None, stride=1, padding=1, groups=self.channel)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)


class MutilScaleEdgeInfoGenetator(nn.Module):
    """Multi-Scale Edge Generator (MSEG).

    Sobel edge magnitude followed by a cascade of max-pooling at rates r in {1, 2, 4},
    yielding multi-scale edge maps that are broadcast to align spatially with the
    semantic levels {P3, P4, P5}. The stage introduces no trainable parameters; the
    1x1 convolution that aligns the edge map to the host width lives in the
    edge-guided fusion stage (ConvEdgeFusion).

    Args:
        inc (int): input channels.
        num_scales (int | list): number of pooled edge scales. A list (e.g. the legacy
            ``[[128, 256, 512]]`` yaml argument) is accepted and only its length is used,
            since the generator is channel-agnostic.
    """

    def __init__(self, inc, num_scales=3):
        super().__init__()
        if isinstance(num_scales, (list, tuple)):
            num_scales = len(num_scales)
        self.num_scales = num_scales
        self.sobel = SobelConv(inc)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        edge = self.sobel(x)          # r = 1
        outputs = [edge]
        for _ in range(self.num_scales - 1):
            edge = self.maxpool(edge)  # r = 2, 4, ...
            outputs.append(edge)
        return outputs


class ConvEdgeFusion(nn.Module):
    """Edge-Guided Feature Fusion (EGFF).

    F_out = Conv1x1( Conv3x3( F_sem (+) Conv1x1(E_edge) ) )

    The pooled edge map is dimensionally aligned by a single 1x1 convolution and fused
    with the semantic feature through element-wise addition, then refined by a 3x3
    convolution and a final 1x1 convolution.

    Args:
        inc (list[int]): ``[edge_channels, semantic_channels]`` of the two inputs.
        ouc (int): output (host) width.
    """

    def __init__(self, inc, ouc):
        super().__init__()
        c_edge, c_sem = inc[0], inc[1]
        self.edge_align = Conv(c_edge, ouc, 1)                              # Conv1x1(E_edge)
        self.sem_align = Conv(c_sem, ouc, 1) if c_sem != ouc else nn.Identity()
        self.conv_3x3_feature_extract = Conv(ouc, ouc, 3)
        self.conv_1x1 = Conv(ouc, ouc, 1)

    def forward(self, x):
        edge, sem = x[0], x[1]
        return self.conv_1x1(self.conv_3x3_feature_extract(self.sem_align(sem) + self.edge_align(edge)))


# --------------------------------------------------------------------------------------
# CDFA: Contrast-Driven Feature Aggregation
# --------------------------------------------------------------------------------------
class HaarWaveletConv(nn.Module):
    """Fixed single-level Haar wavelet decomposition (parameter-free).

    Scaling h = [1/sqrt2, 1/sqrt2] and wavelet g = [1/sqrt2, -1/sqrt2] give the four
    analysis kernels W_LL = h'h, W_LH = h'g, W_HL = g'h, W_HH = g'g. They are applied as
    a grouped convolution with stride 2 and valid boundary handling (no padding; odd
    inputs are replicate-padded by one row/column), so every subband has shape
    (B, C, H/2, W/2).
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        h = torch.tensor([1., 1.]) / math.sqrt(2.)
        g = torch.tensor([1., -1.]) / math.sqrt(2.)
        kernels = torch.stack([torch.outer(h, h),   # LL
                               torch.outer(h, g),   # LH
                               torch.outer(g, h),   # HL
                               torch.outer(g, g)])  # HH  -> [4, 2, 2]
        # grouped conv: out channel index = c * 4 + alpha
        weight = kernels.unsqueeze(1).repeat(in_channels, 1, 1, 1)  # [4C, 1, 2, 2]
        self.register_buffer('weight', weight)

    def forward(self, x):
        b, c, h, w = x.shape
        if h % 2 or w % 2:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode='replicate')
        out = F.conv2d(x, self.weight, bias=None, stride=2, padding=0, groups=c)
        oh, ow = out.shape[-2:]
        out = out.view(b, c, 4, oh, ow).permute(0, 2, 1, 3, 4).contiguous()
        return [out[:, i] for i in range(4)]  # F^LL, F^LH, F^HL, F^HH


class ContrastDrivenFeatureAggregation(nn.Module):
    """Contrast-Driven Feature Aggregation (CDFA).

    F -> Haar decomposition (4C at H/2 x W/2) -> frequency-specific attention ->
    subband branches -> bottleneck -> 2x bilinear upsampling -> sigmoid gated fusion.

    Frequency-specific attention: z^a = GAP(F^a), s^a = w_z^T z^a with a single shared
    projection vector w_z, so each subband receives one scalar weight (four in total,
    broadcast over channels and positions). The scores are normalized by a
    temperature-scaled softmax with tau = sqrt(d_k) = sqrt(C).

    Reconstruction: each weighted subband goes through a branch C_a of two stacked
    3x3 Conv-BN-SiLU (C -> C/2 -> C/2); the 4 branches are concatenated (2C), compressed
    by the 1x1 bottleneck Phi (2C -> C) and upsampled 2x bilinearly.

    Integration: G = sigmoid(W_g * F_sem) with W_g a 1x1 convolution (C -> C), and
    F_out = G * F_sem + (1 - G) * F_cdfa.

    Args:
        dim (int): channel width C of the host feature level.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.wavelet = HaarWaveletConv(dim)

        # shared projection vector w_z in R^C and temperature tau = sqrt(C)
        self.w_z = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
        self.tau = math.sqrt(dim)

        half = max(dim // 2, 1)
        self.branches = nn.ModuleList(
            nn.Sequential(Conv(dim, half, 3), Conv(half, half, 3)) for _ in range(4))
        self.bottleneck = Conv(4 * half, dim, 1)          # Phi: 2C -> C
        self.gate = nn.Conv2d(dim, dim, kernel_size=1)    # W_g: C -> C

    def forward(self, x):
        f_sem = x
        subbands = self.wavelet(f_sem)                                        # 4 x (B, C, H/2, W/2)

        z = torch.stack([torch.flatten(s, 2).mean(-1) for s in subbands], 1)  # (B, 4, C)
        scores = z @ self.w_z                                                 # (B, 4)
        attn = torch.softmax(scores / self.tau, dim=1)                        # (B, 4)

        refined = [self.branches[i](subbands[i] * attn[:, i].view(-1, 1, 1, 1)) for i in range(4)]
        f_cdfa = self.bottleneck(torch.cat(refined, 1))                       # (B, C, H/2, W/2)
        f_cdfa = F.interpolate(f_cdfa, scale_factor=2, mode='bilinear', align_corners=False)
        if f_cdfa.shape[-2:] != f_sem.shape[-2:]:
            f_cdfa = F.interpolate(f_cdfa, size=f_sem.shape[-2:], mode='bilinear', align_corners=False)

        g = torch.sigmoid(self.gate(f_sem))
        return g * f_sem + (1.0 - g) * f_cdfa
