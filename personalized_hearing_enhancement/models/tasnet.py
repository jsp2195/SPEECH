from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.left_padding = (self.kernel_size[0] - 1) * self.dilation[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0))
        return super().forward(x)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, hidden: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(channels, hidden, 1)
        self.dw = CausalConv1d(hidden, hidden, kernel_size, groups=hidden, dilation=dilation)
        self.out_proj = nn.Conv1d(hidden, channels, 1)
        self.prelu = nn.PReLU()
        self.norm = nn.GroupNorm(1, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.in_proj(x)
        x = self.dw(x)
        x = self.norm(x)
        x = self.prelu(x)
        x = self.out_proj(x)
        return x + res


class ConvTasNet(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 192,
        feature_dim: int = 96,
        hidden_dim: int = 384,
        kernel_size: int = 16,
        tcn_layers: int = 8,
        tcn_stacks: int = 3,
        bottleneck_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = nn.Conv1d(1, encoder_dim, kernel_size, stride=kernel_size // 2, bias=False)
        self.bottleneck = nn.Conv1d(encoder_dim, bottleneck_dim, 1)

        blocks = []
        for _ in range(tcn_stacks):
            for i in range(tcn_layers):
                blocks.append(TCNBlock(bottleneck_dim, hidden_dim, 3, dilation=2**i))
        self.tcn = nn.Sequential(*blocks)
        self.mask = nn.Sequential(nn.PReLU(), nn.Conv1d(bottleneck_dim, encoder_dim, 1), nn.Sigmoid())
        self.decoder = nn.ConvTranspose1d(encoder_dim, 1, kernel_size, stride=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        enc = self.encoder(x)
        b = self.bottleneck(enc)
        t = self.tcn(b)
        m = self.mask(t)
        out = self.decoder(enc * m)
        out = out[..., : x.shape[-1]]
        return out.squeeze(1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
