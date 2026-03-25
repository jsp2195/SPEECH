from __future__ import annotations

import torch
import torch.nn as nn

from personalized_hearing_enhancement.models.tasnet import CausalConv1d


class AudiogramEncoder(nn.Module):
    def __init__(self, in_dim: int = 8, hidden: int = 64, out_dim: int = 128) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        if a.ndim != 2:
            raise ValueError(f"Audiogram must be shaped (B, {self.in_dim}), got {tuple(a.shape)}")
        if a.shape[-1] != self.in_dim:
            raise ValueError(f"Audiogram feature dim mismatch: expected {self.in_dim}, got {a.shape[-1]}")
        return self.net(a)


class FiLMTCNBlock(nn.Module):
    def __init__(self, channels: int, hidden: int, dilation: int, cond_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(channels, hidden, 1)
        self.dw = CausalConv1d(hidden, hidden, 3, groups=hidden, dilation=dilation)
        self.norm = nn.GroupNorm(1, hidden)
        self.prelu = nn.PReLU()
        self.out_proj = nn.Conv1d(hidden, channels, 1)
        self.film = nn.Linear(cond_dim, hidden * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.in_proj(x)
        x = self.dw(x)
        x = self.norm(x)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        x = x * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        x = self.prelu(x)
        x = self.out_proj(x)
        return x + res


class ConditionedConvTasNet(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 192,
        feature_dim: int = 96,
        hidden_dim: int = 384,
        kernel_size: int = 16,
        tcn_layers: int = 8,
        tcn_stacks: int = 3,
        bottleneck_dim: int = 128,
        audiogram_dim: int = 8,
    ) -> None:
        super().__init__()
        # feature_dim is accepted for strict config compatibility with ConvTasNet.
        del feature_dim
        self.audiogram_dim = audiogram_dim
        self.encoder = nn.Conv1d(1, encoder_dim, kernel_size, stride=kernel_size // 2, bias=False)
        self.bottleneck = nn.Conv1d(encoder_dim, bottleneck_dim, 1)
        self.audiogram_encoder = AudiogramEncoder(in_dim=audiogram_dim, out_dim=bottleneck_dim)
        self.blocks = nn.ModuleList(
            [
                FiLMTCNBlock(bottleneck_dim, hidden_dim, 2**i, bottleneck_dim)
                for _ in range(tcn_stacks)
                for i in range(tcn_layers)
            ]
        )
        self.mask = nn.Sequential(nn.PReLU(), nn.Conv1d(bottleneck_dim, encoder_dim, 1), nn.Sigmoid())
        self.decoder = nn.ConvTranspose1d(encoder_dim, 1, kernel_size, stride=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor, audiogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: waveform tensor shaped (B, T) or (B, 1, T)
            audiogram: conditioning tensor shaped (B, 8)
        """
        if audiogram is None:
            raise ValueError("ConditionedConvTasNet.forward requires an audiogram tensor; got None")

        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError(f"Waveform must be shaped (B, T) or (B, 1, T), got {tuple(x.shape)}")

        if audiogram.ndim != 2 or audiogram.shape[0] != x.shape[0]:
            raise ValueError(
                f"Audiogram must be shaped (B, {self.audiogram_dim}) with matching batch size; "
                f"got {tuple(audiogram.shape)} for waveform batch {x.shape[0]}"
            )

        cond = self.audiogram_encoder(audiogram)
        enc = self.encoder(x)
        h = self.bottleneck(enc)
        for block in self.blocks:
            h = block(h, cond)
        m = self.mask(h)
        out = self.decoder(enc * m)
        return out[..., : x.shape[-1]].squeeze(1)
