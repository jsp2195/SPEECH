from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)


def mix_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    clean_power = rms(clean)
    noise_power = rms(noise)
    scale = clean_power / (10 ** (snr_db / 20.0) * noise_power + 1e-8)
    return clean + noise * scale


def random_crop_or_pad(wave: torch.Tensor, target_len: int) -> torch.Tensor:
    t = wave.shape[-1]
    if t == target_len:
        return wave
    if t > target_len:
        start = random.randint(0, t - target_len)
        return wave[..., start : start + target_len]
    return F.pad(wave, (0, target_len - t))


def apply_rir(wave: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    rir = rir / (rir.abs().max() + 1e-8)
    out = F.conv1d(
        wave.unsqueeze(1),
        torch.flip(rir.unsqueeze(0).unsqueeze(0), dims=[-1]),
        padding=rir.numel() - 1,
    )
    return out.squeeze(1)[..., : wave.shape[-1]]
