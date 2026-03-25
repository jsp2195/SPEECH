from __future__ import annotations

import torch
import torch.nn.functional as F

AUDIOGRAM_FREQS = torch.tensor([250, 500, 1000, 2000, 4000, 6000, 8000, 9000], dtype=torch.float32)


def _interp_response(freqs: torch.Tensor, audiogram: torch.Tensor, sr: int) -> torch.Tensor:
    # freqs: (F,), audiogram: (B, 8)
    device = audiogram.device
    ag_freqs = AUDIOGRAM_FREQS.to(device=device)
    f = freqs.to(device=device).unsqueeze(0)  # (1, F)

    # Piecewise linear interpolation over audiogram anchors.
    right_idx = torch.searchsorted(ag_freqs, f.squeeze(0), right=True).clamp(1, ag_freqs.numel() - 1)
    left_idx = right_idx - 1

    left_f = ag_freqs[left_idx].unsqueeze(0)
    right_f = ag_freqs[right_idx].unsqueeze(0)

    left_db = torch.gather(audiogram, 1, left_idx.unsqueeze(0).expand(audiogram.size(0), -1))
    right_db = torch.gather(audiogram, 1, right_idx.unsqueeze(0).expand(audiogram.size(0), -1))

    alpha = ((f - left_f) / (right_f - left_f + 1e-8)).clamp(0.0, 1.0)
    interp_db = left_db + alpha * (right_db - left_db)

    low_mask = f <= ag_freqs[0]
    high_mask = f >= ag_freqs[-1]
    interp_db = torch.where(low_mask, audiogram[:, :1], interp_db)
    interp_db = torch.where(high_mask, audiogram[:, -1:], interp_db)

    mag = torch.pow(10.0, -interp_db / 20.0)
    # Mild smoothing with depthwise 1D conv.
    kernel = torch.tensor([0.2, 0.6, 0.2], device=device, dtype=mag.dtype).view(1, 1, 3)
    mag = F.conv1d(mag.unsqueeze(1), kernel, padding=1).squeeze(1)
    return mag


def apply_hearing_loss(waveform: torch.Tensor, audiogram: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """
    Differentiable audiogram-based attenuation using FFT domain shaping.

    Args:
        waveform: (B, T) or (T,)
        audiogram: (B, 8) or (8,)
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if audiogram.ndim == 1:
        audiogram = audiogram.unsqueeze(0)

    if waveform.size(0) != audiogram.size(0):
        if audiogram.size(0) == 1:
            audiogram = audiogram.expand(waveform.size(0), -1)
        else:
            raise ValueError("Batch size mismatch between waveform and audiogram")

    n = waveform.shape[-1]
    spec = torch.fft.rfft(waveform, dim=-1)
    freqs = torch.fft.rfftfreq(n, d=1.0 / sr).to(waveform.device)

    mag = _interp_response(freqs, audiogram, sr)
    shaped = spec * mag.to(spec.dtype)
    out = torch.fft.irfft(shaped, n=n, dim=-1)
    return out
