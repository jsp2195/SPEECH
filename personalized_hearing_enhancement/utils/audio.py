from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


def load_audio(path: str | Path, sr: int = 16000) -> torch.Tensor:
    wav, src_sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)
    if src_sr != sr:
        wav = torchaudio.functional.resample(wav, src_sr, sr)
    return wav.squeeze(0)


def save_audio(path: str | Path, waveform: torch.Tensor, sr: int = 16000) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    wav = waveform.detach().cpu().unsqueeze(0)
    torchaudio.save(str(p), wav, sr)


def normalize_audio(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.abs().max(dim=-1, keepdim=True).values + eps)


def mel_spectrogram(x: torch.Tensor, sr: int = 16000, n_mels: int = 80) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=512,
        hop_length=128,
        win_length=512,
        n_mels=n_mels,
        power=1.0,
    ).to(x.device)
    return mel(x)
