from __future__ import annotations

import torch

from personalized_hearing_enhancement.evaluation.metrics import sisdr_loss
from personalized_hearing_enhancement.simulation.hearing_loss import apply_hearing_loss
from personalized_hearing_enhancement.utils.audio import mel_spectrogram


def signal_space_loss(clean: torch.Tensor, pred: torch.Tensor, sr: int) -> torch.Tensor:
    l_sisdr = sisdr_loss(clean, pred)
    l_mel = torch.mean(torch.abs(mel_spectrogram(clean, sr) - mel_spectrogram(pred, sr)))
    l_l1 = torch.mean(torch.abs(clean - pred))
    return 0.7 * l_sisdr + 0.3 * l_mel + 0.1 * l_l1


def listener_space_loss(clean: torch.Tensor, pred: torch.Tensor, audiogram: torch.Tensor, sr: int) -> torch.Tensor:
    heard_clean = apply_hearing_loss(clean, audiogram, sr=sr)
    heard_pred = apply_hearing_loss(pred, audiogram, sr=sr)
    return torch.mean(torch.abs(heard_clean - heard_pred))


def combined_loss(
    clean: torch.Tensor,
    pred: torch.Tensor,
    audiogram: torch.Tensor,
    sr: int,
    listener_enabled: bool = False,
    listener_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    sig = signal_space_loss(clean, pred, sr=sr)
    total = sig
    details = {"signal_space_loss": float(sig.item())}

    if listener_enabled and listener_weight > 0.0:
        lis = listener_space_loss(clean, pred, audiogram, sr=sr)
        total = total + listener_weight * lis
        details["listener_space_loss"] = float(lis.item())
        details["listener_weight"] = float(listener_weight)

    return total, details
