from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from personalized_hearing_enhancement.simulation.hearing_loss import AUDIOGRAM_FREQS, apply_hearing_loss


@dataclass
class SanityResult:
    passed: bool
    details: dict[str, float]


def hearing_simulator_validation(sr: int = 16000, duration_s: float = 1.0) -> SanityResult:
    n = int(sr * duration_s)
    t = torch.arange(n) / sr
    if float(AUDIOGRAM_FREQS.max().item()) > sr / 2:
        raise ValueError(f"AUDIOGRAM_FREQS exceed Nyquist ({sr/2}): {AUDIOGRAM_FREQS.tolist()}")

    audiogram = torch.tensor([[0, 6, 12, 18, 24, 30, 36, 42]], dtype=torch.float32)
    measured_errors: dict[str, float] = {}

    for i, f in enumerate(AUDIOGRAM_FREQS.tolist()):
        wave = torch.sin(2 * math.pi * float(f) * t).unsqueeze(0)
        out = apply_hearing_loss(wave, audiogram, sr=sr)
        in_rms = torch.sqrt(torch.mean(wave**2)).item()
        out_rms = torch.sqrt(torch.mean(out**2)).item()
        att_db = 20 * math.log10(max(out_rms, 1e-8) / max(in_rms, 1e-8))
        expected = -float(audiogram[0, i].item())
        measured_errors[f"err_{int(f)}hz_db"] = abs(att_db - expected)

    passed = max(measured_errors.values()) < 3.0
    return SanityResult(passed=passed, details=measured_errors)


def identity_model_check(model: torch.nn.Module, length: int = 32000, tol: float = 0.8) -> SanityResult:
    x = torch.randn(1, length)
    with torch.no_grad():
        try:
            y = model(x, torch.zeros(1, 8))
        except TypeError:
            y = model(x)
    rel = (torch.mean(torch.abs(y - x)) / (torch.mean(torch.abs(x)) + 1e-8)).item()
    return SanityResult(passed=rel < tol, details={"relative_l1_error": rel})
