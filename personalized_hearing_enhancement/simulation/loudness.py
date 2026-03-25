from __future__ import annotations

import torch


def rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)


def rms_normalize(x: torch.Tensor, target_rms: float = 0.1, eps: float = 1e-8) -> torch.Tensor:
    scale = target_rms / (rms(x, eps=eps) + eps)
    return x * scale


def peak_limit(x: torch.Tensor, peak: float = 0.98) -> torch.Tensor:
    max_abs = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    scale = torch.clamp(peak / max_abs, max=1.0)
    return x * scale


def loudness_match(x: torch.Tensor, reference: torch.Tensor, enabled: bool = True) -> torch.Tensor:
    if not enabled:
        return x
    return x * (rms(reference) / (rms(x) + 1e-8))


def safe_post_amplification(
    x: torch.Tensor,
    reference: torch.Tensor | None = None,
    peak: float = 0.98,
    target_rms: float | None = None,
    match_reference_loudness: bool = True,
) -> torch.Tensor:
    out = x
    if target_rms is not None:
        out = rms_normalize(out, target_rms=target_rms)
    if reference is not None:
        out = loudness_match(out, reference, enabled=match_reference_loudness)
    out = peak_limit(out, peak=peak)
    return out.clamp(-1.0, 1.0)


def clipping_stats(x: torch.Tensor, clip_threshold: float = 0.999) -> dict[str, float]:
    clipped = (x.abs() >= clip_threshold).float().mean().item()
    peak = x.abs().max().item()
    return {"clip_fraction": float(clipped), "peak_abs": float(peak)}
