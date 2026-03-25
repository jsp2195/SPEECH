from __future__ import annotations

import torch


def si_sdr(target: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if target.ndim == 1:
        target = target.unsqueeze(0)
    if estimate.ndim == 1:
        estimate = estimate.unsqueeze(0)

    target_zm = target - target.mean(dim=-1, keepdim=True)
    estimate_zm = estimate - estimate.mean(dim=-1, keepdim=True)

    proj = (torch.sum(estimate_zm * target_zm, dim=-1, keepdim=True) / (torch.sum(target_zm**2, dim=-1, keepdim=True) + eps)) * target_zm
    noise = estimate_zm - proj
    ratio = (torch.sum(proj**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def sisdr_loss(target: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    return -si_sdr(target, estimate).mean()


def waveform_l1(target: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(target - estimate))


def pesq_proxy(target: torch.Tensor, estimate: torch.Tensor) -> float:
    # Fast differentiable-ish proxy based on SDR range mapping.
    score = si_sdr(target, estimate).mean().item()
    return float(max(1.0, min(4.5, 1.0 + (score + 5.0) * 3.5 / 30.0)))
