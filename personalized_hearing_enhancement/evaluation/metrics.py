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
    score = si_sdr(target, estimate).mean().item()
    return float(max(1.0, min(4.5, 1.0 + (score + 5.0) * 3.5 / 30.0)))


def bandwise_energy(waveform: torch.Tensor, sr: int, band_edges: list[tuple[int, int]] | None = None) -> dict[str, float]:
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)
    band_edges = band_edges or [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
    spec = torch.fft.rfft(waveform)
    freqs = torch.fft.rfftfreq(waveform.numel(), d=1.0 / sr)
    power = spec.abs() ** 2

    out: dict[str, float] = {}
    for lo, hi in band_edges:
        mask = (freqs >= lo) & (freqs < hi)
        value = power[mask].mean().item() if mask.any() else 0.0
        out[f"{lo}_{hi}hz"] = float(value)
    return out


def high_frequency_energy_ratio(reference: torch.Tensor, estimate: torch.Tensor, sr: int, threshold_hz: int = 3000) -> float:
    if reference.ndim == 2:
        reference = reference.squeeze(0)
    if estimate.ndim == 2:
        estimate = estimate.squeeze(0)
    r_spec = torch.fft.rfft(reference)
    e_spec = torch.fft.rfft(estimate)
    freqs = torch.fft.rfftfreq(reference.numel(), d=1.0 / sr)
    mask = freqs >= threshold_hz
    r_energy = (r_spec.abs()[mask] ** 2).mean().clamp(min=1e-8)
    e_energy = (e_spec.abs()[mask] ** 2).mean()
    return float((e_energy / r_energy).item())


def intelligibility_proxy(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    if reference.ndim == 2:
        reference = reference.squeeze(0)
    if estimate.ndim == 2:
        estimate = estimate.squeeze(0)
    ref = reference - reference.mean()
    est = estimate - estimate.mean()
    corr = torch.sum(ref * est) / (torch.sqrt(torch.sum(ref**2) * torch.sum(est**2)) + 1e-8)
    return float(corr.clamp(-1.0, 1.0).item())


def gain_stats(input_signal: torch.Tensor, output_signal: torch.Tensor) -> dict[str, float]:
    in_rms = torch.sqrt(torch.mean(input_signal**2) + 1e-8)
    out_rms = torch.sqrt(torch.mean(output_signal**2) + 1e-8)
    gain_db = 20.0 * torch.log10(out_rms / in_rms + 1e-8)
    return {
        "input_rms": float(in_rms.item()),
        "output_rms": float(out_rms.item()),
        "gain_db": float(gain_db.item()),
    }
