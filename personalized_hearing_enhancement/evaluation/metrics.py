from __future__ import annotations

import math

import torch

from personalized_hearing_enhancement.simulation.hearing_loss import apply_hearing_loss


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


def hf_restoration_metrics(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    impaired: torch.Tensor,
    sr: int,
    threshold_hz: int = 3000,
) -> dict[str, float]:
    estimate_ratio = high_frequency_energy_ratio(reference, estimate, sr=sr, threshold_hz=threshold_hz)
    impaired_ratio = high_frequency_energy_ratio(reference, impaired, sr=sr, threshold_hz=threshold_hz)
    return {
        "hf_energy_ratio": float(estimate_ratio),
        "hf_improvement_vs_impaired": float(estimate_ratio - impaired_ratio),
    }


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


def safety_metrics(reference: torch.Tensor, estimate: torch.Tensor) -> dict[str, float]:
    if estimate.ndim == 2:
        estimate = estimate.squeeze(0)
    if reference.ndim == 2:
        reference = reference.squeeze(0)
    abs_est = torch.abs(estimate)
    clipping_fraction = float((abs_est >= 0.999).float().mean().item())
    peak_linear = float(abs_est.max().item())
    peak_dbfs = float(20.0 * math.log10(max(peak_linear, 1e-8)))
    rms = float(torch.sqrt(torch.mean(estimate**2) + 1e-8).item())
    ref_rms = float(torch.sqrt(torch.mean(reference**2) + 1e-8).item())
    loudness_delta_db = float(20.0 * math.log10((rms + 1e-8) / (ref_rms + 1e-8)))
    return {
        "safety_clipping_fraction": clipping_fraction,
        "safety_peak_dbfs": peak_dbfs,
        "safety_rms": rms,
        "safety_loudness_delta_db": loudness_delta_db,
    }


def log_spectral_distance(reference: torch.Tensor, estimate: torch.Tensor, n_fft: int = 512, hop_length: int = 128) -> float:
    if reference.ndim == 1:
        reference = reference.unsqueeze(0)
    if estimate.ndim == 1:
        estimate = estimate.unsqueeze(0)
    if reference.shape != estimate.shape:
        raise ValueError(f"Shape mismatch for log_spectral_distance: {reference.shape} vs {estimate.shape}")

    win = torch.hann_window(n_fft, device=reference.device)
    ref_stft = torch.stft(reference, n_fft=n_fft, hop_length=hop_length, window=win, return_complex=True)
    est_stft = torch.stft(estimate, n_fft=n_fft, hop_length=hop_length, window=win, return_complex=True)
    ref_log = torch.log1p(ref_stft.abs())
    est_log = torch.log1p(est_stft.abs())
    return float(torch.mean(torch.abs(ref_log - est_log)).item())


def listener_space_metrics(
    original: torch.Tensor,
    output: torch.Tensor,
    audiogram: torch.Tensor,
    sr: int,
    *,
    output_already_impaired: bool = False,
) -> dict[str, float]:
    if original.ndim == 1:
        original = original.unsqueeze(0)
    if output.ndim == 1:
        output = output.unsqueeze(0)
    if audiogram.ndim == 1:
        audiogram = audiogram.unsqueeze(0)

    heard_original = apply_hearing_loss(original, audiogram, sr=sr)
    heard_output = output if output_already_impaired else apply_hearing_loss(output, audiogram, sr=sr)

    return {
        "listener_space_si_sdr": float(si_sdr(heard_original, heard_output).mean().item()),
        "listener_space_spectral_distance": log_spectral_distance(heard_original, heard_output),
    }


def three_way_user_benefit_metrics(
    original: torch.Tensor,
    impaired: torch.Tensor,
    calibration: torch.Tensor,
    conditioned: torch.Tensor,
    audiogram: torch.Tensor,
    sr: int,
) -> dict[str, dict]:
    signal_space = {
        "impaired": {
            "signal_space_si_sdr": float(si_sdr(original, impaired).mean().item()),
            "signal_space_spectral_distance": log_spectral_distance(original, impaired),
        },
        "calibration": {
            "signal_space_si_sdr": float(si_sdr(original, calibration).mean().item()),
            "signal_space_spectral_distance": log_spectral_distance(original, calibration),
        },
        "conditioned": {
            "signal_space_si_sdr": float(si_sdr(original, conditioned).mean().item()),
            "signal_space_spectral_distance": log_spectral_distance(original, conditioned),
        },
    }

    listener_space = {
        "impaired": listener_space_metrics(original, impaired, audiogram, sr=sr, output_already_impaired=True),
        "calibration": listener_space_metrics(original, calibration, audiogram, sr=sr),
        "conditioned": listener_space_metrics(original, conditioned, audiogram, sr=sr),
    }

    hf = {
        "impaired": hf_restoration_metrics(original, impaired, impaired, sr=sr),
        "calibration": hf_restoration_metrics(original, calibration, impaired, sr=sr),
        "conditioned": hf_restoration_metrics(original, conditioned, impaired, sr=sr),
    }

    safety = {
        "impaired": safety_metrics(original, impaired),
        "calibration": safety_metrics(original, calibration),
        "conditioned": safety_metrics(original, conditioned),
    }

    comparison = {
        "question": "Does conditioned ML improve on calibration baseline in listener space?",
        "conditioned_vs_calibration_listener_space_delta": {
            "listener_space_si_sdr": listener_space["conditioned"]["listener_space_si_sdr"]
            - listener_space["calibration"]["listener_space_si_sdr"],
            "listener_space_spectral_distance": listener_space["conditioned"]["listener_space_spectral_distance"]
            - listener_space["calibration"]["listener_space_spectral_distance"],
        },
        "conditioned_vs_calibration_hf_delta": {
            "hf_energy_ratio": hf["conditioned"]["hf_energy_ratio"] - hf["calibration"]["hf_energy_ratio"],
            "hf_improvement_vs_impaired": hf["conditioned"]["hf_improvement_vs_impaired"]
            - hf["calibration"]["hf_improvement_vs_impaired"],
        },
        "conditioned_vs_calibration_safety_delta": {
            "safety_clipping_fraction": safety["conditioned"]["safety_clipping_fraction"] - safety["calibration"]["safety_clipping_fraction"],
            "safety_peak_dbfs": safety["conditioned"]["safety_peak_dbfs"] - safety["calibration"]["safety_peak_dbfs"],
            "safety_loudness_delta_db": safety["conditioned"]["safety_loudness_delta_db"]
            - safety["calibration"]["safety_loudness_delta_db"],
        },
    }

    return {
        "signal_space": signal_space,
        "listener_space": listener_space,
        "hf": hf,
        "safety": safety,
        "comparison": comparison,
    }
