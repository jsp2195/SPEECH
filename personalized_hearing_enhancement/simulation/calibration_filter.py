from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from personalized_hearing_enhancement.audiometry.profiles import HearingProfile
from personalized_hearing_enhancement.simulation.hearing_loss import AUDIOGRAM_FREQS


DEVICE_PROFILE_ALIASES = {
    "earbuds": "generic_earbuds",
    "headphones": "generic_headphones",
    "airpods": "generic_airpods_like_earbuds",
    "overear": "generic_over_ear_headphones",
}


@dataclass
class CalibrationFilter:
    freq_response: torch.Tensor
    fir_kernel: torch.Tensor
    sample_rate: int
    fft_size: int
    hop_size: int
    window: torch.Tensor


class StreamingFFTConvolver:
    """Low-state chunked overlap-add FFT convolver."""

    def __init__(self, filt: CalibrationFilter):
        self.filt = filt
        self.n_fft = int(filt.fft_size)
        self.hop_size = int(filt.hop_size)
        self.kernel_len = int(filt.fir_kernel.numel())
        self.input_tail = torch.zeros(self.kernel_len - 1, dtype=torch.float32)
        self.overlap = torch.zeros(self.n_fft - self.hop_size, dtype=torch.float32)

    def process_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if chunk.ndim != 1:
            raise ValueError("Expected mono chunk tensor with shape (T,)")

        x = torch.cat([self.input_tail, chunk.to(torch.float32)], dim=0)
        self.input_tail = x[-(self.kernel_len - 1) :].clone()

        if x.numel() < self.hop_size:
            x = F.pad(x, (0, self.hop_size - x.numel()))
        x = x[: self.hop_size]

        framed = F.pad(x, (0, self.n_fft - self.hop_size))
        spec = torch.fft.rfft(framed * self.filt.window, n=self.n_fft)
        out = torch.fft.irfft(spec * self.filt.freq_response, n=self.n_fft)

        y = out[: self.n_fft - self.hop_size] + self.overlap
        self.overlap = out[self.hop_size :]
        return y[: chunk.numel()]

    def flush(self) -> torch.Tensor:
        tail = self.overlap.clone()
        self.overlap.zero_()
        return tail


def _interp_db(freqs: torch.Tensor, anchor_freqs: torch.Tensor, anchor_db: torch.Tensor) -> torch.Tensor:
    freqs = freqs.to(anchor_freqs.device)
    right_idx = torch.searchsorted(anchor_freqs, freqs, right=True).clamp(1, anchor_freqs.numel() - 1)
    left_idx = right_idx - 1

    left_f = anchor_freqs[left_idx]
    right_f = anchor_freqs[right_idx]
    left_db = anchor_db[left_idx]
    right_db = anchor_db[right_idx]

    alpha = ((freqs - left_f) / (right_f - left_f + 1e-8)).clamp(0.0, 1.0)
    interp_db = left_db + alpha * (right_db - left_db)
    interp_db = torch.where(freqs <= anchor_freqs[0], anchor_db[0], interp_db)
    interp_db = torch.where(freqs >= anchor_freqs[-1], anchor_db[-1], interp_db)
    return interp_db


def _smooth_db(db_curve: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    if kernel_size <= 1:
        return db_curve
    kernel = torch.hann_window(kernel_size, periodic=False, dtype=db_curve.dtype, device=db_curve.device)
    kernel = kernel / kernel.sum()
    smoothed = F.conv1d(db_curve.view(1, 1, -1), kernel.view(1, 1, -1), padding=kernel_size // 2)
    return smoothed.view(-1)


def _load_device_profiles() -> dict[str, dict[str, list[float]]]:
    profile_path = Path(__file__).resolve().parent / "device_profiles.json"
    with profile_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_device_profile(profile: str, *, debug: bool = False) -> tuple[str, dict[str, list[float]], str | None]:
    profiles = _load_device_profiles()
    normalized = DEVICE_PROFILE_ALIASES.get(profile, profile)
    warning = None
    if normalized not in profiles:
        warning = f"Device profile '{profile}' not found. Falling back to 'headphones'."
        normalized = "generic_headphones"
    if debug:
        warning = warning or f"Using device profile '{normalized}'."
    return normalized, profiles[normalized], warning


def build_calibration_filter(
    audiogram: torch.Tensor,
    sample_rate: int = 16000,
    device_profile: str = "headphones",
    max_gain_db: float = 20.0,
    fft_size: int = 1024,
) -> CalibrationFilter:
    if audiogram.ndim == 2:
        audiogram = audiogram[0]
    audiogram = audiogram.to(torch.float32)
    n_bins = fft_size // 2 + 1
    freqs = torch.linspace(0, sample_rate / 2, steps=n_bins, dtype=torch.float32)

    # Inverse hearing-loss compensation in dB (capped for stability/safety).
    ag_db = _interp_db(freqs, AUDIOGRAM_FREQS.to(torch.float32), audiogram)

    _, profile_data, _ = resolve_device_profile(device_profile)
    dev_freqs = torch.tensor(profile_data["freq_hz"], dtype=torch.float32)
    dev_db = torch.tensor(profile_data["response_db"], dtype=torch.float32)
    dev_interp_db = _interp_db(freqs, dev_freqs, dev_db)

    # Apply inverse device response + inverse hearing-loss compensation.
    target_gain_db = ag_db - dev_interp_db

    # High frequency stability roll-off near Nyquist.
    stability_start = 0.82 * (sample_rate / 2)
    hf_weight = ((sample_rate / 2 - freqs) / max(sample_rate / 2 - stability_start, 1.0)).clamp(0.0, 1.0)
    target_gain_db = target_gain_db * hf_weight

    target_gain_db = _smooth_db(target_gain_db, kernel_size=11)
    target_gain_db = target_gain_db.clamp(min=-12.0, max=max_gain_db)

    mag = torch.pow(10.0, target_gain_db / 20.0)
    mag = mag.clamp(0.25, torch.pow(torch.tensor(10.0), torch.tensor(max_gain_db / 20.0)))

    fir = torch.fft.irfft(mag.to(torch.complex64), n=fft_size)
    fir = torch.roll(fir, shifts=fft_size // 2, dims=0)
    fir = fir * torch.hann_window(fft_size, periodic=False)
    fir = fir / (fir.abs().sum() + 1e-8)

    return CalibrationFilter(
        freq_response=mag.to(torch.complex64),
        fir_kernel=fir.to(torch.float32),
        sample_rate=sample_rate,
        fft_size=fft_size,
        hop_size=fft_size // 2,
        window=torch.hann_window(fft_size, periodic=False),
    )




def build_calibration_from_profile(
    profile: "HearingProfile",
    sample_rate: int = 16000,
    device_profile: str | None = None,
    max_gain_db: float = 20.0,
    fft_size: int = 1024,
) -> CalibrationFilter:
    selected_device_profile = device_profile or profile.get_device_profile("headphones") or "headphones"
    return build_calibration_filter(
        audiogram=profile.to_tensor(),
        sample_rate=sample_rate,
        device_profile=selected_device_profile,
        max_gain_db=max_gain_db,
        fft_size=fft_size,
    )


def apply_profile_calibration(
    waveform: torch.Tensor,
    profile: "HearingProfile",
    sample_rate: int = 16000,
    device_profile: str | None = None,
    max_gain_db: float = 20.0,
    chunk_size: int = 512,
) -> torch.Tensor:
    selected_device_profile = device_profile or profile.get_device_profile("headphones") or "headphones"
    return apply_calibration_filter(
        waveform=waveform,
        audiogram=profile.to_tensor(),
        sample_rate=sample_rate,
        device_profile=selected_device_profile,
        max_gain_db=max_gain_db,
        chunk_size=chunk_size,
    )


def apply_calibration_filter(
    waveform: torch.Tensor,
    audiogram: torch.Tensor,
    sample_rate: int = 16000,
    device_profile: str = "headphones",
    max_gain_db: float = 20.0,
    chunk_size: int = 512,
) -> torch.Tensor:
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)
    filt = build_calibration_filter(
        audiogram=audiogram,
        sample_rate=sample_rate,
        device_profile=device_profile,
        max_gain_db=max_gain_db,
    )
    convolver = StreamingFFTConvolver(filt)
    chunks = []
    for idx in range(0, waveform.numel(), chunk_size):
        chunk = waveform[idx : idx + chunk_size]
        chunks.append(convolver.process_chunk(chunk))
    tail = convolver.flush()
    if tail.numel() > 0:
        chunks.append(tail[: max(0, waveform.numel() - sum(c.numel() for c in chunks))])
    return torch.cat(chunks, dim=0)[: waveform.numel()]
