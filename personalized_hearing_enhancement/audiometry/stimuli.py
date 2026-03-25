from __future__ import annotations

import math

import numpy as np

from personalized_hearing_enhancement.simulation.hearing_loss import AUDIOGRAM_FREQS

STANDARD_FREQS_HZ = [int(x) for x in AUDIOGRAM_FREQS.tolist()]


def _validate_common(duration_s: float, sr: int, ramp_ms: float) -> int:
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr}")
    n = max(1, int(round(duration_s * sr)))
    if ramp_ms < 0:
        raise ValueError(f"ramp_ms must be >= 0, got {ramp_ms}")
    return n


def _amplitude_safe(amplitude: float) -> float:
    if not math.isfinite(amplitude):
        raise ValueError(f"amplitude must be finite, got {amplitude}")
    if amplitude < 0:
        raise ValueError(f"amplitude must be non-negative, got {amplitude}")
    return float(min(amplitude, 0.999))


def _apply_ramp(wave: np.ndarray, sr: int, ramp_ms: float) -> np.ndarray:
    ramp_samples = int(round(sr * ramp_ms / 1000.0))
    if ramp_samples <= 0:
        return wave
    ramp_samples = min(ramp_samples, len(wave) // 2)
    if ramp_samples == 0:
        return wave
    ramp = np.linspace(0.0, 1.0, ramp_samples, dtype=np.float32)
    out = wave.copy()
    out[:ramp_samples] *= ramp
    out[-ramp_samples:] *= ramp[::-1]
    return out


def pad_silence(wave: np.ndarray, sr: int, pre_s: float = 0.0, post_s: float = 0.0) -> np.ndarray:
    pre = np.zeros(max(0, int(round(pre_s * sr))), dtype=np.float32)
    post = np.zeros(max(0, int(round(post_s * sr))), dtype=np.float32)
    return np.concatenate([pre, wave.astype(np.float32), post])


def generate_tone_probe(
    frequency_hz: float,
    amplitude: float,
    duration_s: float,
    sr: int,
    ramp_ms: float = 10.0,
) -> np.ndarray:
    n = _validate_common(duration_s, sr, ramp_ms)
    if frequency_hz <= 0 or frequency_hz >= sr / 2:
        raise ValueError(f"frequency_hz must be in (0, Nyquist={sr/2}), got {frequency_hz}")
    amp = _amplitude_safe(amplitude)

    t = np.arange(n, dtype=np.float32) / float(sr)
    wave = amp * np.sin(2.0 * np.pi * float(frequency_hz) * t)
    wave = _apply_ramp(wave.astype(np.float32), sr=sr, ramp_ms=ramp_ms)
    peak = float(np.max(np.abs(wave)))
    if peak > 1.0:
        wave = wave / peak
    assert np.isfinite(wave).all(), "Probe contains non-finite values"
    assert float(np.max(np.abs(wave))) <= 1.0 + 1e-6, "Probe clips beyond [-1,1]"
    return wave.astype(np.float32)


def generate_narrowband_noise_probe(
    center_frequency_hz: float,
    amplitude: float,
    duration_s: float,
    sr: int,
    bandwidth_hz: float = 400.0,
    ramp_ms: float = 10.0,
) -> np.ndarray:
    n = _validate_common(duration_s, sr, ramp_ms)
    if bandwidth_hz <= 0:
        raise ValueError(f"bandwidth_hz must be > 0, got {bandwidth_hz}")
    amp = _amplitude_safe(amplitude)
    if center_frequency_hz <= 0 or center_frequency_hz >= sr / 2:
        raise ValueError(f"center_frequency_hz must be in (0, Nyquist={sr/2}), got {center_frequency_hz}")

    rng = np.random.default_rng(0)
    white = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    low = max(0.0, center_frequency_hz - bandwidth_hz / 2.0)
    high = min(sr / 2.0, center_frequency_hz + bandwidth_hz / 2.0)
    mask = (freqs >= low) & (freqs <= high)
    spec = spec * mask
    wave = np.fft.irfft(spec, n=n).astype(np.float32)
    denom = float(np.max(np.abs(wave))) + 1e-8
    wave = amp * (wave / denom)
    wave = _apply_ramp(wave, sr=sr, ramp_ms=ramp_ms)
    peak = float(np.max(np.abs(wave)))
    if peak > 1.0:
        wave = wave / peak
    assert np.isfinite(wave).all(), "Probe contains non-finite values"
    return wave.astype(np.float32)
