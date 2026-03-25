from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from personalized_hearing_enhancement.audiometry.stimuli import STANDARD_FREQS_HZ


@dataclass
class HearingProfile:
    frequencies: list[int]
    thresholds_db: list[float]
    uncertainty: list[float] = field(default_factory=list)
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    device_profile: str | None = None
    notes: str = ""
    source: str = "estimated"
    sample_rate: int | None = None
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_tensor(self, *, batch_dim: bool = True, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        validate_profile(self)
        values = torch.tensor(self.thresholds_db, dtype=dtype)
        return values.unsqueeze(0) if batch_dim else values

    def get_device_profile(self, fallback: str | None = None) -> str | None:
        return self.device_profile or fallback

    def profile_summary(self) -> str:
        pairs = ", ".join(f"{f}Hz:{t:.1f}dB" for f, t in zip(self.frequencies, self.thresholds_db, strict=True))
        return (
            f"Profile source={self.source}, timestamp={self.timestamp_utc}\n"
            f"Device profile={self.device_profile or 'N/A'}\n"
            f"Thresholds: {pairs}"
        )

    def as_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["threshold_tensor"] = self.to_tensor().squeeze(0).tolist()
        return payload


def validate_profile(profile: HearingProfile) -> None:
    if profile.frequencies != STANDARD_FREQS_HZ:
        raise ValueError(f"Profile frequencies must equal {STANDARD_FREQS_HZ}, got {profile.frequencies}")
    if len(profile.thresholds_db) != len(profile.frequencies):
        raise ValueError("Profile thresholds length must match frequencies length")
    if profile.uncertainty and len(profile.uncertainty) != len(profile.frequencies):
        raise ValueError("Uncertainty length must match frequencies length")
    for i, value in enumerate(profile.thresholds_db):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite threshold at index {i}: {value}")
    for i, value in enumerate(profile.uncertainty or []):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite uncertainty at index {i}: {value}")


def save_profile(profile: HearingProfile, path: str | Path) -> Path:
    validate_profile(profile)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")
    return out


def load_profile(path: str | Path) -> HearingProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    profile = HearingProfile(**payload)
    validate_profile(profile)
    return profile


def parse_manual_audiogram(audiogram: str) -> list[float]:
    values = [float(x.strip()) for x in audiogram.split(",") if x.strip()]
    if len(values) != len(STANDARD_FREQS_HZ):
        raise ValueError(f"Manual audiogram must contain {len(STANDARD_FREQS_HZ)} values; got {len(values)}")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Manual audiogram contains non-finite values")
    return values


def create_manual_profile(
    manual_audiogram: str,
    *,
    device_profile: str | None = None,
    sample_rate: int | None = None,
    source: str = "manual",
    notes: str = "",
) -> HearingProfile:
    profile = HearingProfile(
        frequencies=list(STANDARD_FREQS_HZ),
        thresholds_db=parse_manual_audiogram(manual_audiogram),
        device_profile=device_profile,
        source=source,
        sample_rate=sample_rate,
        notes=notes,
    )
    validate_profile(profile)
    return profile


def resolve_profile_input(
    profile_json: str | None,
    manual_audiogram: str | None,
    *,
    logger=None,
    sample_rate: int | None = None,
    default_device_profile: str | None = None,
) -> tuple[HearingProfile, str]:
    if profile_json:
        profile = load_profile(profile_json)
        if manual_audiogram and logger is not None:
            logger.warning("Both --profile-json and --audiogram were provided; --profile-json takes precedence.")
        return profile, f"profile_json:{profile_json}"

    if manual_audiogram is None:
        raise ValueError("Either profile_json or manual_audiogram must be provided")

    profile = create_manual_profile(
        manual_audiogram,
        sample_rate=sample_rate,
        device_profile=default_device_profile,
        source="manual_cli",
    )
    return profile, "manual"


def audiogram_tensor_from_profile(profile: HearingProfile) -> torch.Tensor:
    return profile.to_tensor()


def resolve_audiogram_tensor(
    profile_json: str | None,
    manual_audiogram: str | None,
    *,
    logger=None,
) -> tuple[torch.Tensor, str]:
    profile, source = resolve_profile_input(profile_json, manual_audiogram, logger=logger)
    return profile.to_tensor(), source


def print_profile_summary(profile: HearingProfile) -> str:
    return profile.profile_summary()


def save_profile_plot(profile: HearingProfile, output_path: str | Path) -> Path:
    validate_profile(profile)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    errs = profile.uncertainty if profile.uncertainty else None
    ax.errorbar(profile.frequencies, profile.thresholds_db, yerr=errs, marker="o", capsize=4)
    ax.set_xscale("log")
    ax.set_xticks(profile.frequencies)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Threshold (dB HL)")
    ax.set_title("Estimated Hearing Profile")
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
