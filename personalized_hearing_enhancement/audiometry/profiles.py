from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

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


def audiogram_tensor_from_profile(profile: HearingProfile) -> torch.Tensor:
    validate_profile(profile)
    return torch.tensor([profile.thresholds_db], dtype=torch.float32)


def parse_manual_audiogram(audiogram: str) -> list[float]:
    values = [float(x.strip()) for x in audiogram.split(",") if x.strip()]
    if len(values) != len(STANDARD_FREQS_HZ):
        raise ValueError(f"Manual audiogram must contain {len(STANDARD_FREQS_HZ)} values; got {len(values)}")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Manual audiogram contains non-finite values")
    return values


def resolve_audiogram_tensor(
    profile_json: str | None,
    manual_audiogram: str | None,
    *,
    logger=None,
) -> tuple[torch.Tensor, str]:
    if profile_json:
        profile = load_profile(profile_json)
        if manual_audiogram and logger is not None:
            logger.warning("Both --profile-json and --audiogram were provided; --profile-json takes precedence.")
        return audiogram_tensor_from_profile(profile), f"profile_json:{profile_json}"

    if manual_audiogram is None:
        raise ValueError("Either profile_json or manual_audiogram must be provided")
    values = parse_manual_audiogram(manual_audiogram)
    return torch.tensor([values], dtype=torch.float32), "manual"


def print_profile_summary(profile: HearingProfile) -> str:
    pairs = ", ".join(f"{f}Hz:{t:.1f}dB" for f, t in zip(profile.frequencies, profile.thresholds_db, strict=True))
    return (
        f"Profile source={profile.source}, timestamp={profile.timestamp_utc}\n"
        f"Device profile={profile.device_profile or 'N/A'}\n"
        f"Thresholds: {pairs}"
    )


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
