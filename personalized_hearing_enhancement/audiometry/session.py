from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from personalized_hearing_enhancement.audiometry.stimuli import STANDARD_FREQS_HZ


@dataclass
class FrequencyState:
    frequency_hz: int
    current_db_hl: float
    step_size_db: float
    reversals: int = 0
    max_trials_reached: bool = False
    complete: bool = False
    threshold_estimate_db_hl: float | None = None
    uncertainty_db: float | None = None
    trials: list[dict[str, float | bool]] = field(default_factory=list)

    # Bayesian tracking fields (Phase 5A/5B1)
    posterior_grid_db_hl: list[float] = field(default_factory=list)
    posterior_probs: list[float] = field(default_factory=list)
    posterior_mean_db_hl: float | None = None
    posterior_variance_db2: float | None = None
    posterior_entropy: float | None = None
    posterior_map_db_hl: float | None = None
    posterior_ci95: list[float] | None = None
    inconsistency_count: int = 0


@dataclass
class AudiometrySession:
    frequencies_hz: list[int] = field(default_factory=lambda: list(STANDARD_FREQS_HZ))
    start_amplitude_db_hl: float = 40.0
    initial_step_size_db: float = 10.0
    min_step_size_db: float = 2.0
    max_trials_per_frequency: int = 18
    max_reversals: int = 4
    seed: int | None = None
    source: str = "interactive"
    session_started_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    active_frequency_index: int = 0
    active_amplitude_db_hl: float | None = None
    frequencies: dict[int, FrequencyState] = field(default_factory=dict)
    joint_inference_state: dict = field(default_factory=dict)

    # Reliability summary fields (Phase 5B1)
    total_trials: int = 0
    inconsistency_count: int = 0
    reliability_score: float = 1.0
    lapse_rate_used: float = 0.0
    guess_rate_used: float = 0.5
    fatigue_enabled: bool = False
    low_confidence: bool = False
    device_gain_mean_db: float | None = None
    device_gain_variance_db2: float | None = None
    device_gain_entropy: float | None = None

    def __post_init__(self) -> None:
        if not self.frequencies:
            self.frequencies = {
                f: FrequencyState(
                    frequency_hz=f,
                    current_db_hl=float(self.start_amplitude_db_hl),
                    step_size_db=float(self.initial_step_size_db),
                )
                for f in self.frequencies_hz
            }

    @property
    def active_frequency_hz(self) -> int:
        return self.frequencies_hz[self.active_frequency_index]

    def state_for(self, frequency_hz: int) -> FrequencyState:
        return self.frequencies[frequency_hz]

    def is_complete(self) -> bool:
        return all(state.complete for state in self.frequencies.values())

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["frequencies"] = {str(k): v for k, v in payload["frequencies"].items()}
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "AudiometrySession":
        payload = dict(payload)
        raw_freq = payload.get("frequencies", {})
        payload["frequencies"] = {int(k): FrequencyState(**v) for k, v in raw_freq.items()}
        payload.setdefault("total_trials", 0)
        payload.setdefault("inconsistency_count", 0)
        payload.setdefault("reliability_score", 1.0)
        payload.setdefault("lapse_rate_used", 0.0)
        payload.setdefault("guess_rate_used", 0.5)
        payload.setdefault("fatigue_enabled", False)
        payload.setdefault("low_confidence", False)
        payload.setdefault("active_amplitude_db_hl", None)
        payload.setdefault("joint_inference_state", {})
        payload.setdefault("device_gain_mean_db", None)
        payload.setdefault("device_gain_variance_db2", None)
        payload.setdefault("device_gain_entropy", None)
        return cls(**payload)
